"""forward.py

"""

import numpy as np
import itertools as it
import gym

from numba import jit


#@jit("void(float64[:], int64[:])", nopython=True)
@jit(nopython=True)
def _forward_inplace(state, u):
    gravity = 9.8
    masscart = 1.0
    masspole = 0.1
    total_mass = (masspole + masscart)
    length = 0.5 # actually half the pole's length
    polemass_length = (masspole * length)
    force_mag = 10.0
    tau = 0.02  # seconds between state updates

    theta_threshold_radians = 12 * 2 * np.pi / 360
    x_threshold = 2.4

    x, x_dot, theta, theta_dot = state
    cost = 0.0
    oob = False
    pwr = 1.25
    amp = 1.0
    for k in range(len(u)):
        uk = u[k]
        force = force_mag if uk==1 else -force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + polemass_length * theta_dot * theta_dot * sintheta) / total_mass
        thetaacc = (gravity * sintheta - costheta* temp) / (length * (4.0/3.0 - masspole * costheta * costheta / total_mass))
        xacc  = temp - polemass_length * thetaacc * costheta / total_mass
        x = x + tau * x_dot
        x_dot = x_dot + tau * xacc
        theta = theta + tau * theta_dot
        theta_dot = theta_dot + tau * thetaacc
        cost += amp * (x*x + x_dot*x_dot + theta*theta + theta_dot*theta_dot)
        done =  x < -x_threshold \
                or x > x_threshold \
                or theta < -theta_threshold_radians \
                or theta > theta_threshold_radians
        done = bool(done)
        oob = oob or done
        amp *= pwr

    state[0] = x
    state[1] = x_dot
    state[2] = theta
    state[3] = theta_dot

    return cost, done



def forward(state, u):
    """Advance the forward model len(u) steps forward starting
    from x0

    See https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    """

    state = np.array(state)
    return _forward_inplace(state, u)


def make_sequence(state, lookahead=14):
    best_inp, best_cost = None, None
    for inp in it.product([0, 1], repeat=lookahead):
        inp = np.array(inp, dtype=int)
        cost, done = forward(state, inp)
        if done:
            cost = max(cost, 1e15)
        if best_cost is None or cost < best_cost:
            best_inp, best_cost = inp, cost
    return best_inp


def main():

    #state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
    #best_inp = make_sequence(state)
    #print(forward(state, best_inp))

    env = gym.make('CartPole-v0')

    u_ptr = None
    u = None
    take_max = 3

    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()

            if u is None or u_ptr >= min(len(u), take_max):
                u = make_sequence(observation)
                u_ptr = 0
                cost, done = forward(observation, u)


            observation, reward, done, info = env.step(u[u_ptr])
            u_ptr += 1
            print("reward, done, info")
            print(reward, done, info)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break



if __name__ == '__main__':
    main()


