import gym
from gym.spaces import Box
from Dynamics import state_to_coords, get_next_state
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
# normalized angle

def normalize_angle(angle):
    """
    3*pi gives -pi, 4*pi gives 0 etc, etc. (returns the negative difference
    from the closest multiple of 2*pi)
    """
    normalized_angle = abs(angle)
    normalized_angle = normalized_angle % (2*np.pi)
    if normalized_angle > np.pi:
        normalized_angle = normalized_angle - 2*np.pi
    normalized_angle = abs(normalized_angle)
    return normalized_angle



class DoublePendulumEnv(gym.Env):

    def __init__(self, init_state, dt=0.02, max_initial_angle = 3 * 2 * np.pi / 360):
        self.action_space = Box(low=-10, high=10)
        self.observation_space = 6
        self.state = init_state
        self.init_state = init_state
        self.dt = dt
        print('Environment initialized')
        self.init_coords = state_to_coords(init_state)
        self.max_initial_angle = max_initial_angle
    def _take_action(self, action):
        self.state = get_next_state(self.state, action, self.dt)

    def _reward_function(self, done):
        """

        # Reward system 1
        Check whether 1 and 2 cart pole are in angle range between 80 and 100 degrees
        agent will agent a reward in range [0, 1]

        else:
            If angle of pole 1 and 2  are greater than 10 degrees, therefore, it makes sense
            to terminate the environment and reset/restart.
            agent will get a  reward = -1

        # Reward system 2
        If cart is in given range of x = [-5, 5] then agent will get a reward 0.5 every steps.
        Otherwise, it penalies the system heavily of a penalty = -50 and system is done here.

        # Reward system 3
        # this is unused. This an analog to reward system 1 but for coordinates
        If cart pole is not in the same line with cart then it will give additional penalty


        # Reward system 4
        Velocity penalty (halves the reward if spinning too fast)


        """
        state = self.state
        reward = 0
        # degree reward
        normalized_angle_1 = np.degrees(normalize_angle(state[1]))
        normalized_angle_2 = np.degrees(normalize_angle(state[2]))

        # if normalized_angle_1 > 80 and normalized_angle_1 < 100:
        #     reward = 1 - (90 - normalized_angle_1) * 0.01
        #     if normalized_angle_2 > 80 and normalized_angle_2 < 100:
        #         reward += reward + 1 - (90 - normalized_angle_2) * 0.01
        #         reward *= 0.5
        # #             if np.abs(np.degrees(state[2])) < 100:
        # #                 reward = reward + 9 - (90 - np.degrees(state[2]))*0.1
        # #                 reward *= 1
        # else:
        #     reward = -1
        #     done = True

       # another degree reward system
        cost = 2*(normalize_angle(state[1]) - np.pi/2) + \
                       2*(normalize_angle(state[2]) - np.pi/2)

        reward = -np.abs(cost)

        # another degree_reward system

        #         deg_reward = ((np.sin(state[1]))*10 + (np.sin(state[2]))*10)/2
        #         #if np.sin(state[1]
        #         reward += deg_reward
        #         print(state[1])

        # distance penalty
        if state[0] < 5 and state[0] > -5:
            reward += 0.5
        else:
            reward -= -50
            done = True

        # distance2 rew
        #         state_coords = state_to_coords(state)
        #        # dist_pen = (state_coords[0][1] - state_coords[0][0])**2 +  (state_coords[0][2] - state_coords[0][0])**2
        #         dist_rew =  self.init_coords[1][1] + self.init_coords[1][2] + ( state_coords[1][1] - self.init_coords[1][1])*5 +  ( state_coords[1][2] - self.init_coords[1][2])*5
        #         reward += dist_pen

       # velocity penalty
        vel_pen = ((1 + np.exp(-0.5 * state[-3:] ** 2)) / 2).sum()/10
        reward -= vel_pen

        return reward, done

    def step(self, action):
        """
        observation -  [x,phi,theta,dx,dphi,dtheta]
        Num     Observation               Min                     Max
        0       Cart Position             -5 m                  5 m
        1       Pole1 Angle               -pi                     +pi
        2       Pole2 Angle               -pi                     +pi
        3       Cart Velocity             -Inf                    Inf
        4       Pole1 Angular Velocity    -Inf                    Inf
        5       Pole1 Angular Velocity    -Inf                    Inf

        """
        done = False
        info = {}
        self._take_action(action)

        # alive_rew = 10
        reward, done = self._reward_function(done)
        return np.array(self.state), reward, done, info

    def render(self):
        """
        Compute the render frames as specified by render_mode attribute during initialization of the environment.

        """
        state = self.state
        ani = animation.FuncAnimation(fig, animate, frames=300,
                                      interval=20, blit=True, init_func=init)
        plt.show()

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        """
        self.rew_sum = 0
        self.state = self.init_state

        self.state[1] = np.pi/2 + np.random.uniform(-self.max_initial_angle, self.max_initial_angle)
        self.state[2] = np.pi/2 + np.random.uniform(-self.max_initial_angle, self.max_initial_angle)

        return np.array(self.state)

