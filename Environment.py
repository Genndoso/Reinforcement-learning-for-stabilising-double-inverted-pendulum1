import gym
from gym.spaces import Box
from Dynamics import state_to_coords, get_next_state, normalize_angle
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
# normalized angle

class ObservationSpaceCartPole():
    def __init__(self):
        self.shape = (6,)


class ActionSpaceCartPole():
    def __init__(self):
        self.shape = (1,)
        self.bounds = (-2, 2)


class DoublePendulumEnv(gym.Env):

    def __init__(self, init_state, dt=0.02, max_initial_angle=3 * 2 * np.pi / 360):
        self.action_space = 1
        self.observation_space = 6
        self.state = init_state
        self.init_state = init_state
        self.dt = dt
        print('Environment initialized')
        self.init_coords = state_to_coords(init_state)
        self.max_initial_angle = max_initial_angle
        self.state_history = [self.init_state]
        self.action_history = []

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

        if normalized_angle_1 > 87 and normalized_angle_1 < 93:
            reward = 1 - (90 - normalized_angle_1) * 0.01
            if normalized_angle_2 > 87 and normalized_angle_2 < 93:
                reward += reward + 1 - (90 - normalized_angle_2) * 0.01
            reward *= 4

        else:
            reward = -100
            done = True

        # another degree reward system
        # cost = 2*(normalize_angle(state[1]) - np.pi/2) + \
        #                2*(normalize_angle(state[2]) - np.pi/2)

        # reward = -np.abs(cost)

        # another degree_reward system

        #         deg_reward = ((np.sin(state[1]))*10 + (np.sin(state[2]))*10)/2
        #         #if np.sin(state[1]
        #         reward += deg_reward
        #         print(state[1])

        # distance penalty
        if state[0] < 2 and state[0] > -2:
            pass
        else:
            reward -= -100
            done = True

        # distance2 rew
        # state_coords = state_to_coords(state)
        #        # dist_pen = (state_coords[0][1] - state_coords[0][0])**2 +  (state_coords[0][2] - state_coords[0][0])**2
        # dist_rew =  -( state_coords[1][1] - self.init_coords[1][1]) -  ( state_coords[1][2] - self.init_coords[1][2])*2
        # reward -= dist_rew

        # if np.abs(state[4])>0.5:
        #     vel_reward = 1000   #minus points - we dont want it to spin super fast
        #     reward -= vel_reward

        return reward, done

    def reward_function3(self):
        goal = np.array([-1, 0])
        coords = np.array(state_to_coords(self.state)[:, 0])
        reward = max(np.linalg.norm(coords - goal), 0.0001)
        reward = 1 / reward

        done = False

        return reward, done

    def reward_function2(self):
        final_node_coords = np.array(state_to_coords(self.state)[:, -1])
        goal_coords = np.array(state_to_coords([0, np.pi / 2, np.pi / 2])[:, -1])

        reward = max(np.linalg.norm(final_node_coords - goal_coords), 0.0001)
        reward = 1 / reward

        done = False

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
        self.state_history.append(self.state)
        self.action_history.append(action)
        reward, done = self.reward_function3()
        return np.array(self.state), reward, done, info

    def animate(self, i, line):
        """perform animation step"""
        XY = state_to_coords(self.state_history[i])
        line.set_data(XY[0], XY[1])
        return line,

    def render(self):
        """
        Compute the render frames as specified by render_mode attribute during initialization of the environment.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-4, 4), ylim=(-2, 2))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=2)
        animation_func = lambda i: self.animate(i, line)

        ani = animation.FuncAnimation(fig, animation_func, frames=len(self.state_history), interval=20, blit=True)
        return ani

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        """

        self.state = self.init_state
        self.action_history = []

        self.state[1] = np.pi / 2 + np.random.uniform(-self.max_initial_angle, self.max_initial_angle)
        self.state[2] = np.pi / 2 + np.random.uniform(-self.max_initial_angle, self.max_initial_angle)

        self.state_history = [self.init_state]
        done = True

        return np.array(self.state), done




