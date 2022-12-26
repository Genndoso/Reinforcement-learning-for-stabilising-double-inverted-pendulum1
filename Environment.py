import gym
from gym.spaces import Box

from Dynamics import state_to_coords, get_next_state, get_energy

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

    def __init__(self, init_state, dt=0.02, plotEnergy = False):
        self.action_space = ActionSpaceCartPole()
        self.observation_space = ObservationSpaceCartPole()
        self.state = init_state
        self.init_state = init_state
        self.dt = dt
        self.init_coords = state_to_coords(init_state)
        self.state_history = [self.init_state]
        self.plotEnergy = plotEnergy
        self.action_history = []
        print('Environment initialized')

    def _take_action(self, action):
        self.state = get_next_state(self.state, action, self.dt)

    def _reward_function(self):
        """

        # Reward system 1
        Check whether 1 and 2 cart pole are in angle range between 80 and 100 degrees
        agent will agent a reward in range [0, 1]
        else:
            If angle of pole 1 and 2  are lower than 80 and higher than 100 degrees, therefore, it makes sense
            to terminate the environment and reset/restart.
            agent will get a  reward = -100

        # Reward system 2
        If cart is out of a given range of x = [-2, 2] then terminates the environment and
         penalize the system heavily of a penalty = -100 and system is done here.


        # Reward system 3
        Penalize the system if spinning to fast


        """
        done = False
        state = self.state
        reward = 0
        # degree reward
        normalized_angle_1 = np.degrees((state[1]))
        normalized_angle_2 = np.degrees((state[2]))

        #
        if normalized_angle_1 > 80 and normalized_angle_1 < 95:
            reward = 1 - (90 - normalized_angle_1) * 0.01
            if normalized_angle_2 > 85 and normalized_angle_2 < 95:
                reward += reward + 1 - (90 - normalized_angle_2) * 0.01
            reward *= 4

        else:

            reward = -100
            done = True

      #  another degree reward system
      #   cost = 2*(normalize_angle(state[1])/2 - np.pi/2) + \
      #                  2*(normalize_angle(state[2])/2 - np.pi/2)
      #
      #   reward = -np.abs(cost)

        # another degree_reward system

        #         deg_reward = ((np.sin(state[1]))*10 + (np.sin(state[2]))*10)/2
        #         #if np.sin(state[1]
        #         reward += deg_reward
        #         print(state[1])

        # distance penalty

        if state[0] > 1 or state[0] < -1:
            reward -= 100
            done = True


        # distance2 rew
        # state_coords = state_to_coords(state)
        #        # dist_pen = (state_coords[0][1] - state_coords[0][0])**2 +  (state_coords[0][2] - state_coords[0][0])**2
        # dist_rew =  -( state_coords[1][1] - self.init_coords[1][1]) -  ( state_coords[1][2] - self.init_coords[1][2])*2
        # reward -= dist_rew


        vel_reward = np.abs(state[4]*10)  #minus points - we dont want it to spin super fast
        reward -= vel_reward
       # print(state[4]*10)

        return reward, done

    def reward_function3(self):
        goal = np.array([0, 0])
        coords = np.array(state_to_coords(self.state)[:, 0])
        reward = max(np.linalg.norm(coords - goal), 0.0001)
        reward = 1 / reward
        return reward, done

    def _reward_function2(self):
        final_node_coords = np.array(state_to_coords(self.state)[:, -1])
        goal_coords = np.array(state_to_coords([0, np.pi / 2, np.pi / 2])[:, -1])

        reward = max(np.linalg.norm(final_node_coords - goal_coords), 0.0001)
        reward = 1 / reward

        done = False
        #print(reward)
        vel_reward = np.abs(self.state[4] * 5)  # minus points - we dont want it to spin super fast
        reward -= vel_reward
        #print(vel_reward)

        normalized_angle_1 = np.degrees(self.state[1])
        normalized_angle_2 = np.degrees(self.state[2])

        if normalized_angle_2 < 85 or normalized_angle_2 > 95:
            reward -= 100
            done = True


        if self.state[0] < 2 and self.state[0] > -2:
            pass
        else:
            reward -= 100
            done = True

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
        reward, done = self._reward_function()
        return np.array(self.state), reward, done, info
    
    def animate(self,i,line,energy_text):
        """perform animation step"""
        XY = state_to_coords(self.state_history[i])
        line.set_data(XY[0],XY[1])
        if self.plotEnergy:
            en = get_energy(self.state_history[i])
            energy_text.set_text(f'energy = {en}')
        return line,
    
    def render(self):
        """
        Compute the render frames as specified by render_mode attribute during initialization of the environment.
        """
        fig = plt.figure(figsize=(10, 3), dpi=200)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,xlim=(-8, 8), ylim=(-3, 3))
        ax.grid()
        line, = ax.plot([], [], 'o-', lw=1)
        
        energy_text = None
        if self.plotEnergy:
            energy_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)    
        animation_func = lambda i: self.animate(i,line,energy_text)
        
        ani = animation.FuncAnimation(fig, animation_func, frames=len(self.state_history),interval=20, blit=True)
        return ani

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.
        """

        self.state = self.init_state
        self.action_history = []

        self.state_history = [self.init_state]
        done = False

        return np.array(self.state), done




