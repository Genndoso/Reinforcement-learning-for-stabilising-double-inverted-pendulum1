# Reinforcement learning for stabilising double inverted pendulum
This is a final project of Reinforcement learning course at skoltech which is devoted for stabilising double inverted pendulum


# 

Algorithms which was used
- Proximal policy optimization (PPO)
- Model predictive control (MPC)
- Deep double Q-network (DDQN)

## Folders and Files Description

### Folders and files

|Folder name       |                     Description                                    |
|------------------|--------------------------------------------------------------------|
|`1_pole_inverted_pendulum`             |  Source code for simple case of inverted pendulum                                           |
|`DDQN`            | Results of training deep double Q-network for stabilising double inverted pendulum                              |
|`MPC`          | Model predictive control using Casadi optimization                 |
|`MPC`          |  Results of training Proximal policy optimization for stabilising double inverted pendulum                 |


### Files

|File name            |                     Description                                    |
|---------------------|--------------------------------------------------------------------|
|`Dynamics.py`            | Double inverted pendulum dynamics which was written on Casadi    |
|`Environment.py`          | Containing the SAc agent class                                     |
|`networks.py`        | Networks in used by agents (Actor, Critic and Value networks)      |
|`utils.py`           | General utility functions                                          |
|`buffer.py`          | A replay buffer class, used for offline training                   |


You can see below the learning curves along with gifs of agents  play the Inverted Double Pendulum and Inverted Pendulum Swing environment.
## Proximal policy optimization 
Episode rewards curves:

Random initial state (90 +- 3 degrees)
![Random_initial_state](https://user-images.githubusercontent.com/53058704/197342315-3c3afa99-9ba3-4a4e-b0fa-f3119e7fe339.png)


Fixed initial state (90 degrees)
![Fixed initial state](https://user-images.githubusercontent.com/53058704/197342307-4e591c1a-606b-4fc1-aacc-9076d766bb12.png)


Agent after 100k timesteps of training
https://user-images.githubusercontent.com/53058704/197342248-bbffe58d-8622-4905-a9ad-700115f19b6b.mp4

Agent after 500k timesteps of training
https://user-images.githubusercontent.com/53058704/197342203-baab4294-b834-4a31-b5dc-99aec6f2a4aa.mp4

Agent after 1000k timesteps of training
https://user-images.githubusercontent.com/53058704/197342264-83ee60cb-3d8e-4172-a191-b5023569f3ab.mp4



## Model predictive control



## How to use

