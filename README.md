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
|`SAC_notebook.ipynb` | A SAC notebook to run on Colab with GPU                            |


## Proximal policy optimization 


## Model predictive control



## How to use

