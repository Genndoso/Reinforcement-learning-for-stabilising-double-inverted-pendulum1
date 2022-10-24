# Reinforcement learning for stabilising double inverted pendulum
This is a final project of Reinforcement learning course at skoltech which is devoted for stabilising double inverted pendulum
by Kovalev.V.V and  Maximilian.P


### We pursued several problems:
* stabilizing cart pole with the horizontal axis, when initstate  = $[0,\pi / 2 + \sigma,\pi / 2 + \sigma,0,0,0]$
* force the swing up of pendulum


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
|`DDQN`            | Results of training deep double Q-network                               |
|`MPC`          | Model predictive control using Casadi optimization                 |
|`PPO`          |  Results of training Proximal policy optimization              |
|`SAC`          |  Results of training Soft Actor critic                 |


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
<p align="center">
<img src="https://user-images.githubusercontent.com/53058704/197342315-3c3afa99-9ba3-4a4e-b0fa-f3119e7fe339.png" width="500">
</p>


Fixed initial state (90 degrees)
<p align="center">
<img  src="https://user-images.githubusercontent.com/53058704/197342642-01feb722-0eac-4e39-8b77-87731752b208.png" width="500">
</p>


Agent after 100k timesteps of training
![PPO_100K](https://user-images.githubusercontent.com/53058704/197342394-5273b20d-a462-4ffc-bd20-7fb08159e4ed.gif)


Agent after 500k timesteps of training
![PPO_500K](https://user-images.githubusercontent.com/53058704/197342398-55ca8314-a958-4e78-b3b6-6c6901a39e16.gif)


Agent after 1000k timesteps of training
![2022-10-24-19-20-36](https://user-images.githubusercontent.com/53058704/197576432-e5773292-9b62-4f3f-821b-2555f7183c58.gif)



## Model predictive control
Balancing task
![balancing](https://github.com/Genndoso/Reinforcement-learning-for-stabilising-double-inverted-pendulum1/blob/Slava/T2N14Steps500.gif)


Swing up
![swingup](https://github.com/Genndoso/Reinforcement-learning-for-stabilising-double-inverted-pendulum1/blob/Slava/T2N14x0.1.gif)


## How to use
For each algorithm used you can try to launch and test the provided jupyter notebooks

All PPO the trained models are stored in `PPO/PPO3_Trained`. In order check results launch 'Visualization.ipynb'


