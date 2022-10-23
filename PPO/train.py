from .Proximal_Policy_Optimization import PPO, unscaled_action
from Environment import DoublePendulumEnv
import numpy as np
import os
from .config_PPO import config

state = [0, np.pi/2, np.pi/2, 0, 0, 0]

def train():
    print("============================================================================================")
    # max. timestep per episode. For DoubleCartPoleEnv, time constraint is 200 timesteps. After that environment is
    # reset.
    max_ep_len = config['max_episode_length']
    # The training phase will sample and update for 1 million timestep.
    max_training_steps = int(config['max_training_steps'])

    # In order, to check ongoing progress, average reward is printed at every 10_000 timesteps.
    print_freq = 10_000

    # Saving model parameters at every 1_00_000 timesteps.
    save_model_freq = int(config['save_model_freq'])

    action_std = config['action_std']  # Initial standard deviation.
    action_std_decay_rate = config['action_std_decay_rate']  # Decay rate of standard deviation.
    min_action_std = config['min_action_std']  # Threshold standard deviation.
    action_std_decay_freq = int(2e5)  # Decay the standard deviation every 200000 timesteps

    update_timestep = config['update_timestep']  # set old_policy parameters to new_policy parameters.
    K_epochs = config['max_number_of_epoch']  # Number of epochs before updating old policy parameters.
    eps_clip = config['eps_clip']  # clip range for surrogate loss function.
    gamma = config['gamma']  # Discount factor.

    lr_actor = config['lr_actor']  # Learning rate for optimizer of actor network.
    lr_critic = config['lr_critic']  # Learning rate for optimizer of critic network.
    env_name = 'DoubleInvPendulum'
    print("Training Environment:" + env_name)
    env = DoublePendulumEnv(init_state=state, dt=config['dt'])

    observation_shape = env.observation_space.shape[0]  # Observation shape
    action_shape = env.action_space.shape[0]  # Action shape

    directory = "PPO2_Trained"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)


    checkpoint_path = directory + "PPO2_{}.pth".format(env_name)


    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    print("============================================================================================")

    agent = PPO(observation_shape,
                action_shape,
                lr_actor,
                lr_critic,
                gamma,
                K_epochs,
                eps_clip,
                action_std)


    print("============================================================================================")

    # To keep track of the progress
    print_running_reward = 0
    print_running_episodes = 0

    time_step = 0
    i_episode = 0
    counter = 0

    plot_episode = []
    plot_reward = []

    while time_step <= max_training_steps:
        obs, done = env.reset()
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            action = agent.select_action(obs)  # Get action under old_policy given state.
            action = unscaled_action(action)  # Unscale the action.
            obs, reward, done, _ = env.step(action)  # Apply the action to environment.

            # Append the reward and done flag to buffer for calculating Monte Carlo returns during updating phase.
            agent.buffer.rewards.append(reward)
            agent.buffer.dones.append(done)

            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                agent.update()

            if time_step % action_std_decay_freq == 0:
                # Decay standard deviation by 0.1.
                agent.decay_action_std(action_std_decay_rate, min_action_std)

            if time_step % print_freq == 0:
                # print average reward during 10_000 timesteps
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            if time_step % save_model_freq == 0:

                # Save the model parameters for test phase and tracking performance.
                print("--------------------------------------------------------------------------------------------")
                print("Saving model at : " + checkpoint_path)
                agent.save(checkpoint_path)
                print("Model saved")
                counter += 1
                inter_checkpoint = directory + "PPO_{}_{}00K.pth".format(env_name, counter)
                print("--------------------------------------------------------------------------------------------")
                print("Model parameters to check for intermediate performance saving:.")
                print("saving model at : " + inter_checkpoint)
                agent.save(inter_checkpoint)
                if counter == 10:
                    print(f"Intermediate model saved for {counter}M")
                else:
                    print(f"Intermediate model saved for {counter}00K")
                print("--------------------------------------------------------------------------------------------")

            if done:
                break

        print_running_reward += current_ep_reward

        # plot reward per 10 episode.
        if i_episode % 10 == 0:
            plot_episode.append(i_episode)
            plot_reward.append(current_ep_reward)

        print_running_episodes += 1

        i_episode += 1

    env.close()

    print("============================================================================================")

    return plot_episode, plot_reward