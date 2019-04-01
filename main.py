import argparse
import os
import time

import gym
import numpy as np
import torch

import SAC
import utils


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy, eval_episodes=10):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print("Evaluation over %d episodes: %f" % (eval_episodes, avg_reward))
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env_name", default="HalfCheetah-v1")  # OpenAI gym environment name
    parser.add_argument(
        "--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument(
        "--start_timesteps", default=1e4,
        type=int)  # How many time steps purely random policy is run for
    parser.add_argument(
        "--eval_freq", default=5e3,
        type=float)  # How often (time steps) we evaluate
    parser.add_argument(
        "--max_timesteps", default=1e6,
        type=float)  # Max time steps to run environment for
    parser.add_argument(
        "--save_models",
        action="store_true")  # Whether or not models are saved
    parser.add_argument(
        "--batch_size", default=100,
        type=int)  # Batch size for both actor and critic
    parser.add_argument(
        "--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument(
        "--tau", default=0.005, type=float)  # Target network update rate
    args = parser.parse_args()

    file_name = "%s_%s" % (args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")

    env = gym.make(args.env_name)

    # Set seeds
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = SAC.SAC(state_dim, action_dim, max_action)

    replay_buffer = utils.ReplayBuffer()

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    start_time = time.time()
    while total_timesteps < args.max_timesteps:

        if done:
            total_time = time.time() - start_time
            fps = total_timesteps / total_time
            if total_timesteps != 0:
                print((
                    "Total T: %d FPS: %d Episode Num: %d Episode T: %d Reward: %f"
                ) % (total_timesteps, fps, episode_num, episode_timesteps,
                     episode_reward))
            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(evaluate_policy(policy))

                if args.save_models:
                    policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

            # Reset environment
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        # Select action randomly or according to policy
        if total_timesteps < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = policy.sample_action(np.array(obs))

        if total_timesteps > 1e3:
            policy.train(replay_buffer, args.batch_size, args.discount,
                         args.tau)

        # Perform action
        new_obs, reward, done, _ = env.step(action)
        done_bool = 0 if episode_timesteps + 1 == env._max_episode_steps else float(
            done)
        episode_reward += reward

        # Store data in replay buffer
        replay_buffer.add((obs, new_obs, action, reward, done_bool))

        obs = new_obs

        episode_timesteps += 1
        total_timesteps += 1
        timesteps_since_eval += 1

    # Final evaluation
    evaluations.append(evaluate_policy(policy))
    if args.save_models:
        policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)
