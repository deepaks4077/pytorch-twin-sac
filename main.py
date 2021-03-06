import argparse
import os
import random
import time

import gym
import imageio
import numpy as np
import torch
from skimage.transform import resize

import SAC
import utils


# Runs policy for X episodes and returns average reward
def evaluate_policy(policy,
                    total_timesteps,
                    eval_episodes=10,
                    render=False,
                    skip_frame=10):
    avg_reward = 0.
    for i in range(eval_episodes):
        obs = env.reset()

        if render and i == 0:
            frames = [env.render(mode='rgb_array').copy()]

        done = False
        t = 0
        while not done:
            t += 1
            action = policy.select_action(np.array(obs))
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
            if render and i == 0 and t % skip_frame == 0:
                frames.append(env.render(mode='rgb_array').copy())

    avg_reward /= eval_episodes

    if render:
        frames = [(resize(frame, (256, 256)) * 255).astype('uint8')
                  for frame in frames]
        filename = os.path.join('videos', '%d.mp4' % total_timesteps)
        imageio.mimsave(filename, frames)

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
        "--save_videos",
        action="store_true")  # Whether or not evaluation vides are saved
    parser.add_argument(
        "--print_fps", action="store_true")  # Whether or not print fps
    parser.add_argument(
        "--batch_size", default=100,
        type=int)  # Batch size for both actor and critic
    parser.add_argument(
        "--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument(
        "--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument(
        "--initial_temperature", default=0.2, type=float)  # SAC temperature
    parser.add_argument(
        "--learn_temperature",
        action="store_true")  # Whether or not learn the temperature
    parser.add_argument(
        "--policy_freq", default=2,
        type=int)  # Frequency of delayed policy updates
    parser.add_argument(
        "--normalize_returns", action="store_true")  # Normalize returns
    parser.add_argument("--linear_lr_decay", action="store_true")  # Decay lr
    args = parser.parse_args()

    if args.normalize_returns and args.initial_temperature != 0.01:
        print("Please use temperature of 0.01 for normalized returns")

    file_name = "%s_%s" % (args.env_name, str(args.seed))
    print("---------------------------------------")
    print("Settings: %s" % (file_name))
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_models and not os.path.exists("./pytorch_models"):
        os.makedirs("./pytorch_models")
    if args.save_videos and not os.path.exists("./videos"):
        os.makedirs("./videos")

    if args.env_name.startswith('DM'):
        import dmc_registration

    env = gym.make(args.env_name)

    # Set seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env.seed(args.seed)

    if torch.cuda.is_available():
        torch.set_num_threads(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize policy
    policy = SAC.SAC(state_dim, action_dim, max_action,
                     args.initial_temperature)

    replay_buffer = utils.ReplayBuffer(norm_ret=args.normalize_returns)

    # Evaluate untrained policy
    evaluations = [evaluate_policy(policy, 0, render=args.save_videos)]

    total_timesteps = 0
    timesteps_since_eval = 0
    episode_num = 0
    done = True

    if args.print_fps:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prev_time = time.time()
        prev_eval_timesteps = 0

    while total_timesteps < args.max_timesteps:

        if args.linear_lr_decay:
            policy.set_lr(
                1e-3 * (1 - float(total_timesteps) / args.max_timesteps))

        if done:
            if total_timesteps != 0:
                if args.print_fps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    fps = (total_timesteps - prev_eval_timesteps) / (
                        time.time() - prev_time)
                    print((
                        "Total T: %d FPS %d Episode Num: %d Episode T: %d Reward: %f"
                    ) % (total_timesteps, fps, episode_num, episode_timesteps,
                         episode_reward))
                else:
                    print(
                        ("Total T: %d Episode Num: %d Episode T: %d Reward: %f"
                         ) % (total_timesteps, episode_num, episode_timesteps,
                              episode_reward))

            # Evaluate episode
            if timesteps_since_eval >= args.eval_freq:
                timesteps_since_eval %= args.eval_freq
                evaluations.append(
                    evaluate_policy(
                        policy, total_timesteps, render=args.save_videos))

                if args.save_models:
                    policy.save(file_name, directory="./pytorch_models")
                np.save("./results/%s" % (file_name), evaluations)

                if args.print_fps:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    prev_time = time.time()
                    prev_eval_timesteps = total_timesteps

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
            policy.train(replay_buffer, total_timesteps, args.batch_size,
                         args.discount, args.tau, args.policy_freq,
                         -action_dim if args.learn_temperature else None)

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
    evaluations.append(
        evaluate_policy(policy, total_timesteps, render=args.save_videos))
    if args.save_models:
        policy.save("%s" % (file_name), directory="./pytorch_models")
    np.save("./results/%s" % (file_name), evaluations)
