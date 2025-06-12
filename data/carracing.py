"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gymnasium as gym
import numpy as np
from utils.misc import sample_continuous_policy

def generate_random_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    # Initialize environment
    env = gym.make("CarRacing-v3") # obs: (96, 96, 3 RGB), continuous action: steering [-1, 1], throttle [0, 1], brake [0, 1]
    seq_len = 1000

    for i in range(rollouts):
        # Reset environment
        env.reset()

        # Generate a sequence of actions
        if noise_type == 'white': # Random actions
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown': # Random brownian motion
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        # Collect obs (s), rewards (r) and done (d) flags
        s_rollout = []
        r_rollout = []
        d_rollout = []

        # Loop through scenario
        t = 0
        while True:
            action = a_rollout[t]
            t += 1

            s, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]

            # Stops when done = True (crash, completion, or timeout)
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)), # Save rollout as rollout_0.npz, rollout_1.npz, etc.
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

from utils.misc import RolloutGenerator
import torch
from utils.config import RSIZE, RED_SIZE
from torchvision import transforms

def generate_policy_data(mdir, device, time_limit, rollouts, data_dir):
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."

    # Initialize environment
    rgen= RolloutGenerator(mdir, device, time_limit)
    env = rgen.env

    # Transform from numpy 96x96x3 to 64x64x3 tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((RED_SIZE, RED_SIZE)),
        transforms.ToTensor()])

    for i in range(rollouts):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        done = False
        s_rollout, r_rollout, a_rollout, d_rollout = [], [], [], []
        hidden = [torch.zeros(1, RSIZE).to(device) for _ in range(2)]
        while not done:
            obs_t = obs
            obs_t = transform(obs_t).unsqueeze(0).to(device)
            action, hidden = rgen.get_action_and_transition(obs_t, hidden)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result

            s_rollout.append(obs)
            r_rollout.append(reward)
            a_rollout.append(action)
            d_rollout.append(done)

        print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
        np.savez(join(data_dir, 'rollout_{}'.format(i)), # Save rollout as rollout_0.npz, rollout_1.npz, etc.
                    observations=np.array(s_rollout),
                    rewards=np.array(r_rollout),
                    actions=np.array(a_rollout),
                    terminals=np.array(d_rollout))

# Temporarily reload old VAE, MDNRNN and Controller models to exp_dir
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    # generate_random_data(args.rollouts, args.dir, args.policy)
    generate_policy_data('exp_dir', 'cuda:0', 1000, args.rollouts, args.dir)
