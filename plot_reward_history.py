import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from os.path import exists, join
import torch

def plot_reward_history(logdir):
    """ Plot the reward history from a CSV file. """
    ctrl_dir = join(logdir, 'ctrl')
    reward_history_file = join(ctrl_dir, 'reward_history.csv')

    if not exists(reward_history_file):
        print(f"Reward history file does not exist: {reward_history_file}")
        return

    df = pd.read_csv(reward_history_file)
    
    if df.empty:
        print("Reward history is empty.")
        return

    generations = range(len(df.index))
    mean_reward = df['mean_reward'].values
    best_reward = df['best_reward'].values
    worst_reward = df['worst_reward'].values

    plt.figure(figsize=(10, 5))
    plt.plot(generations, mean_reward, color='b')
    plt.plot(generations, best_reward, color='g')
    plt.plot(generations, worst_reward, color='r')
    plt.title('Reward History')
    plt.xlabel('Generation')
    plt.ylabel('Reward')
    plt.grid()
    plt.show()

def print_best_model_stats(logdir):
    """Print the evaluation and std of the best loaded model."""
    ctrl_dir = join(logdir, 'ctrl')
    ctrl_file = join(ctrl_dir, 'best.tar')
    if not exists(ctrl_file):
        print(f"No best model found at {ctrl_file}")
        return

    state = torch.load(ctrl_file, map_location='cpu', weights_only=False)
    reward = state.get('reward', None)
    std = state.get('std', None)
    print(f"Loaded best model from {ctrl_file}")
    if reward is not None:
        print(f"Best model evaluation: mean reward = {reward:.2f}", end='')
        if std is not None:
            print(f", std = {std:.2f}")
        else:
            print()
    else:
        print("No reward information found in the model file.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True, help='Where everything is stored.')
    args = parser.parse_args()

    logdir = args.logdir  # assuming models are in logdir

    print_best_model_stats(logdir)
    plot_reward_history(logdir)