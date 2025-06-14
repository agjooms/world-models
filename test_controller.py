""" Test controller """
import argparse
from os.path import join, exists
from utils.misc import RolloutGenerator, RolloutGenerator2, RolloutGenerator3
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--logdir', type=str, help='Where models are stored.')
args = parser.parse_args()

ctrl_file = join(args.logdir, 'ctrl', 'best.tar')

assert exists(ctrl_file),\
    "Controller was not trained..."

device = torch.device('cpu')

# generator = RolloutGenerator(args.logdir, device, 1000)
# generator = RolloutGenerator2(args.logdir, device, 1000) # New
generator = RolloutGenerator3(args.logdir, device, 1000)

with torch.no_grad():
    generator.rollout(None, save_video=True)
