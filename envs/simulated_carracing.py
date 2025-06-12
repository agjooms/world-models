"""
Simulated carracing environment.
"""
import argparse
from os.path import join, exists
import torch
from torch.distributions.categorical import Categorical
import gymnasium as gym
from gymnasium import spaces
from models.vae import VAE
from models.mdrnn import MDRNNCell
from utils.config import ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE

import numpy as np

# World Model (dream) environment: uses the world model as environment to move (Required methods for gym env: reset, step, render)
class SimulatedCarracing(gym.Env): # pylint: disable=too-many-instance-attributes
    """
    Simulated Car Racing.

    Gym environment using learnt VAE and MDRNN to simulate the
    CarRacing-v3 environment.

    :args directory: directory from which the vae and mdrnn are
    loaded.
    """
    def __init__(self, directory, time_limit=1000):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.time_limit = time_limit
        self.steps = 0

        # Check if trained models exist
        vae_file = join(directory, 'vae', 'best.tar')
        rnn_file = join(directory, 'mdrnn', 'best.tar')
        assert exists(vae_file), "No VAE model in the directory..."
        assert exists(rnn_file), "No MDRNN model in the directory..."

        # Define action and observation spaces
        self.action_space = spaces.Box(np.array([-1, 0, 0]), np.array([1, 1, 1]))
        self.observation_space = spaces.Box(low=0, high=255, shape=(RED_SIZE, RED_SIZE, 3),
                                            dtype=np.uint8)

        # Load VAE
        self.vae = VAE(3, LSIZE).to(self.device)
        vae_state = torch.load(vae_file, map_location=self.device)
        print("Loading VAE at epoch {}, "
              "with test error {}...".format(
                  vae_state['epoch'], vae_state['precision']))
        self.vae.load_state_dict(vae_state['state_dict'])
        self._decoder = self.vae.decoder # Extract decoder from VAE

        # Load MDRNN
        self._rnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(self.device)
        rnn_state = torch.load(rnn_file, map_location=self.device)
        print("Loading MDRNN at epoch {}, "
              "with test error {}...".format(
                  rnn_state['epoch'], rnn_state['precision']))
        rnn_state_dict = {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()}
        self._rnn.load_state_dict(rnn_state_dict)

        # print("VAE device:", next(self.vae.parameters()).device, flush=True)
        # print("MDRNN device:", next(self._rnn.parameters()).device, flush=True)

        # Initialize state
        self._lstate = torch.randn(1, LSIZE, device=self.device, dtype=torch.float32)  # Latent state
        self._hstate = 2 * [torch.zeros(1, RSIZE, device=self.device, dtype=torch.float32)]  # Hidden state

        # Initialize observation and visual observation
        self._obs = None
        self._visual_obs = None

        # Initialize rendering attributes
        self.monitor = None
        self.figure = None

    def reset(self):
        """ Resetting """
        # print("SimulatedCarracing.reset called") # Debugging
        import matplotlib.pyplot as plt
        self.steps = 0

        # Reset lstate and hstate
        self._lstate = torch.randn(1, LSIZE, device=self.device)
        self._hstate = 2 * [torch.zeros(1, RSIZE, device=self.device)]

        # Generate initial observation
        self._obs = self._decoder(self._lstate)
        np_obs = self._obs.cpu().detach().numpy() # added .detach() (new)
        np_obs = np.clip(np_obs, 0, 1) * 255
        np_obs = np.transpose(np_obs, (0, 2, 3, 1))
        np_obs = np_obs.squeeze()
        np_obs = np_obs.astype(np.uint8)
        self._visual_obs = np_obs

        # Initialize monitor if not already set
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
        
        return np_obs

    def step(self, action):
        """ One step forward """
        # print("SimulatedCarracing.step called") # Debugging
        with torch.no_grad():
            action = torch.tensor(action, device=self.device, dtype=torch.float32).unsqueeze(0) # Converts action (3,) to (1, 3) tensor

            # Pass action, lstate and hstate through RNN: return (mu, sigma, pi): params of Gaussian mixture for next lstate
            #                                                    r: predicted reward
            #                                                    d: done flag, n_h: next hidden state
            mu, sigma, pi, r_pred, d, n_h = self._rnn(action, self._lstate, self._hstate)

            # Sample mixture component index from Categorical distribution defined by pi
            pi = pi.squeeze()
            mixt = Categorical(torch.exp(pi)).sample().item()

            # Update lstate and hstate
            self._lstate = mu[:, mixt, :] # + sigma[:, mixt, :] * torch.randn_like(mu[:, mixt, :])
            self._hstate = n_h

            # Generate next observation using the decoder + process it for visualization
            self._obs = self._decoder(self._lstate)
            np_obs = self._obs.cpu().detach().numpy() # added .detach() (new)
            np_obs = np.clip(np_obs, 0, 1) * 255
            np_obs = np.transpose(np_obs, (0, 2, 3, 1))
            np_obs = np_obs.squeeze()
            np_obs = np_obs.astype(np.uint8)
            self._visual_obs = np_obs

            # Custom reward function
            # gas = float(action[0, 1].item())
            # on_track = float(d_t.item())
            # r = self.alpha * gas - self.beta * on_track

            self.steps += 1
            done = d.item() > 0 or self.steps >= self.time_limit
            # print(f"Step: {self.steps}, d: {d.item()}, done: {done}") # Debugging
            return np_obs, r_pred.item(), done # return (obs, reward, done=True/False)

    def render(self): # pylint: disable=arguments-differ
        """ Rendering """
        import matplotlib.pyplot as plt

        # Initialize monitor if not already set
        if not self.monitor:
            self.figure = plt.figure()
            self.monitor = plt.imshow(
                np.zeros((RED_SIZE, RED_SIZE, 3),
                         dtype=np.uint8))
            
        # Update monitor with current visual observation + pause briefly to display it
        self.monitor.set_data(self._visual_obs)
        plt.pause(.01)

        return self._visual_obs

if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, help='Directory from which MDRNN and VAE are '
                        'retrieved.')
    args = parser.parse_args()

    # Create environment
    env = SimulatedCarracing(args.logdir)

    # Initialize environment and action
    env.reset()
    action = np.array([0., 0., 0.])

    def on_key_press(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 1
        if event.key == 'down':
            action[2] = .8
        if event.key == 'left':
            action[0] = -1
        if event.key == 'right':
            action[0] = 1

    def on_key_release(event):
        """ Defines key pressed behavior """
        if event.key == 'up':
            action[1] = 0
        if event.key == 'down':
            action[2] = 0
        if event.key == 'left' and action[0] == -1:
            action[0] = 0
        if event.key == 'right' and action[0] == 1:
            action[0] = 0

    env.figure.canvas.mpl_connect('key_press_event', on_key_press)
    env.figure.canvas.mpl_connect('key_release_event', on_key_release)
    while True:
        _, _, done = env.step(action)
        env.render()
        if done:
            break
