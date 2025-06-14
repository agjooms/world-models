""" Various auxiliary utilities """
import math
from os.path import join, exists
import torch
from torchvision import transforms
import numpy as np
from models import MDRNNCell, VAE, Controller
import gymnasium as gym
import gymnasium.envs.box2d
import imageio
from envs.simulated_carracing import SimulatedCarracing # Dream environment

# Manually change environment output to 64x64 from 96x96
gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Constants
from utils.config import ASIZE, RSIZE, LSIZE, RED_SIZE, SIZE

# Transform from numpy 96x96x3 to 64x64x3 tensor
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])

# Sample random brownian motion policy
def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

# Save model state to file and best model if is_best.
def save_checkpoint(state, is_best, filename, best_filename):
    """ Save state in filename. Also save in best_filename if is_best. """
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)

# Flatten model parameters (CMA-ES compatible).
def flatten_parameters(params):
    """ Flattening parameters.

    :args params: generator of parameters (as returned by module.parameters())

    :returns: flattened parameters (i.e. one tensor of dimension 1 with all
        parameters concatenated)
    """
    return torch.cat([p.detach().view(-1) for p in params], dim=0).cpu().numpy()

# Restore parameters
def unflatten_parameters(params, example, device):
    """ Unflatten parameters.

    :args params: parameters as a single 1D np array
    :args example: generator of parameters (as returned by module.parameters()),
        used to reshape params
    :args device: where to store unflattened parameters

    :returns: unflattened parameters
    """
    params = torch.Tensor(params).to(device)
    idx = 0
    unflattened = []
    for e_p in example:
        unflattened += [params[idx:idx + e_p.numel()].view(e_p.size())]
        idx += e_p.numel()
    return unflattened

# Load parameters into controller.
def load_parameters(params, controller):
    """ Load flattened parameters into controller.

    :args params: parameters as a single 1D np array
    :args controller: module in which params is loaded
    """
    proto = next(controller.parameters())
    params = unflatten_parameters(
        params, controller.parameters(), proto.device)

    for p, p_0 in zip(controller.parameters(), params):
        p.data.copy_(p_0)

# Generate rollout in CarRacing-v3 environment using pre-trained VAE, MDRNN and Controller
class RolloutGenerator(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v3 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Finding VAE, MDRNN and Controller files
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        # Load VAE and MDRNN states
        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        # Initialize controller
        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)}, weights_only=False)
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        # self.env = gym.make('CarRacing-v3') # For training
        # self.env = gym.make('CarRacing-v3', render_mode="human") # For testing
        self.env = gym.make('CarRacing-v3', render_mode='rgb_array') # For rendering

        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs) # Encode observation to latent space
        action = self.controller(latent_mu, hidden[0]) # Get action from controller
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden) # Predict next hidden state
        return action.squeeze().cpu().detach().numpy(), next_hidden # Added detach() for generation script (remove it if problem)

    def rollout(self, params, render=False, save_video=False, video_path=None):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """

        frames = [] if save_video else None

        # Copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # New gymnasium return tuple
        if isinstance(obs, tuple):
            obs = obs[0]

        # This first render is required !
        frame = self.env.render() if (render or save_video) else None
        if save_video and frame is not None:
            frames.append(frame)

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0

        # New for gymnasium API
        while True:
            obs_t = obs
            # if isinstance(obs, tuple):
            #     obs = obs[0]
            obs_t = transform(obs_t).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs_t, hidden)
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result

            frame = self.env.render() if (render or save_video) else None
            if save_video and frame is not None:
                frames.append(frame)

            cumulative += reward
            if done or i > self.time_limit:
                if save_video and frames:
                    import imageio
                    # Use the actual reward in the filename
                    reward_str = f"{cumulative:.2f}"
                    final_video_path = video_path
                    if final_video_path is None:
                        final_video_path = f"videos/controller_rollout_reward_{reward_str}.mp4"
                    else:
                        base, ext = os.path.splitext(video_path)
                        import os
                        # Always save in videos/ even if a path is given
                        base = os.path.join("videos", os.path.basename(base))
                        final_video_path = f"{base}_reward_{reward_str}{ext}"
                    imageio.mimsave(final_video_path, frames, fps=30)
                    print(f"Saved video to {final_video_path}")
                return - cumulative
            i += 1

# New version: smoothness penalty
class RolloutGenerator2(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v3 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit, smoothness_weight=0.5):
        """ Build vae, rnn, controller and environment. """
        # New: smoothness penalty
        self.smoothness_weight = smoothness_weight # Set to 10 by default

        # Finding VAE, MDRNN and Controller files
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        # Load VAE and MDRNN states
        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        # Initialize controller
        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)}, weights_only=False)
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = gym.make('CarRacing-v3') # For training
        # self.env = gym.make('CarRacing-v3', render_mode="human") # For testing
        # self.env = gym.make('CarRacing-v3', render_mode='rgb_array') # For rendering

        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs) # Encode observation to latent space
        action = self.controller(latent_mu, hidden[0]) # Get action from controller
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden) # Predict next hidden state
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False, save_video=False, video_path=None):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """

        frames = [] if save_video else None

        # Copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # New gymnasium return tuple
        if isinstance(obs, tuple):
            obs = obs[0]

        # This first render is required !
        frame = self.env.render() if (render or save_video) else None
        if save_video and frame is not None:
            frames.append(frame)

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        prev_action = None # Track previous action for smoothness penalty
        prev_prev_action = None # Track second previous action for smoothness penalty
        i = 0

        # New for gymnasium API
        while True:
            obs_t = obs
            # if isinstance(obs, tuple):
            #     obs = obs[0]
            obs_t = transform(obs_t).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs_t, hidden)

            # Apply smoothness penalty
            if prev_action is not None and prev_prev_action is not None:
                acceleration = action[0] - 2 * prev_action[0] + prev_prev_action[0]
                smoothness_penalty = self.smoothness_weight * (acceleration ** 2)
                cumulative -= smoothness_penalty
            prev_prev_action = prev_action.copy() if prev_action is not None else None
            prev_action = action.copy()

            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                obs, reward, done, _ = step_result

            frame = self.env.render() if (render or save_video) else None
            if save_video and frame is not None:
                frames.append(frame)

            cumulative += reward
            if done or i > self.time_limit:
                if save_video and frames:
                    import imageio
                    # Use the actual reward in the filename
                    reward_str = f"{cumulative:.2f}"
                    final_video_path = video_path
                    if final_video_path is None:
                        final_video_path = f"videos/controller_rollout_reward_{reward_str}.mp4"
                    else:
                        import os
                        base, ext = os.path.splitext(video_path)
                        base = join("videos", os.path.basename(base))
                        final_video_path = f"{base}_reward_{reward_str}{ext}"
                    imageio.mimsave(final_video_path, frames, fps=30)
                    print(f"Saved video to {final_video_path}")
                return - cumulative
            i += 1

# New version: dream reward prediction
class RolloutGenerator3(object):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    :attr env: instance of the CarRacing-v3 gym environment
    :attr device: device used to run VAE, MDRNN and Controller
    :attr time_limit: rollouts have a maximum of time_limit timesteps
    """
    def __init__(self, mdir, device, time_limit):
        """ Build vae, rnn, controller and environment. """
        # Finding VAE, MDRNN and Controller files
        vae_file, rnn_file, ctrl_file = \
            [join(mdir, m, 'best.tar') for m in ['vae', 'mdrnn', 'ctrl']]
        assert exists(vae_file) and exists(rnn_file),\
            "Either vae or mdrnn is untrained."

        # Load VAE and MDRNN states
        vae_state, rnn_state = [
            torch.load(fname, map_location={'cuda:0': str(device)})
            for fname in (vae_file, rnn_file)]

        for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
            print("Loading {} at epoch {} "
                  "with test loss {}".format(
                      m, s['epoch'], s['precision']))

        self.vae = VAE(3, LSIZE).to(device)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5).to(device)
        self.mdrnn.load_state_dict(
            {k.strip('_l0'): v for k, v in rnn_state['state_dict'].items()})

        # Initialize controller
        self.controller = Controller(LSIZE, RSIZE, ASIZE).to(device)

        # load controller if it was previously saved
        if exists(ctrl_file):
            ctrl_state = torch.load(ctrl_file, map_location={'cuda:0': str(device)}, weights_only=False)
            print("Loading Controller with reward {}".format(
                ctrl_state['reward']))
            self.controller.load_state_dict(ctrl_state['state_dict'])

        self.env = SimulatedCarracing(directory=mdir, time_limit=time_limit) # Dream environment

        self.device = device

        self.time_limit = time_limit

    def get_action_and_transition(self, obs, hidden):
        """ Get action and transition.

        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.

        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor

        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        _, latent_mu, _ = self.vae(obs) # Encode observation to latent space
        action = self.controller(latent_mu, hidden[0]) # Get action from controller
        _, _, _, _, _, next_hidden = self.mdrnn(action, latent_mu, hidden) # Predict next hidden state
        return action.squeeze().cpu().numpy(), next_hidden

    def rollout(self, params, render=False, save_video=False, video_path=None):
        """ Execute a rollout and returns minus cumulative reward.

        Load :params: into the controller and execute a single rollout. This
        is the main API of this class.

        :args params: parameters as a single 1D np array

        :returns: minus cumulative reward
        """

        frames = [] if save_video else None

        # Copy params into the controller
        if params is not None:
            load_parameters(params, self.controller)

        obs = self.env.reset()

        # New gymnasium return tuple
        if isinstance(obs, tuple):
            obs = obs[0]

        # This first render is required !
        frame = self.env.render() if (render or save_video) else None
        if save_video and frame is not None:
            frames.append(frame)

        hidden = [
            torch.zeros(1, RSIZE).to(self.device)
            for _ in range(2)]

        cumulative = 0
        i = 0

        # New for gymnasium API
        while True:
            obs_t = obs
            # if isinstance(obs, tuple):
            #     obs = obs[0]
            obs_t = transform(obs_t).unsqueeze(0).to(self.device)
            action, hidden = self.get_action_and_transition(obs_t, hidden)
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            elif len(step_result) == 3: # For SimulatedCarracing
                obs, reward, done = step_result
            else:
                obs, reward, done, _ = step_result

            frame = self.env.render() if (render or save_video) else None
            if save_video and frame is not None:
                frames.append(frame)

            cumulative += reward
            if done or i > self.time_limit:
                if save_video and frames:
                    import imageio
                    # Use the actual reward in the filename
                    reward_str = f"{cumulative:.2f}"
                    final_video_path = video_path
                    if final_video_path is None:
                        final_video_path = f"videos/controller_rollout_reward_{reward_str}.mp4"
                    else:
                        base, ext = os.path.splitext(video_path)
                        import os
                        # Always save in videos/ even if a path is given
                        base = os.path.join("videos", os.path.basename(base))
                        final_video_path = f"{base}_reward_{reward_str}{ext}"
                    imageio.mimsave(final_video_path, frames, fps=30)
                    print(f"Saved video to {final_video_path}")
                return - cumulative
            # print(f"Rollout step {i}, done: {done}") # Debugging
            i += 1