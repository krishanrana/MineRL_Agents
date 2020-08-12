# Author: Krishan Rana
# MineRL Competition
# 08/2020

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
from tensorboardX import SummaryWriter
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import minerl
from model import Segmenter
from rgb2cmap import cmap2rgb, tree_cmap

ENV = 'MineRLTreechop-v0'
model_name = 'minerl_' + str(time.time())
log_dir = "runs/" + model_name
writer    = SummaryWriter(log_dir=log_dir)


env = gym.make(ENV).unwrapped

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

segmenter = Segmenter('../ckpt/checkpoint-no-downscale-rgb.pth.tar', 4, "101", "segmenter", "module.")
segmenter.eval()

if torch.cuda.is_available():
    segmenter.cuda()


# Scales an image from uint8 range [0,255] to float [0,1]
def scale( image ):
    return image / 255.0

# Preprocesses sample image and mask
def pre( sample ):
    return { 'image' : scale(sample['image']), 'mask' : sample['mask'] }

# Converts image to tensor with batch dimension
def batch_tensor( image ):
    return torch.tensor(image.transpose(2, 0, 1)[None]).float()


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Q-network
# ^^^^^^^^^
#
# Our model will be a convolutional neural network that takes in the
# difference between the current and previous screen patches. It has two
# outputs, representing :math:`Q(s, \mathrm{left})` and
# :math:`Q(s, \mathrm{right})` (where :math:`s` is the input to the
# network). In effect, the network is trying to predict the *expected return* of
# taking each action given the current input.


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))



def process_obs(obs):

    pov = scale(obs['pov'])
    pov_tensor = batch_tensor(pov)

    if torch.cuda.is_available():
        pov_tensor = pov_tensor.cuda()

    segm = segmenter.rgb(pov_tensor, tree_cmap, pov.shape[:2])
    # Only extract the red channel
    segm = scale(segm[0,:,:,:1])

    tree_trunk_size = len((np.nonzero(segm)[0]))
    # Scale reward by size of tree mask
    reward = tree_trunk_size/1000

    combined = np.concatenate((pov, segm), 2)

    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    combined = np.ascontiguousarray(combined, dtype=np.float32).transpose((2,0,1))
    combined = torch.from_numpy(combined)
    # Resize, and add a batch dimension (BCHW)
    return combined.unsqueeze(0).to(device), reward


obs = env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

obs, _ = process_obs(obs)
_, _, screen_height, screen_width = obs.shape

# Get number of actions from gym action space
n_actions = 4

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def create_action(idx):
    
    idx = idx.item()
    act = env.action_space.noop()
    if (idx == 0):
        act['camera'] = [0,-5]
    elif (idx == 1):
        act['camera'] = [0,5]
    elif (idx == 2):
        act['forward'] = 1
    elif (idx == 3):
        act['jump'] = 1
    
    act['attack'] = 1

    return act


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            idx = policy_net(state).max(1)[1].view(1, 1)
            act = create_action(idx)
            return act, idx
    else: 
        idx = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        act = create_action(idx)
        return act, idx

episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    state_action_values = policy_net(state_batch).gather(1, action_batch)


    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    writer.add_scalar('{}/loss'.format(ENV), loss, total_steps)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


total_steps = 0

num_episodes = 10000
rollout_size = 5000

for i_episode in range(num_episodes):
    # Initialize the environment and state
    obs = env.reset()
    last_screen, _ = process_obs(obs)
    current_screen, _ = process_obs(obs)
    #state = current_screen - last_screen
    state = current_screen
    ep_rewards = 0
    steps = 0

    for _ in range(rollout_size):
        # Select and perform an action
        act_long, act = select_action(state)
        nobs, reward, done, _ = env.step(act_long)
        last_screen = current_screen
        
        # Updated reward based off the tree trunk mask
        #current_screen, _  = process_obs(nobs)
        current_screen, trunk_rew = process_obs(nobs)
        reward = 10*reward + trunk_rew

        ep_rewards += reward
        reward = torch.tensor([reward], device=device)
        
        if not (total_steps%10):
            print('Episode Rewards: ', ep_rewards)
            print('Done: ', done)
        
        # Observe new state
        if not done:
            #next_state = current_screen - last_screen
            next_state = current_screen
        else:
            next_state = None
        
        # Store the transition in memory
        memory.push(state, act, next_state, reward)
         
        # Move to the next state
        state = next_state
        
        steps += 1
        total_steps += 1
         
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done or steps>=rollout_size:
            writer.add_scalar('{}/ep_length'.format(ENV), steps, total_steps)
            writer.add_scalar('{}/ep_reward'.format(ENV), ep_rewards, total_steps)
            print('Episode Done')
            break

    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()

