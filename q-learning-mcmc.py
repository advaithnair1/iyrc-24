import random
import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery = True)

# TODO: fix hyperparameters if needed
# Hyperparameters (same as regular Q-learning)
epsilon = 0.1
alpha = 0.1
gamma = 0.95
num_episodes = 1000
