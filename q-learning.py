import random
import numpy as np
import gym

# TODO: find other suitable environments
env = gym.make("FrozenLake-v1", is_slippery = True)

# TODO: fix hyperparameters if needed
epsilon = 0.1
alpha = 0.1
gamma = 0.95
num_episodes = 1000

# Q-table

Q = np.zeros((env.observation_space.n, env.action_space.n))

# Training loop
for episode in range (num_episodes): 
    state = env.reset()
    done = False

    while not done: 
        # Epsilon greedy policy
        if random.uniform(0, 1) < epsilon: 
            action = env.action_space.sample()
        else: 
            action = np.argmax(Q[state])
            
        
        next_state, reward, done, _ = env.step(action)
        
        # Q-learning update rule
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
        state = next_state
        

print("Q-table: ")
print(Q)

# Testing the intrinsic policy

state = env.reset()
env.render()
done = False

while not Done: 
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    env.render()
    
print(f"Reward: {reward}")
        

        



