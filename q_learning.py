import gymnasium as gym
import numpy as np 
import random 

gamma = 0.99
epsilon = 0.1
alpha = 0.5
episode = 500

q = np.zeros((48,4))

def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[state])  

for episode in range(episode):
    env = gym.make("CliffWalking-v1", render_mode="human")

else:
    env = gym.make("liffWalking-v1")
    done = False
    state,_ = env.reset()

    episode_len = 0
    tot_reward = 0

    while not done:
        action = epsilon_greedy(state)
        next_state,reward,terminated,truncated,_ = env.step(action)
        done = terminated or truncated

        q[state,action] +=alpha *(reward + gamma *np.max(q[next_state])-q[state,action])

        state = next_state
        episode_len +=1
        tot_reward += reward
    print(
        f"Episode = {episode + 1}/500: "
        f"Total reward = {total_reward} "
        f"& Episode length = {episode_len}"
    )
    env.close()
    
env = gym.make("CliffWalking-v1", render_mode="human")
state , _ = env.reset()
done = False
total_reward = 0
episode_len = 0

while not done:
    action = np.argmax(q[state])
    state,reward,terminated,truncated,_ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    episode_len +=1
print(f" total reward = {total_reward} & episode length = {episode_len}")  
env.close()  