
import gymnasium as gym
import numpy as np
import random

env = gym.make("CliffWalking-v1")

gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500

q = np.zeros((48, 4))


def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q[state])


for episode in range(episodes):

    done = False
    state, _ = env.reset()

    action = epsilon_greedy(state)

    total_reward = 0
    episode_len = 0

    while not done:

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        next_action = epsilon_greedy(next_state)

        # SARSA update
        if done:
            target = reward
        else:
            target = reward + gamma * q[next_state, next_action]

        q[state, action] += alpha * (target - q[state, action])

        state = next_state
        action = next_action

        total_reward += reward
        episode_len += 1

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