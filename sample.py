import gymnasium as gym
from env import HomeostaticEnv
import time
import numpy as np

gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=HomeostaticEnv,
    max_episode_steps=1000,  # Prevent infinite episodes
)
num_vars = 30
env = gym.make("gymnasium_env/GridWorld-v0", num_vars=num_vars, rewarding="operational_regimes", reward_scale=0.01)
#env = gym.make("gymnasium_env/GridWorld-v0", num_vars=num_vars, rewarding="euclidian", reward_scale=0.01)
#env = gym.make("gymnasium_env/GridWorld-v0", num_vars=num_vars, rewarding="default", reward_scale=0.01)
reward, info = env.reset(seed=42)
done = False
ended = False

average = []
num_episodes = 50
for i in range(num_episodes):
    rewards = []
    while True:
        action = np.random.normal(0, 0.1, num_vars)  # Replace with your action selection logic
        observation, reward, done, ended, info = env.step(action)
        rewards.append(reward)
        #env.render()
        if done or ended:
            print("Episode finished. Total reward:", sum(rewards))
            std = np.std(rewards)
            average.append(std)
            print("Reward STD: ", std)
            rewards = []
            observation, info = env.reset(seed=42)
            time.sleep(0.1)
            break
print(f"Average Reward STD over {num_episodes} episodes: ", np.mean(average))