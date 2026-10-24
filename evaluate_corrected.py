import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Load the trained corrected-reward model
model = PPO.load("models/ppo_lunarlander_corrected_v2")

# IMPORTANT: Evaluate without the custom reward wrapper
env = gym.make("LunarLander-v3")

episodes = 100
rewards = []
lengths = []

for episode in range(episodes):
    observation, info = env.reset(seed=episode)

    total_reward = 0
    steps = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        steps += 1

    rewards.append(total_reward)
    lengths.append(steps)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, "
        f"Steps = {steps}"
    )

print("\n--- Corrected Model Results ---")
print(f"Average reward: {np.mean(rewards):.2f}")
print(f"Best reward: {np.max(rewards):.2f}")
print(f"Worst reward: {np.min(rewards):.2f}")
print(f"Average episode length: {np.mean(lengths):.1f}")
print(f"Episodes scoring 200+: {sum(r >= 200 for r in rewards)}/{episodes}")

env.close()