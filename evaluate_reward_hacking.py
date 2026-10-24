
import gymnasium as gym
from stable_baselines3 import PPO

model = PPO.load(
    "models/ppo_lunarlander_reward_hacking.zip"
)

# Evaluate WITHOUT the custom reward wrapper.
env = gym.make("LunarLander-v3")

scores = []
episode_lengths = []
successful_landings = 0

for episode in range(10):
    observation, info = env.reset()

    total_reward = 0.0
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

    scores.append(total_reward)
    episode_lengths.append(steps)

    # A 200+ score is our conventional benchmark.
    if total_reward >= 200:
        successful_landings += 1

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}, "
        f"Steps = {steps}"
    )

env.close()

print("\n--- Reward-Hacking Evaluation ---")
print(f"Average reward: {sum(scores) / len(scores):.2f}")
print(f"Average episode length: {sum(episode_lengths) / len(episode_lengths):.1f}")
print(f"Episodes scoring 200+: {successful_landings}/10")
