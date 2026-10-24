
import gymnasium as gym
from stable_baselines3 import PPO

# Load the model trained with the time penalty.
model = PPO.load(
    "models/ppo_lunarlander_time_penalty.zip"
)

# IMPORTANT: No reward wrapper here.
# We want to evaluate using the original environment reward.
env = gym.make("LunarLander-v3")

scores = []

for episode in range(10):
    observation, info = env.reset()
    total_reward = 0.0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Use the trained policy to choose an action.
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

    scores.append(total_reward)
    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}"
    )

env.close()

average_reward = sum(scores) / len(scores)
successful_episodes = sum(
    score >= 200 for score in scores
)

print("\n--- Time-Penalty Evaluation ---")
print(f"Average reward: {average_reward:.2f}")
print(
    f"Episodes scoring 200+: "
    f"{successful_episodes}/10"
)
