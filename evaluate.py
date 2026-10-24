
import gymnasium as gym
from stable_baselines3 import PPO

# Load the final trained model
model = PPO.load("models/ppo_lunarlander_final")

# Create the game window
env = gym.make("LunarLander-v3", render_mode="human")

episodes = 10
scores = []

for episode in range(episodes):
    observation, info = env.reset()
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        # Let the trained AI choose its action
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    scores.append(total_reward)

    print(
        f"Episode {episode + 1}: "
        f"Reward = {total_reward:.2f}"
    )

print("\n--- Evaluation Results ---")
print(f"Average reward: {sum(scores) / len(scores):.2f}")
print(f"Episodes scoring 200+: {sum(s >= 200 for s in scores)}/10")

env.close()
