
import gymnasium as gym
from stable_baselines3 import PPO

# Create the LunarLander environment with a visible window
env = gym.make("LunarLander-v3", render_mode="human")

# Load your best-performing trained model
model = PPO.load("models/ppo_lunarlander_corrected_v2")

obs, info = env.reset()
episode_reward = 0
done = False
truncated = False

print("Watching Corrected V2 land...")
print("Close the simulation window to stop.")

try:
    while True:
        obs, info = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            # Let the trained agent choose an action
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward

        print(f"Episode reward: {episode_reward:.2f}")

except KeyboardInterrupt:
    print("Simulation stopped.")

finally:
    env.close()
