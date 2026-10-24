
import gymnasium as gym

# Create the game with a visible window
env = gym.make("LunarLander-v3", render_mode="human")

# Start an episode
observation, info = env.reset(seed=42)

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)
print("Initial observation:", observation)

total_reward = 0

for step in range(1000):
    # The rocket chooses a random action
    action = env.action_space.sample()

    # Advance the simulation by one step
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward

    if terminated or truncated:
        print("Episode finished!")
        print("Total reward:", round(total_reward, 2))
        break

env.close()
