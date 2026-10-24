
import gymnasium as gym

from corrected_reward_wrapper import CorrectedRewardWrapper

env = CorrectedRewardWrapper(
    gym.make("LunarLander-v3"),
    landing_bonus=25.0,
)

observation, info = env.reset(seed=42)

print("Initial observation:", observation)

for step in range(10):
    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    print(f"Step {step + 1}: reward = {reward:.2f}")

    if terminated or truncated:
        break

env.close()
