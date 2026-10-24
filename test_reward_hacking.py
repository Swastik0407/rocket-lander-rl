
import gymnasium as gym

from reward_hacking_wrapper import RewardHackingWrapper

env = RewardHackingWrapper(
    gym.make("LunarLander-v3"),
    upright_bonus=1.0,
    slow_speed_threshold=0.3,
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
