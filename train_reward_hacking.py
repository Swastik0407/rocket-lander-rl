
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from reward_hacking_wrapper import RewardHackingWrapper


def make_env():
    env = gym.make("LunarLander-v3")
    env = RewardHackingWrapper(
        env,
        upright_bonus=1.0,
        slow_speed_threshold=0.3,
        time_penalty=0.0,
    )
    return Monitor(env)


env = make_env()

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]},
)

model.learn(total_timesteps=200_000)

model.save("models/ppo_lunarlander_reward_hacking")

print("Reward-hacking training finished!")

env.close()
