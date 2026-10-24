
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from reward_wrapper import TimePenaltyWrapper
import os

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Training environment with the custom reward
env = TimePenaltyWrapper(
    gym.make("LunarLander-v3"),
    time_penalty=0.05,
)
env = Monitor(env)

# Evaluation environment with the same reward
eval_env = TimePenaltyWrapper(
    gym.make("LunarLander-v3"),
    time_penalty=0.05,
)
eval_env = Monitor(eval_env)

model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    verbose=1,
    policy_kwargs={"net_arch": [64, 64]},
    seed=42,
)

callback = EvalCallback(
    eval_env,
    best_model_save_path="models/shaped/",
    log_path="results/shaped/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
)

model.learn(
    total_timesteps=200_000,
    callback=callback,
)

model.save("models/ppo_lunarlander_time_penalty")

env.close()
eval_env.close()

print("Time-penalty training finished!")
