
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import os

# Create folders for saving models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Create and monitor the training environment
env = Monitor(gym.make("LunarLander-v3"))

# Create a separate environment for evaluation
eval_env = Monitor(gym.make("LunarLander-v3"))

# PPO creates and trains the neural network
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

# Evaluate periodically and save the best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="models/",
    log_path="results/",
    eval_freq=5000,
    n_eval_episodes=5,
    deterministic=True,
)

# Train the agent
model.learn(
    total_timesteps=200_000,
    callback=eval_callback,
    progress_bar=False,
)

# Save the final model
model.save("models/ppo_lunarlander_final")

env.close()
eval_env.close()

print("Training finished! Model saved.")
