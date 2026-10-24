
import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

# Model paths: check these against your actual files.
MODELS = {
    "Baseline": "models/ppo_lunarlander_final",
    "Time Penalty": "models/ppo_lunarlander_time_penalty",
    "Reward Hacking": "models/ppo_lunarlander_reward_hacking",
    "Corrected V1": "models/ppo_lunarlander_corrected",
    "Corrected V2": "models/ppo_lunarlander_corrected_v2",
}

N_EPISODES = 100
SEED_START = 0

all_results = []

for model_name, model_path in MODELS.items():
    print(f"\nEvaluating {model_name}...")

    try:
        model = PPO.load(model_path)
    except Exception as error:
        print(f"Could not load {model_name}: {error}")
        print("Check the model path and filename.")
        continue

    # Evaluate every model in the original environment.
    env = gym.make("LunarLander-v3")

    rewards = []
    lengths = []

    for episode in range(N_EPISODES):
        # The same reset seeds are used for every model.
        observation, info = env.reset(seed=SEED_START + episode)

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

        rewards.append(total_reward)
        lengths.append(steps)

    env.close()

    results = {
        "Model": model_name,
        "Episodes": len(rewards),
        "Average Reward": np.mean(rewards),
        "Std Reward": np.std(rewards, ddof=1),
        "Best Reward": np.max(rewards),
        "Worst Reward": np.min(rewards),
        "Average Length": np.mean(lengths),
        "Episodes 200+": sum(r >= 200 for r in rewards),
        "Success Rate (%)": 100 * sum(r >= 200 for r in rewards) / len(rewards),
    }

    all_results.append(results)

    print(f"Average reward: {results['Average Reward']:.2f}")
    print(f"Success rate: {results['Success Rate (%)']:.1f}%")

# Save only if at least one model was evaluated.
if all_results:
    df = pd.DataFrame(all_results)
    df.to_csv("model_comparison.csv", index=False)

    print("\n--- Final Model Comparison ---")
    print(df.round(2).to_string(index=False))
    print("\nSaved results to model_comparison.csv")
else:
    print("No models were evaluated. Check your model paths.")
