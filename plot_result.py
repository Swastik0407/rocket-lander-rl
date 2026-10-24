
import pandas as pd
import matplotlib.pyplot as plt

# Read the model comparison results
df = pd.read_csv("model_comparison.csv")

# Sort models by average reward
df = df.sort_values("Average Reward", ascending=False)

# Chart 1: Average reward
plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["Average Reward"])
plt.axhline(y=200, linestyle="--", label="200-point benchmark")
plt.ylabel("Average evaluation reward")
plt.title("LunarLander: Model Reward Comparison")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("average_reward_comparison.png", dpi=200)
plt.show()

# Chart 2: 200-point success rate
plt.figure(figsize=(10, 6))
plt.bar(df["Model"], df["Success Rate (%)"])
plt.ylabel("Episodes scoring 200+ (%)")
plt.title("LunarLander: 200-Point Success Rate")
plt.ylim(0, 100)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("success_rate_comparison.png", dpi=200)
plt.show()

print("Charts saved successfully.")
