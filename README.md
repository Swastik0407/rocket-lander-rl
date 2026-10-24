# 🚀 Rocket Lander RL — Reinforcement Learning with PPO

**An experimental reinforcement learning project exploring reward engineering, policy optimization, and model evaluation in a simulated lunar landing environment.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Gymnasium](https://img.shields.io/badge/Environment-Gymnasium-orange)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/RL-Stable--Baselines3-green)](https://stable-baselines3.readthedocs.io/)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-purple)](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

## 📌 Overview

Rocket Lander RL investigates how different reward strategies affect the behavior and performance of a reinforcement learning agent in the **LunarLander-v3 environment**.

The project uses **Proximal Policy Optimization (PPO)** to train an agent to control a lunar lander through simulated actions. Multiple reward configurations are evaluated to investigate how reward shaping influences cumulative reward, episode duration, and evaluation performance.

Rather than training a single model and reporting its best result, this project compares several experimental configurations using a common 100-episode evaluation setup.

### 🎯 Project Objectives

- Train a lunar landing agent using PPO.
- Experiment with different reward-engineering strategies.
- Compare baseline learning against modified reward functions.
- Evaluate models using average reward, reward variability, best and worst reward, episode length, and a reward-threshold metric.
- Visualize experimental results and document the limitations of the evaluation.

## 🧠 How It Works

The agent interacts with the simulated environment in a repeated feedback loop:

1. **Observe:** The environment provides the lander's state.
2. **Act:** The PPO policy selects an action.
3. **Interact:** The environment updates the lander's position, velocity, and landing state.
4. **Reward:** The environment provides feedback according to its reward function and any configured reward modifications.
5. **Learn:** PPO updates the policy using experience collected during training.
6. **Evaluate:** The trained models are compared using a common evaluation procedure.

## ⚙️ Algorithm: Proximal Policy Optimization (PPO)

PPO is a policy-gradient reinforcement learning algorithm that learns a policy for selecting actions based on the current state.

It uses a clipped policy objective to limit excessively large policy updates, helping improve training stability compared with unconstrained policy-gradient updates.

The central idea is to improve the policy while restricting how far each update can move from the previous policy.

In this project, PPO learns to control the lander's simulated actions through repeated interaction with the environment.

**Why PPO?**

- Suitable for continuous-control reinforcement learning tasks.
- Supported by Stable-Baselines3.
- Provides a practical starting point for policy optimization experiments.
- Allows different reward configurations to be compared using the same general algorithm.

## 🧪 Experiments

The project explores five configurations:

| Experiment | Description |
|---|---|
| Baseline | PPO trained with the environment's original reward function. |
| Time Penalty | Tests the effect of adding a penalty associated with time or episode progression. |
| Reward Hacking | An experimental reward modification that produced undesirable results. |
| Corrected V1 | Tests an initial correction to the reward-shaping strategy. |
| Corrected V2 | Uses a one-time bonus of +25 for qualifying terminal outcomes involving leg contact and low velocity and tilt. |

The Corrected V2 experiment aims to reward a more desirable terminal landing state rather than relying only on cumulative reward.

Reward engineering requires careful evaluation: an agent may learn to exploit a modified reward function without necessarily becoming better at the intended task.

## 📊 Experimental Results

Each configuration was evaluated over **100 episodes**. The results below are from the project's recorded evaluation runs.

| Model | Average Reward | Std. Reward | Best Reward | Worst Reward | Avg. Episode Length | Episodes Scoring 200+ |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 117.48 | 97.21 | 227.39 | -132.90 | 664.46 | 7/100 |
| Time Penalty | 66.05 | 101.09 | 197.33 | -166.42 | 805.34 | 0/100 |
| Reward Hacking | -81.28 | 44.01 | 58.47 | -197.88 | 858.70 | 0/100 |
| Corrected V1 | 27.21 | 144.23 | 256.37 | -251.88 | 668.23 | 21/100 |
| **Corrected V2** | **155.72** | 106.79 | 257.42 | -62.73 | 361.43 | **62/100** |

### 🏆 Key Findings

- **Best average reward:** Corrected V2 achieved an average reward of **155.72**.
- **Reward-threshold performance:** Corrected V2 scored at least 200 in **62 out of 100 evaluation episodes (62%)**.
- **Lower average episode length:** Corrected V2 averaged 361.43 steps per episode, compared with 664.46 for the baseline.
- **Reward engineering matters:** The experimental configurations produced substantially different results, with the reward-hacking configuration performing poorly.
- **The task is not fully solved:** Corrected V2 still shows considerable reward variability, and 38 of its 100 episodes scored below 200.

> **Important:** A score of 200 or higher is used here as an evaluation threshold, not as a verified physical landing-success metric. The 62% figure should not be interpreted as a 62% successful-landing rate without a separate, explicitly defined landing-success evaluation.

### 📈 Result Visualization

The repository includes `average_reward_comparison.png`, which visualizes average reward across the five configurations.

The recorded metrics are also available in `model_comparison.csv` for further analysis and visualization.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Reinforcement learning algorithm | PPO |
| Environment | Gymnasium LunarLander-v3 |
| RL framework | Stable-Baselines3 |
| Numerical computing | NumPy |
| Data analysis | Pandas |
| Visualization | Matplotlib |
| Version control | Git and GitHub |

## 📁 Project Structure

```text
rocket-lander-rl/
├── models/
│   ├── ppo_lunarlander_final.zip
│   ├── ppo_lunarlander_time_penalty.zip
│   ├── ppo_lunarlander_reward_hacking.zip
│   ├── ppo_lunarlander_corrected.zip
│   └── ppo_lunarlander_corrected_v2.zip
├── results/
│   └── evaluations.npz
├── corrected_reward_wrapper.py
├── reward_wrapper.py
├── reward_hacking_wrapper.py
├── train_baseline.py
├── train_shaped.py
├── train_reward_hacking.py
├── train_corrected.py
├── evaluate.py
├── evaluate_time_penalty.py
├── evaluate_reward_hacking.py
├── evaluate_corrected.py
├── compare_models.py
├── plot_result.py
├── record_landing.py
├── model_comparison.csv
├── baseline_results.csv
├── average_reward_comparison.png
├── random_test.py
├── test_corrected_reward.py
├── test_reward_hacking.py
├── .gitignore
└── README.md
```

*Note: The tree highlights the main project files; additional saved models or evaluation artifacts may also be present in the repository.*

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Swastik0407/rocket-lander-rl.git
cd rocket-lander-rl
```

### 2. Create a virtual environment

Python 3.12 is the recommended starting point for reproducing this project's environment.

**Windows (PowerShell):**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install "gymnasium[box2d]" "stable-baselines3[extra]" pandas matplotlib
```

If an installation error occurs, check the Python version and the Gymnasium Box2D installation requirements.

### 4. Run an evaluation

For the baseline evaluation script:

```bash
python evaluate.py
```

For the corrected reward experiment:

```bash
python evaluate_corrected.py
```

**Note:** Check the relevant script before running it to confirm the model path, environment settings, number of episodes, and output behavior. Script defaults may differ from the evaluation configuration used to generate the table above.

### 5. Compare the recorded results

```bash
python compare_models.py
```

To regenerate the average-reward comparison chart:

```bash
python plot_result.py
```

Review each script before rerunning an experiment, because training or evaluation may overwrite existing outputs depending on how the script is configured.

## 💾 Pretrained Models

The `models/` directory contains saved PPO model checkpoints from the experiments.

The main model to inspect is:

```text
models/ppo_lunarlander_corrected_v2.zip
```

Loading a saved checkpoint allows evaluation without repeating the entire training process, provided the required dependencies and compatible environment are installed.

## ⚠️ Limitations

- **Evaluation scope:** Results are based on 100 episodes per recorded configuration. Larger evaluation runs would provide stronger evidence.
- **Randomness:** Environment initialization and training randomness can affect performance. Repeated training runs with different seeds are needed to assess consistency.
- **Reward-threshold proxy:** A reward of 200 or higher does not independently prove a successful landing. A dedicated success metric should examine terminal state, contact conditions, velocity, and tilt.
- **Reward shaping:** Modified rewards can encourage unintended behavior. A reward improvement must be checked against the actual task objective.
- **Fair comparison:** A common evaluation protocol helps comparison, but identical evaluation conditions alone do not establish statistical significance or eliminate training variance.
- **Reproducibility:** Training hyperparameters, random seeds, dependency
