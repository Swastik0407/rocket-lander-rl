
import gymnasium as gym


class TimePenaltyWrapper(gym.Wrapper):
    """
    Adds a small penalty at every step.
    The agent is encouraged to finish efficiently.
    """

    def __init__(self, env, time_penalty=0.05):
        super().__init__(env)
        self.time_penalty = time_penalty

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        # Penalize every time step
        shaped_reward = reward - self.time_penalty

        return (
            observation,
            shaped_reward,
            terminated,
            truncated,
            info,
        )
