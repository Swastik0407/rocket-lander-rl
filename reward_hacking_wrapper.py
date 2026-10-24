
import gymnasium as gym


class RewardHackingWrapper(gym.Wrapper):
    """
    Deliberately flawed reward function.

    Gives a bonus whenever the lander is upright
    and moving slowly, even if it has not landed.
    """

    def __init__(
        self,
        env,
        upright_bonus=1.0,
        slow_speed_threshold=0.3,
        time_penalty=0.0,
    ):
        super().__init__(env)

        self.upright_bonus = upright_bonus
        self.slow_speed_threshold = slow_speed_threshold
        self.time_penalty = time_penalty

    def step(self, action):
        (
            observation,
            original_reward,
            terminated,
            truncated,
            info,
        ) = self.env.step(action)

        # LunarLander observation:
        # index 2 = horizontal velocity
        # index 3 = vertical velocity
        # index 4 = angle / tilt
        # index 5 = angular velocity

        horizontal_speed = abs(float(observation[2]))
        vertical_speed = abs(float(observation[3]))
        tilt = abs(float(observation[4]))
        angular_speed = abs(float(observation[5]))

        # Deliberately flawed rule:
        # reward an upright, slowly moving lander every step.
        if (
            horizontal_speed < self.slow_speed_threshold
            and vertical_speed < self.slow_speed_threshold
            and tilt < 0.1
            and angular_speed < 0.1
        ):
            original_reward += self.upright_bonus

        # Optional cost for taking another step.
        original_reward -= self.time_penalty

        return (
            observation,
            original_reward,
            terminated,
            truncated,
            info,
        )
