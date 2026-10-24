import gymnasium as gym


class CorrectedRewardWrapper(gym.Wrapper):
    def __init__(self, env, landing_bonus=25.0):
        super().__init__(env)
        self.landing_bonus = landing_bonus
        self.bonus_given = False

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.bonus_given = False
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = (
            self.env.step(action)
        )

        left_leg = observation[6]
        right_leg = observation[7]

        horizontal_speed = abs(float(observation[2]))
        vertical_speed = abs(float(observation[3]))
        tilt = abs(float(observation[4]))

        successful_landing = (
            terminated
            and reward > 0
            and left_leg > 0.5
            and right_leg > 0.5
            and horizontal_speed < 0.3
            and vertical_speed < 0.3
            and tilt < 0.1
        )

        if successful_landing and not self.bonus_given:
            reward += self.landing_bonus
            self.bonus_given = True

        return observation, reward, terminated, truncated, info