import gymnasium as gym


class RewardRecordVideo(gym.wrappers.RecordVideo):
    def __init__(
        self,
        env,
        video_folder,
        episode_trigger=None,
        step_trigger=None,
        video_length=0,
        name_prefix="rl-video",
        fps=None,
        disable_logger=True,
        gc_trigger=lambda episode: True,
    ):
        super().__init__(
            env,
            video_folder,
            episode_trigger,
            step_trigger,
            video_length,
            name_prefix,
            fps,
            disable_logger,
        )
        self.gc_trigger = gc_trigger  # backwards compatibility
        self.reward_sum = 0.0

    def reset(self, **kwargs):
        retval =  super().reset(**kwargs)
        self.reward_sum = 0.0
        return retval

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.reward_sum += reward
        return obs, reward, terminated, truncated, info

    def stop_recording(self):
        if self._video_name:
            self._video_name = self._video_name + "_" + str(self.reward_sum)
        return super().stop_recording()
