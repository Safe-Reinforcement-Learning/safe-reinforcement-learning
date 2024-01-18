# Run this script with this command: xvfb-run -a python3 mujoco_starter.py
# (Since our Docker containers don't have graphics hardware, Xvfb acts as a graphics server for us.)

import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

env = gym.make('Walker2d-v4', render_mode="rgb_array")
env = RecordVideo(env, './video', step_trigger=lambda s: s == 0, video_length=500)
observation, info = env.reset()

for _ in range(500):
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())

    if terminated or truncated:
        observation, info = env.reset()

env.close()