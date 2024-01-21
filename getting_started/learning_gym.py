import gymnasium as gym
from gymnasium.wrappers.record_video import RecordVideo

env = gym.make("LunarLander-v2", render_mode="rgb_array")
env = RecordVideo(env, './lunar_lander_videos')
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()