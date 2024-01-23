import safety_gymnasium
from gymnasium.utils.save_video import save_video

env_id = 'SafetyCarGoal2-v0'
env = safety_gymnasium.make(env_id,
                            render_mode="rgb_array_list",
                            camera_name="track",
                            width=1920,
                            height=1080)

num_episodes = 5
for episode in range(num_episodes):
    observation, info = env.reset()
    
    while True:
        observation, reward, cost, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break

    save_video(env.render(),
        f"./videos_{env_id}",
        fps=30,
        episode_index=episode,
        episode_trigger=lambda episode: True)