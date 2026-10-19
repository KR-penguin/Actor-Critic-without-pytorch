import gymnasium as gym


def make_lunar_lander(render_mode: str | None = "human") -> gym.Env:
    return gym.make("LunarLander-v3", render_mode=render_mode)


env = make_lunar_lander()

try:
    observation, info = env.reset(seed=42)
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
finally:
    env.close()