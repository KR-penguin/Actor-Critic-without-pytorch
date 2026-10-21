import gymnasium as gym
import model
import config
import json
from datetime import datetime

# Actions on the Lunar Lander environment
NO_OPERATION = 0
EXCUTION_LEFT_ENGINE = 1
EXCUTION_MAIN_ENGINE = 2
EXCUTION_RIGHT_ENGINE = 3

'''
observation = [x_position, y_position, x_velocity, y_velocity, 
lander_angle, angular_velocity, left_leg_contact (boolean), right_leg_contact (boolean)]

reward funtion = "reward function table.png"
'''

def make_lunar_lander(render_mode: str | None = "human") -> gym.Env:
    return gym.make("LunarLander-v3", render_mode=render_mode)

def save_ai_model() -> None:
    data = agent.get_ai_model()
    date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"ai_model_{date_string}.json"

    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


agent = model.Agent()
env = make_lunar_lander()

for episode in range(config.TOTAL_EPISODES):
    print(f"episode : {episode} started")
    
    try:
        observation, _ = env.reset() # observation = state
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = agent.get_action_from_policy(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            agent.update(action, observation, reward)
    finally:
        env.close()

save_ai_model()