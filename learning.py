import gymnasium as gym
import model
import config
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg") # 창을 띄우지 않고 파일로만 저장
import matplotlib.pyplot as plt
from datetime import datetime
from batch import Batch

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

MOVING_AVERAGE_WINDOW = 100
MODELS_DIRECTORY = "models" # ai_model_*.json, review_log_*.json, training_graph_*.png 저장 위치

def make_lunar_lander(render_mode: str | None) -> gym.Env:
    return gym.make("LunarLander-v3", render_mode=render_mode)

def moving_average(values, window=MOVING_AVERAGE_WINDOW):
    values = np.array(values, dtype=float)
    return np.array([values[max(0, i - window + 1): i + 1].mean() for i in range(len(values))])

def save_ai_model(date_string):
    data = agent.get_ai_model()
    file_name = os.path.join(MODELS_DIRECTORY, f"ai_model_{date_string}.json")

    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)
    print(f"saved : {file_name}")

def save_training_graph(date_string, training_log):
    file_name = os.path.join(MODELS_DIRECTORY, f"training_graph_{date_string}.png")
    episodes = np.arange(len(training_log["rewards"]))

    figure, (reward_axis, policy_axis, state_value_axis) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    reward_axis.plot(episodes, training_log["rewards"], alpha=0.3, label="episode reward")
    reward_axis.plot(episodes, moving_average(training_log["rewards"]), label=f"moving average ({MOVING_AVERAGE_WINDOW})")
    reward_axis.axhline(200, color="green", linestyle="--", label="solved (200)")
    reward_axis.set_ylabel("total reward")
    reward_axis.legend(loc="lower right")
    reward_axis.grid(alpha=0.3)

    policy_axis.plot(episodes, training_log["policy_losses"], alpha=0.3)
    policy_axis.plot(episodes, moving_average(training_log["policy_losses"]))
    policy_axis.set_ylabel("policy loss")
    policy_axis.grid(alpha=0.3)

    state_value_axis.plot(episodes, training_log["state_value_losses"], alpha=0.3)
    state_value_axis.plot(episodes, moving_average(training_log["state_value_losses"]))
    state_value_axis.set_ylabel("state value loss (MSE)")
    state_value_axis.set_yscale("log")
    state_value_axis.set_xlabel("episode")
    state_value_axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(file_name, dpi=120)
    plt.close(figure)
    print(f"saved : {file_name}")

def save_review_log(date_string, review_log):
    '''
    에피소드마다 (환경 seed, 실제 했던 행동들)을 저장한다.
    같은 seed로 reset하고 같은 행동을 넣으면 똑같은 게임이 재현되므로 review.py로 다시 볼 수 있다.
    actions는 용량을 줄이려고 "0213..." 같은 문자열로 저장한다.
    '''
    file_name = os.path.join(MODELS_DIRECTORY, f"review_log_{date_string}.json")

    with open(file_name, "w", encoding="utf-8") as file:
        json.dump(review_log, file)
    print(f"saved : {file_name}")

def save_all(training_log, review_log):
    date_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(MODELS_DIRECTORY, exist_ok=True)
    save_ai_model(date_string)
    save_training_graph(date_string, training_log)
    save_review_log(date_string, review_log)


agent = model.Agent()
env = make_lunar_lander(None)
batch = Batch(agent)

training_log = {
    "rewards": [],            # 에피소드 총 reward
    "policy_losses": [],      # 에피소드 동안 업데이트한 policy loss 평균
    "state_value_losses": [], # 에피소드 동안 업데이트한 state value loss 평균
}

review_log = {
    "environment": "LunarLander-v3",
    "episodes": [], # {"episode", "seed", "reward", "actions"}
}

try:
    for episode in range(config.TOTAL_EPISODES):
        episode_seed = int(np.random.randint(0, 2**31 - 1))
        observation, _ = env.reset(seed=episode_seed) # observation = state
        terminated = False
        truncated = False
        is_done = terminated or truncated

        total_reward = 0.0
        taken_actions = []
        policy_losses = []
        state_value_losses = []

        while not is_done:

            last_observation = observation
            action = agent.get_action_from_policy(observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            is_done = terminated or truncated
            total_reward += reward
            taken_actions.append(int(action))

            batch.add(action, last_observation, reward, observation, is_game_over = terminated)
            if batch.is_full() or is_done:
                policy_loss, state_value_loss = agent.update(batch)
                policy_losses.append(policy_loss)
                state_value_losses.append(state_value_loss)
                batch.clear()

        training_log["rewards"].append(float(total_reward))
        training_log["policy_losses"].append(float(np.mean(policy_losses)))
        training_log["state_value_losses"].append(float(np.mean(state_value_losses)))

        review_log["episodes"].append({
            "episode": episode,
            "seed": episode_seed,
            "reward": float(total_reward),
            "actions": "".join(str(taken_action) for taken_action in taken_actions),
        })

        average_reward = np.mean(training_log["rewards"][-MOVING_AVERAGE_WINDOW:])
        print(f"episode {episode} : reward = {total_reward:8.1f} | "
              f"average({MOVING_AVERAGE_WINDOW}) = {average_reward:8.1f} | "
              f"policy loss = {training_log['policy_losses'][-1]:9.4f} | "
              f"state value loss = {training_log['state_value_losses'][-1]:10.2f}")

except KeyboardInterrupt:
    print("\n학습 중단됨 (Ctrl+C) - 지금까지 결과를 저장합니다.")

finally:
    env.close()
    if training_log["rewards"]:
        save_all(training_log, review_log)
