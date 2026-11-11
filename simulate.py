'''
학습된 모델(ai_model_*.json)을 불러와서 LunarLander를 화면에 띄워 시연한다.
행동 선택 방식은 아래 IS_GREEDY로 정한다.
    True  : argmax(greedy) - 항상 확률이 가장 큰 행동
    False : 샘플링 - 학습 때와 같이 확률대로 행동을 뽑음

사용법:
    python simulate.py                       # 가장 최근 ai_model_*.json, 5 에피소드
    python simulate.py ai_model_xxx.json     # 특정 파일
    python simulate.py ai_model_xxx.json 10  # 특정 파일, 10 에피소드
'''
import os
import sys
import glob
import json
import numpy as np
import gymnasium as gym
from poilcy_neural_network import PolicyNeuralNetwork


MODELS_DIRECTORY = "models"
IS_GREEDY = True # True: argmax(greedy) / False: 확률 샘플링


def find_latest_model_file():
    # models/ 를 먼저 찾고, 없으면 예전처럼 현재 폴더에서 찾는다.
    files = sorted(glob.glob(os.path.join(MODELS_DIRECTORY, "ai_model_*.json"))) or sorted(glob.glob("ai_model_*.json"))
    if not files:
        sys.exit("ai_model_*.json 파일이 없습니다. learning.py로 먼저 학습하세요.")
    return files[-1]


def resolve_file_name(file_name):
    # 파일 이름만 적어도 models/ 안에서 찾아준다.
    if not os.path.exists(file_name) and os.path.exists(os.path.join(MODELS_DIRECTORY, file_name)):
        return os.path.join(MODELS_DIRECTORY, file_name)
    return file_name


def load_policy_neural_network(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        data = json.load(file)

    # 예전 형식: [policy, state_value] 리스트 / 현재 형식: {'policy_neural_network': ..., ...}
    policy_data = data[0] if isinstance(data, list) else data['policy_neural_network']

    policy_neural_network = PolicyNeuralNetwork()
    for name, parameters in policy_data.items():
        layer = policy_neural_network.layers[name]
        layer.W = np.array(parameters['W'])
        layer.b = np.array(parameters['b'])
    return policy_neural_network


def get_action(policy_neural_network, state):
    state = np.array(state).reshape(1, -1)
    actions_probabilities = policy_neural_network.get_actions_probabilities(state)[0]
    if IS_GREEDY:
        return int(np.argmax(actions_probabilities))
    return int(np.random.choice(len(actions_probabilities), p=actions_probabilities))


def simulate(file_name, episodes):
    policy_neural_network = load_policy_neural_network(file_name)
    env = gym.make("LunarLander-v3", render_mode="human")
    print(f"model : {file_name} | action : {'greedy (argmax)' if IS_GREEDY else 'sampling'}")

    for episode in range(episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        is_done = False

        while not is_done:
            action = get_action(policy_neural_network, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            is_done = terminated or truncated

        result = "착륙 성공" if total_reward >= 200 else "실패"
        print(f"episode {episode} : total reward = {total_reward:.1f} ({result})")

    env.close()


if __name__ == "__main__":
    file_name = resolve_file_name(sys.argv[1]) if len(sys.argv) > 1 else find_latest_model_file()
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    simulate(file_name, episodes)
