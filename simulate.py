'''
학습된 모델(ai_model_*.json)을 불러와서 LunarLander를 화면에 띄워 시연한다.
행동은 확률 샘플링이 아니라 argmax(greedy)로 고른다.

사용법:
    python simulate.py                       # 가장 최근 ai_model_*.json, 5 에피소드
    python simulate.py ai_model_xxx.json     # 특정 파일
    python simulate.py ai_model_xxx.json 10  # 특정 파일, 10 에피소드
'''
import sys
import glob
import json
import numpy as np
import gymnasium as gym
from poilcy_neural_network import PolicyNeuralNetwork


def find_latest_model_file():
    files = sorted(glob.glob("ai_model_*.json"))
    if not files:
        sys.exit("ai_model_*.json 파일이 없습니다. learning.py로 먼저 학습하세요.")
    return files[-1]


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


def get_greedy_action(policy_neural_network, state):
    state = np.array(state).reshape(1, -1)
    actions_probabilities = policy_neural_network.get_actions_probabilities(state)[0]
    return int(np.argmax(actions_probabilities))


def simulate(file_name, episodes):
    policy_neural_network = load_policy_neural_network(file_name)
    env = gym.make("LunarLander-v3", render_mode="human")
    print(f"model : {file_name}")

    for episode in range(episodes):
        observation, _ = env.reset()
        total_reward = 0.0
        is_done = False

        while not is_done:
            action = get_greedy_action(policy_neural_network, observation)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            is_done = terminated or truncated

        result = "착륙 성공" if total_reward >= 200 else "실패"
        print(f"episode {episode} : total reward = {total_reward:.1f} ({result})")

    env.close()


if __name__ == "__main__":
    file_name = sys.argv[1] if len(sys.argv) > 1 else find_latest_model_file()
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    simulate(file_name, episodes)
