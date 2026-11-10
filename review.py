'''
learning.py가 저장한 review_log_*.json을 불러와서 학습 중 에피소드를 그대로 재생한다.
(같은 seed로 reset + 같은 행동을 순서대로 넣으면 똑같은 게임이 재현된다)

사용법:
    python review.py                      # 가장 최근 로그 요약 + 최고 점수 에피소드 재생
    python review.py 9999                 # 9999번 에피소드 재생
    python review.py 0 5000 9999          # 여러 개를 순서대로 재생
    python review.py best                 # 최고 점수 에피소드
    python review.py worst                # 최저 점수 에피소드
    python review.py last                 # 마지막 에피소드
    python review.py 9999 --file review_log_xxx.json   # 특정 로그 파일
    python review.py --list               # 재생 없이 요약만 출력
'''
import os
import sys
import glob
import json
import argparse
import gymnasium as gym


MODELS_DIRECTORY = "models"


def find_latest_review_log_file():
    files = sorted(glob.glob(os.path.join(MODELS_DIRECTORY, "review_log_*.json")))
    if not files:
        sys.exit("models/review_log_*.json 파일이 없습니다. learning.py로 먼저 학습하세요.")
    return files[-1]


def resolve_file_name(file_name):
    # 파일 이름만 적어도 models/ 안에서 찾아준다.
    if not os.path.exists(file_name) and os.path.exists(os.path.join(MODELS_DIRECTORY, file_name)):
        return os.path.join(MODELS_DIRECTORY, file_name)
    return file_name


def load_review_log(file_name):
    with open(file_name, "r", encoding="utf-8") as file:
        return json.load(file)


def print_summary(review_log, file_name):
    episodes = review_log["episodes"]
    rewards = [episode["reward"] for episode in episodes]
    sorted_episodes = sorted(episodes, key=lambda episode: episode["reward"], reverse=True)

    print(f"log : {file_name}")
    print(f"episodes : {len(episodes)} | best = {max(rewards):.1f} | worst = {min(rewards):.1f} | "
          f"200점 이상 = {sum(reward >= 200 for reward in rewards)}개")
    print("top 5 :", ", ".join(f"#{episode['episode']} ({episode['reward']:.1f})" for episode in sorted_episodes[:5]))
    print("last 5 :", ", ".join(f"#{episode['episode']} ({episode['reward']:.1f})" for episode in episodes[-5:]))


def select_episodes(review_log, targets):
    episodes = review_log["episodes"]
    episode_by_number = {episode["episode"]: episode for episode in episodes}
    selected = []

    for target in targets:
        if target == "best":
            selected.append(max(episodes, key=lambda episode: episode["reward"]))
        elif target == "worst":
            selected.append(min(episodes, key=lambda episode: episode["reward"]))
        elif target == "last":
            selected.append(episodes[-1])
        elif target.isdigit() and int(target) in episode_by_number:
            selected.append(episode_by_number[int(target)])
        else:
            print(f"에피소드 '{target}'를 찾을 수 없습니다. (0 ~ {episodes[-1]['episode']}, best, worst, last)")
    return selected


def replay(review_log, episode):
    env = gym.make(review_log["environment"], render_mode="human")
    env.reset(seed=episode["seed"])
    total_reward = 0.0

    for action in episode["actions"]:
        _, reward, terminated, truncated, _ = env.step(int(action))
        total_reward += reward
        if terminated or truncated:
            break

    env.close()
    print(f"episode {episode['episode']} : steps = {len(episode['actions'])} | "
          f"replayed reward = {total_reward:.1f} (logged {episode['reward']:.1f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="학습 중 에피소드 재생")
    parser.add_argument("episodes", nargs="*", help="에피소드 번호 또는 best / worst / last")
    parser.add_argument("--file", help="review_log 파일 (기본: 가장 최근)")
    parser.add_argument("--list", action="store_true", help="재생 없이 요약만 출력")
    arguments = parser.parse_args()

    file_name = resolve_file_name(arguments.file) if arguments.file else find_latest_review_log_file()
    review_log = load_review_log(file_name)
    print_summary(review_log, file_name)

    if not arguments.list:
        for episode in select_episodes(review_log, arguments.episodes or ["best"]):
            replay(review_log, episode)
