import os
import json
import random
import time
import argparse
import logging
import requests
import subprocess
import wikienv, wrappers

def llm(prompt, model="llama3.2:1b", stop=None):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False   # 不要流式，直接返回完整结果
        }
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        output = response.json().get("response", "").strip()
        if stop:
            for s in stop:
                if s in output:
                    output = output.split(s)[0]
        return output
    except Exception as e:
        logging.error(f"LLM 调用失败: {e}")
        return ""


def init_env():
    # env = wikienv.WikiEnv()
    # env = wrappers.FeverWrapper(env, split="dev")
    # env = wrappers.LoggingWrapper(env)
    env = wikienv.WikiEnv()
    env = wrappers.LoggingWrapper(env)        # 先加 logging，保证有 steps
    env = wrappers.FeverWrapper(env, split="dev")

    return env


def step(env, action):
    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
            logging.warning(f"请求超时，重试 {attempts}/10 ...")
    raise RuntimeError("连续 10 次超时，放弃。")


def webthink(env, prompt_dict, idx=None, to_print=False):
    prompt = prompt_dict['webthink_simple3']
    question = env.reset(idx=idx)
    print(f"\n[Q{idx}] {question}")
    prompt += question + "\n"

    n_calls, n_badcalls, done = 0, 0, False
    turn = 3
    for i in range(1, turn):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
        # print(f"\n=== LLM 原始输出 (step {i}) ===\n{repr(thought_action)}\n")

        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except:
            logging.warning(f"解析错误: {thought_action}")
            n_badcalls += 1
            n_calls += 1
            # 宽松 fallback：抓第一行作为 thought
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()

        print(f"解析结果 → Thought {i}: {thought}\nAction {i}: {action}\n")

        if not action:
            print(f"⚠️ 空 action, 跳过 (LLM 输出 = {thought_action})")
            continue

        obs, r, done, info = step(env, action[0].lower() + action[1:])
        obs = obs.replace('\\n', '')

        print(f"env 返回 info: {info}")
        print(f"Observation {i}: {obs}\n")

        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str

        if done:
            print("✅ 提前结束 (done=True)")
            break

    # if not done:
    #     # 调试时强制填一个标签，避免空答案
    #     obs, r, done, info = step(env, "finish[NOT ENOUGH INFO]")
    #     print("\n⚠️ 没有 finish，被兜底填充: finish[NOT ENOUGH INFO]")

    print(f"\n最终 info: {info}\n")
    print(f"预测 = {info.get('answer')}, 真值 = {info.get('gt_answer')}\n")

    info.update({'n_calls': n_calls, 'n_badcalls': n_badcalls, 'traj': prompt})
    return r, info



def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--logdir", type=str, default="./example/log", help="日志和结果保存目录")
    argparser.add_argument("--n", type=int, default=500, help="运行多少条数据")
    argparser.add_argument("--checkpoint_every", type=int, default=50, help="多少条保存一次checkpoint")
    args = argparser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)

    #logging
    logfile = os.path.join(args.logdir, "experiment.log")
    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    env = init_env()

    with open("./prompts/fever.json", "r") as f:
        prompt_dict = json.load(f)

    idxs = list(range(7405))
    random.Random(233).shuffle(idxs)

    rs = []
    infos = []
    old_time = time.time()

    #checkpoint
    for i in idxs[:args.n]:
        r, info = webthink(env, prompt_dict, i, to_print=False)
        rs.append(info.get('em', 0))
        infos.append(info)

        avg_score = sum(rs) / len(rs)
        speed = (time.time() - old_time) / len(rs)
        msg = f"进度: {len(rs)} 条, 正确 {sum(rs)}, 平均准确率 {avg_score:.3f}, 每条耗时 {speed:.2f}s"
        logging.info(msg)
        print(msg)

        if len(rs) % args.checkpoint_every == 0:
            ckpt_file = os.path.join(args.logdir, f"checkpoint_{len(rs)}.json")
            with open(ckpt_file, "w") as f:
                json.dump(infos, f, indent=2)
            logging.info(f"已保存 {ckpt_file}")


    result_file = os.path.join(args.logdir, "results.json")
    with open(result_file, "w") as f:
        json.dump(infos, f, indent=2)
    logging.info(f"实验完成，结果已保存 {result_file}")


if __name__ == "__main__":
    main()
