import os
import json
import random
import time
import argparse
import logging
import requests
import subprocess
import wikienv, wrappers
from collections import Counter

# ====== 可调参数 ======
MODEL_NAME = "llama3.2:1b"
MAX_STEPS_FEVER = 5              # 论文设置：FEVER = 5 步
COT_SAMPLES = 21                 # CoT-SC 自一致采样条数
COT_TEMPERATURE = 0.7            # 采样温度
EXTRA_REACT_STEPS = 2            # CoT-SC 置信度不足时，再补跑 ReAct 的步数
ALLOWED_PREFIX = ("search[", "lookup[", "finish[", "think[")

def llm(prompt, model=MODEL_NAME, stop=None, timeout_s=60):
    try:
        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        response = requests.post(url, json=payload, timeout=timeout_s)
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
    env = wikienv.WikiEnv()
    env = wrappers.LoggingWrapper(env)        # 记录步数/日志
    env = wrappers.FeverWrapper(env, split="dev")
    return env

def step(env, action):
    # 规范化 + 简单白名单
    action = normalize_action(action)
    if not is_valid_action(action):
        return ("Invalid action: {}".format(action), 0, False, env._get_info() if hasattr(env, "_get_info") else {})

    attempts = 0
    while attempts < 10:
        try:
            return env.step(action)
        except requests.exceptions.Timeout:
            attempts += 1
            logging.warning(f"请求超时，重试 {attempts}/10 ...")
    raise RuntimeError("连续 10 次超时，放弃。")

def normalize_action(action: str) -> str:
    """把开头字母转小写，去掉多余空白。"""
    if not action:
        return action
    a = action.strip()
    return a[0].lower() + a[1:] if a else a

def is_valid_action(action: str) -> bool:
    a = action.strip().lower()
    return any(a.startswith(p) and a.endswith("]") for p in ALLOWED_PREFIX)

def cot_answer(question, n=COT_SAMPLES, temperature=COT_TEMPERATURE, model=MODEL_NAME):
    """CoT-SC：采样 n 条 CoT，做多数表决。返回 (majority, votes, all_answers)"""
    prompt = (
        "You are a helpful fact-checker. Think step by step, then answer with only one of:\n"
        "SUPPORTS, REFUTES, or NOT ENOUGH INFO.\n\n"
        f"Question/Claim: {question}\n"
        "Reasoning:"
    )
    answers = []
    for _ in range(n):
        out = llm(prompt, model=model)
        up = out.upper()
        if "SUPPORTS" in up:
            answers.append("SUPPORTS")
        elif "REFUTES" in up:
            answers.append("REFUTES")
        elif "NOT ENOUGH INFO" in up:
            answers.append("NOT ENOUGH INFO")
        else:
            # 无法解析时忽略或标注为未知
            answers.append("NOT ENOUGH INFO")
    cnt = Counter(answers)
    maj, maj_cnt = cnt.most_common(1)[0]
    return maj, maj_cnt, answers

def webthink(env, prompt_dict, idx=None, to_print=False):
    prompt = prompt_dict['webthink_simple3']
    question = env.reset(idx=idx)
    print(f"\n[Q{idx}] {question}")
    prompt += question + "\n"

    n_calls, n_badcalls, done = 0, 0, False
    info, r = {}, 0

    # ====== 第一阶段：ReAct 主循环（最多 MAX_STEPS_FEVER 步）======
    for i in range(1, MAX_STEPS_FEVER + 1):
        n_calls += 1
        thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:", f"\nThought {i+1}:"])
        # 解析 Thought / Action
        try:
            thought, action = thought_action.strip().split(f"\nAction {i}: ")
        except Exception:
            logging.warning(f"解析错误: {thought_action}")
            n_badcalls += 1
            n_calls += 1
            thought = thought_action.strip().split('\n')[0]
            action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()

        print(f"解析结果 → Thought {i}: {thought}\nAction {i}: {action}\n")
        if not action:
            print(f"空 action, 跳过 (LLM 输出 = {thought_action})")
            continue

        obs, r, done, info = step(env, action)
        obs = obs.replace('\\n', '')

        print(f"env 返回 info: {info}")
        print(f"Observation {i}: {obs}\n")

        step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
        prompt += step_str

        if done:
            print("✅ 提前结束 (done=True)")
            break

    react_finished = bool(done and info.get('answer') is not None)

    # ====== 第二阶段：回退 A（ReAct → CoT-SC）======
    if not react_finished:
        maj, maj_cnt, all_ans = cot_answer(question)
        print(f"\n[CoT-SC] majority={maj} (votes={maj_cnt}/{COT_SAMPLES})\n")

        # 多数阈值：严格过半（例如 21 -> 11）
        majority_need = (COT_SAMPLES // 2) + 1
        if maj is not None and maj_cnt >= majority_need:
            info['answer'] = maj
            info['mode'] = 'CoT-SC'
            info['cot_votes'] = maj_cnt
            info['cot_all'] = all_ans
            r = 0
            print(f"采用 CoT-SC 多数表决结果 → {maj}")
        else:
            # ====== 第三阶段：回退 B（CoT-SC → ReAct 再补跑）======
            print("[CoT-SC] 多数不足，回退再跑一次 ReAct")
            for j in range(EXTRA_REACT_STEPS):
                i = MAX_STEPS_FEVER + j + 1
                n_calls += 1
                thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:", f"\nThought {i+1}:"])
                try:
                    thought, action = thought_action.strip().split(f"\nAction {i}: ")
                except Exception:
                    logging.warning(f"解析错误: {thought_action}")
                    n_badcalls += 1
                    n_calls += 1
                    thought = thought_action.strip().split('\n')[0]
                    action = llm(prompt + f"Thought {i}: {thought}\nAction {i}:", stop=["\n"]).strip()

                print(f"解析结果 → Thought {i}: {thought}\nAction {i}: {action}\n")
                if not action:
                    continue

                obs, r, done, info = step(env, action)
                obs = obs.replace('\\n', '')
                print(f"env 返回 info: {info}")
                print(f"Observation {i}: {obs}\n")

                step_str = f"Thought {i}: {thought}\nAction {i}: {action}\nObservation {i}: {obs}\n"
                prompt += step_str

                if done:
                    print("✅ 回退 ReAct 成功结束")
                    break

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

    # logging
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

    # checkpoint
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
