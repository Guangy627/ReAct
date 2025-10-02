# ReAct Fever Fact-Checking

## 使用方法

入口脚本为 **`run.sh`**，通过命令行参数控制实验。

---

## 示例

```bash
./run.sh --logdir ./logs/fever --n 100 --checkpoint_every 10
```

---

## 参数说明

```text
--logdir           保存日志和结果的目录 (默认: ./example/log)
--n                运行多少条数据 (默认: 500)
--checkpoint_every 每多少条保存一次 checkpoint (默认: 50)
```

---

## 输出示例

```text
进度: 15 条, 正确 1, 平均准确率 0.067, 每条耗时 4.67s
```

保存目录结构：

```text
logs/
 ├─ fever/
 │   ├─ experiment.log
 │   ├─ checkpoint_10.json
 │   ├─ checkpoint_20.json
 │   └─ results.json
```

---

## 成功案例

### [Q3543] Claim: Andrew Kevin Walker is from North America.

```text
Thought 1: I should search for Andrew Kevin Walker.
Action 1: Search[Andrew Kevin Walker]

Observation 1: Andrew Kevin Walker is an American screenwriter, born in Altoona, Pennsylvania.

Thought 2: Since he was born in a US city, he is from North America.
Action 2: Finish[SUPPORTS]
```

**最终结果**

```text
answer = SUPPORTS
gt_answer = SUPPORTS
✅ 正确
```

---

## 常见问题

```text
- Observation 为空 → Wikipedia 返回 403，需要设置 User-Agent 或改用本地数据
- Invalid action → 模型输出未严格遵循格式，必须使用 Search[...] / Lookup[...] / Finish[...]
- 中断恢复 → 直接加载 --logdir 下的 checkpoint 文件继续
```

---

## 改进方向

```text
- 强化 Prompt 约束，减少 invalid action
- 增加 fallback 策略（search 失败时换 query）
- 强制 episode 以 Finish[...] 结束
```
