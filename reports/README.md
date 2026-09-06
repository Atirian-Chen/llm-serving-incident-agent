# 报告与运行证据

| 文件 | 内容 | 可以证明的范围 |
| --- | --- | --- |
| [example.md](../example.md) | 真实在线单案例的可读时间线 | CUDA Hybrid Top-3、DeepSeek 响应、MCP fixture 工具调用 |
| [rag_eval.md](rag_eval.md) / [JSON](rag_eval.json) | 24 queries，五种检索方法 | 已保存真实排序结果的 source 级指标；本次重算，未重新推理 |
| [online_smoke.md](online_smoke.md) / [JSON](online_smoke.json) | 一次真实 Hybrid 在线运行核验 | 1 个成功案例，不能当成 30 条 Hybrid 回归 |
| [e2e_results.md](e2e_results.md) / [JSON](e2e_results.json) | 30 条历史 DeepSeek 在线结果 | 4 组固定 fixture；缺少 Hybrid 阶段证据，provider 未核验 |
| [traces/online_smoke.jsonl](traces/online_smoke.jsonl) | 在线单案例原始快照 | 对应 example 与 online_smoke，保留 Run ID 和时间 |
| [traces/legacy_e2e.jsonl](traces/legacy_e2e.jsonl) | 历史评测原始快照 | 保留旧结果来源，不能补出未记录的阶段 |

RAG 指标已修正旧版分母错误：旧值将每种方法的命中数除以全部 120 条方法/query 组合。现在分别报告真正的 source Recall 和 query Hit rate。标注仍是 source 级，Keyword 是当前 token-overlap 实现，尚不等同于原始中文单字 baseline。

本组数据中的 reranker 排序指标低于 RRF，不应宣传为效果提升。历史 E2E 的 100% 仅针对这些固定 fixture 和有限检查项，不是生产可靠性。

## 重建展示

从仓库内的在线快照重建根目录示例，不下载模型、不调用 API：

```powershell
.\.venv-standard\Scripts\python.exe scripts/generate_example.py
```

维护者可用本地 `data/online_current_trace.jsonl`、`data/online_evaluation_trace.jsonl` 和历史 `data/online_evaluation.json` 重新整理全部报告：

```powershell
.\.venv-standard\Scripts\python.exe scripts/prepare_reports.py
```

新评测请分别运行 `scripts/rag_eval.py --mode real` 和 `scripts/evaluate.ps1`。`prepare_reports.py` 用于整理这些已有历史输入，不是执行新的评测。

## 文件保留

`reports/traces/` 只存放已检查、用于展示的 fixture 日志快照。日常日志、Chroma 索引、模型和 pip 缓存分别留在被 Git 忽略的 `data/`、`.runtime/`。密钥仍放在本地 `config/llm.toml`。
