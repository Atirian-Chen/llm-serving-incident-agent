# CUDA OOM

## 常见信号
- 日志出现 `CUDA out of memory` 或 `OutOfMemoryError`。
- `gpu_memory_utilization` 接近 1.0，部分请求失败。

## 可能根因
- `max_model_len` 或 `max_num_seqs` 过大。
- KV Cache 预算不足，批处理瞬时占用过高。

## 建议动作
- 先保留指标和日志证据，再逐步降低 `max_num_seqs` 或 `max_model_len`。
- 检查模型权重、KV Cache 与 CUDA Graph 的显存预算。

## 风险提示
- 没有测量证据时不要直接修改生产参数；修改后需要回归吞吐和质量。

