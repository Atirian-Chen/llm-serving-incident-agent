# DeepSeek 在线 Agent 评测复习记录

本文件来自真实在线调用。失败案例计入全部 30 个案例的准确率分母。指标与日志来自本地 fixture，评测不代表真实生产集群上的诊断准确率。
工具选择准确率仅检查需要的工具名称是否出现在报告中；证据覆盖率仅检查关键词，不等同于工具参数正确或根因推理正确。

## 汇总

- 状态：已完成 30/30 个案例
- 成功/失败：30/0
- 故障类型准确率：100.00%
- 工具选择准确率：100.00%
- Schema 成功率：100.00%
- 证据词覆盖率：100.00%
- 全部已完成案例平均延迟：46276.21 ms
- 全部已完成案例 P95 延迟：90918.42 ms
- 成功案例平均延迟：46276.21 ms
- 成功案例 P95 延迟：90918.42 ms

## 运行来源

- Trace：[legacy_e2e.jsonl](traces/legacy_e2e.jsonl)
- 有 Hybrid 阶段事件的运行：0/30
- 验证状态：`hybrid_not_verified`

这组是历史在线结果，配套 trace 没有提供完整 Hybrid 阶段证据，RAG provider、strict 设置和最终 Top-k 均无法由这组日志核验。此前追加的 strict hybrid / Top-3 标记已移除。
已验证的真实 Hybrid 单案例见 [online_smoke.md](online_smoke.md) 和根目录 [example.md](../example.md)。单案例不能替代 30 条重测；本组也未覆盖独立的 SGLang 5xx、NCCL 故障 fixture。

## 两个完整在线案例

### oom-01

输入：
```json
{
  "service_id": "demo-oom",
  "symptom": "服务出现 CUDA out of memory，显存接近满载。"
}
```

实际工具调用及返回：
```json
[
  {
    "event": "tool_execution",
    "name": "get_metrics_snapshot",
    "arguments": {
      "service_id": "demo-oom"
    },
    "output": {
      "service_id": "demo-oom",
      "gpu_memory_utilization": 0.97,
      "waiting_requests": 18,
      "ttft_p95_ms": 920,
      "prefix_cache_hit_rate": 0.05,
      "request_success_rate": 0.81,
      "kv_cache_free_blocks": 12
    },
    "latency_ms": 1053.78
  },
  {
    "event": "tool_execution",
    "name": "search_logs",
    "arguments": {
      "service_id": "demo-oom",
      "keyword": "CUDA out of memory",
      "limit": 20
    },
    "output": [
      "2026-08-27T09:00:02Z ERROR CUDA out of memory while allocating KV cache"
    ],
    "latency_ms": 755.61
  }
]
```

最终模型输出：
```json
{
  "incident_type": "cuda_oom",
  "severity": "high",
  "root_cause": "日志确认 CUDA out of memory 发生在 KV Cache 分配阶段；同时 GPU 利用率达 0.97、KV Cache 空闲块仅剩 12、请求成功率 0.81，表明峰值时可供 KV Cache 分配的显存容量不足。prefix_cache_hit_rate=0.05 使请求难以复用已有 KV Cache，进一步放大了峰值显存需求。",
  "evidence": [
    {
      "source": "search_logs('demo-oom', 'CUDA out of memory')",
      "detail": "2026-08-27T09:00:02Z ERROR CUDA out of memory while allocating KV cache"
    },
    {
      "source": "get_metrics_snapshot('demo-oom')",
      "detail": "gpu_memory_utilization=0.97; request_success_rate=0.81; kv_cache_free_blocks=12; waiting_requests=18; ttft_p95_ms=920; prefix_cache_hit_rate=0.05"
    },
    {
      "source": "runbook: cuda_oom.md",
      "detail": "常见信号包括：日志出现 CUDA out of memory 或 OutOfMemoryError；gpu_memory_utilization 接近 1.0，部分请求失败。"
    },
    {
      "source": "runbook: gpu_memory.md",
      "detail": "可能根因包括：GPU memory utilization 长时间接近上限，权重、激活、KV Cache 与 CUDA Graph 共同占用显存。"
    }
  ],
  "recommended_actions": [
    "逐步降低 max_num_seqs 或 max_model_len，观察显存利用率和请求成功率变化。",
    "重新核算权重、激活、KV Cache 与 CUDA Graph 的显存预算，优先保障 KV Cache 可用容量。",
    "排查 prefix_cache_hit_rate=0.05 过低的原因，提升前缀复用以降低 KV Cache 峰值需求。"
  ],
  "tools_used": [
    "get_metrics_snapshot",
    "search_logs"
  ],
  "confidence": 0.93
}
```

### ttft-01

输入：
```json
{
  "service_id": "demo-high-ttft",
  "symptom": "TTFT P95 达到 1850ms，等待队列很长。"
}
```

实际工具调用及返回：
```json
[
  {
    "event": "tool_execution",
    "name": "get_metrics_snapshot",
    "arguments": {
      "service_id": "demo-high-ttft"
    },
    "output": {
      "service_id": "demo-high-ttft",
      "gpu_memory_utilization": 0.78,
      "waiting_requests": 64,
      "ttft_p95_ms": 1850,
      "prefix_cache_hit_rate": 0.42,
      "request_success_rate": 0.995,
      "kv_cache_free_blocks": 320
    },
    "latency_ms": 742.44
  },
  {
    "event": "tool_execution",
    "name": "search_logs",
    "arguments": {
      "service_id": "demo-high-ttft",
      "keyword": "scheduler queue",
      "limit": 20
    },
    "output": [
      "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
    ],
    "latency_ms": 709.07
  },
  {
    "event": "tool_execution",
    "name": "search_logs",
    "arguments": {
      "service_id": "demo-high-ttft",
      "keyword": "batching delay",
      "limit": 20
    },
    "output": [],
    "latency_ms": 737.75
  },
  {
    "event": "tool_execution",
    "name": "search_logs",
    "arguments": {
      "service_id": "demo-high-ttft",
      "keyword": "prefill backlog",
      "limit": 20
    },
    "output": [
      "2026-08-27T09:10:02Z WARN prefill backlog queue_time_ms=1700"
    ],
    "latency_ms": 784.03
  }
]
```

最终模型输出：
```json
{
  "incident_type": "high_ttft",
  "severity": "medium",
  "root_cause": "Prefill阶段排队/过载导致TTFT显著升高。日志显示prefill backlog queue_time_ms=1700，与TTFT P95=1850ms在量级上吻合，说明排队主要由prefill阶段引起；等待队列有64个请求，且无batching delay日志。GPU内存利用率为0.78、KV cache仍有空闲块，可排除显存或KV cache容量瓶颈。",
  "evidence": [
    {
      "source": "get_metrics_snapshot(demo-high-ttft)",
      "detail": "ttft_p95_ms=1850，waiting_requests=64，gpu_memory_utilization=0.78，kv_cache_free_blocks=320，prefix_cache_hit_rate=0.42，request_success_rate=0.995。"
    },
    {
      "source": "search_logs(demo-high-ttft, scheduler queue)",
      "detail": "WARN scheduler queue waiting_requests=64（2026-08-27T09:10:01Z）。"
    },
    {
      "source": "search_logs(demo-high-ttft, prefill backlog)",
      "detail": "WARN prefill backlog queue_time_ms=1700（2026-08-27T09:10:02Z）。"
    },
    {
      "source": "search_logs(demo-high-ttft, batching delay)",
      "detail": "未检索到batching delay相关日志。"
    },
    {
      "source": "runbook: high_ttft.md / scheduler_queue.md",
      "detail": "TTFT P95持续超过服务目标且等待队列增长时，需检查scheduler queue/prefill backlog；建议区分prefill和decode的排队时间，并调整并发/批处理参数。"
    }
  ],
  "recommended_actions": [
    "统计prefill与decode阶段的排队耗时，确认本次排队集中在prefill阶段后，调整并发参数（如max_num_seqs与请求长度的匹配），避免长请求挤占调度窗口。",
    "以固定负载回归测试吞吐、TTFT和P95的联合变化，验证参数调整效果。"
  ],
  "tools_used": [
    "get_metrics_snapshot",
    "search_logs",
    "search_logs",
    "search_logs"
  ],
  "confidence": 0.85
}
```

## 30 条案例明细

| # | case_id | 状态 | 预测 | 正确 | 工具正确 | Schema | 延迟 ms | 错误 |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- |
| 1 | oom-01 | 成功 | cuda_oom | 是 | 是 | 是 | 33437.19 |  |
| 2 | oom-02 | 成功 | cuda_oom | 是 | 是 | 是 | 44467.36 |  |
| 3 | oom-03 | 成功 | cuda_oom | 是 | 是 | 是 | 25029.03 |  |
| 4 | oom-04 | 成功 | cuda_oom | 是 | 是 | 是 | 21916.76 |  |
| 5 | oom-05 | 成功 | cuda_oom | 是 | 是 | 是 | 44654.23 |  |
| 6 | oom-06 | 成功 | cuda_oom | 是 | 是 | 是 | 45414.08 |  |
| 7 | oom-07 | 成功 | cuda_oom | 是 | 是 | 是 | 45593.26 |  |
| 8 | oom-08 | 成功 | cuda_oom | 是 | 是 | 是 | 25274.73 |  |
| 9 | ttft-01 | 成功 | high_ttft | 是 | 是 | 是 | 29469.19 |  |
| 10 | ttft-02 | 成功 | high_ttft | 是 | 是 | 是 | 36569.80 |  |
| 11 | ttft-03 | 成功 | high_ttft | 是 | 是 | 是 | 60449.93 |  |
| 12 | ttft-04 | 成功 | high_ttft | 是 | 是 | 是 | 36080.46 |  |
| 13 | ttft-05 | 成功 | high_ttft | 是 | 是 | 是 | 31852.20 |  |
| 14 | ttft-06 | 成功 | high_ttft | 是 | 是 | 是 | 52032.97 |  |
| 15 | ttft-07 | 成功 | high_ttft | 是 | 是 | 是 | 40945.48 |  |
| 16 | ttft-08 | 成功 | high_ttft | 是 | 是 | 是 | 45466.72 |  |
| 17 | prefix-01 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 58961.82 |  |
| 18 | prefix-02 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 47341.38 |  |
| 19 | prefix-03 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 110721.44 |  |
| 20 | prefix-04 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 25102.41 |  |
| 21 | prefix-05 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 90918.42 |  |
| 22 | prefix-06 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 77585.86 |  |
| 23 | prefix-07 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 61985.58 |  |
| 24 | prefix-08 | 成功 | low_prefix_cache_hit | 是 | 是 | 是 | 41941.35 |  |
| 25 | unknown-01 | 成功 | unknown | 是 | 是 | 是 | 74181.09 |  |
| 26 | unknown-02 | 成功 | unknown | 是 | 是 | 是 | 32523.08 |  |
| 27 | unknown-03 | 成功 | unknown | 是 | 是 | 是 | 62685.80 |  |
| 28 | unknown-04 | 成功 | unknown | 是 | 是 | 是 | 29022.21 |  |
| 29 | unknown-05 | 成功 | unknown | 是 | 是 | 是 | 36441.30 |  |
| 30 | unknown-06 | 成功 | unknown | 是 | 是 | 是 | 20221.19 |  |
