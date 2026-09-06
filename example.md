# Hybrid RAG 真实在线流程示例

本页从已保存的真实运行日志整理：CUDA BGE + Chroma + Cross-Encoder，DeepSeek 在线响应，官方 MCP SDK 调用本地 fixture 服务。没有重新请求模型，也没有使用测试替身生成本页的诊断。

- Run ID：`28c990bb-c43e-44bc-afa7-25ab78190a2a`
- 原始运行时间：`2026-09-04T16:02:46.709000+00:00`
- 原始日志：[online_smoke.jsonl](reports/traces/online_smoke.jsonl)（完整内容与精度以 JSONL 为准）
- SHA-256：`57c491a88ddde23de5ec884c6099d04460eec3d7a4c6337f8ae4c98d6b82b58b`
- 在线模型：`deepseek-v4-flash`；总耗时：68.31 秒；工具调用：6 次

日志边界：这里只保存了最终 Top-3 的 Cross-Encoder 分数；其余候选的精排分数未记录。最终结构化响应保存的是解析后的 IncidentReport，未保留该次 HTTP 原始响应。工具结果通过下一轮 user JSON context 送入模型，日志中的 tool_message 是展示对象，不是实际发送的原生 ToolMessage。

版本说明：当前代码会在发送给 LLM 前去掉检索分数、排名及重复元数据，只保留排序后的正文、引用信息和必要元数据。下方忠实展示快照中的实际请求；历史请求可能仍含上述检索字段，不代表当前消息格式，也不会通过改写旧日志来模拟新运行。

## 用户输入

```json
{
  "service_id": "demo-high-ttft",
  "symptom": "vLLM 请求排队增长，TTFT 和 P99 延迟升高"
}
```

## 检索过程

- Embedding：`BAAI/bge-small-zh-v1.5`；Reranker：`BAAI/bge-reranker-v2-m3`；Device：`cuda`
- 候选数：BM25=20，Dense=20，去重=28，RRF=28，最终=3
- 耗时（ms）：BM25=0.39，Dense=38.06，RRF=0.1，Cross-Encoder=212.3
- 表格中的 `-` 表示该阶段未召回或日志没有此项，不代表零分。

### BM25 候选

| chunk_id | BM rank | BM score | Dense rank | Cosine | RRF rank | RRF score | Cross rank | Cross score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vllm_queue_batching.md::115::故障现象 | 1 | 8.975357 | - | - | - | - | - | - |
| prefill_decode_latency.md::70::故障现象 | 2 | 8.672922 | - | - | - | - | - | - |
| gpu_utilization_anomaly.md::23::故障现象 | 3 | 7.073209 | - | - | - | - | - | - |
| autoscaling_capacity.md::0::故障现象 | 4 | 6.925619 | - | - | - | - | - | - |
| rate_limit_retry.md::86::故障现象 | 5 | 6.411361 | - | - | - | - | - | - |
| scheduler_queue.md::92::建议动作 | 6 | 6.218017 | - | - | - | - | - | - |
| multi_instance_load_balance.md::47::需要检查的指标和日志 | 7 | 6.193233 | - | - | - | - | - | - |
| continuous_batching.md::4::故障现象 | 8 | 5.634264 | - | - | - | - | - | - |
| ttft_tpot_p99.md::106::需要检查的指标和日志 | 9 | 5.614938 | - | - | - | - | - | - |
| vllm_queue_batching.md::116::需要检查的指标和日志 | 10 | 5.147069 | - | - | - | - | - | - |
| ncuda_oom_fragmentation.md::60::排查步骤 | 11 | 5.007074 | - | - | - | - | - | - |
| high_ttft.md::34::风险提示 | 12 | 4.221894 | - | - | - | - | - | - |
| vllm_paged_attention.md::111::故障现象 | 13 | 4.131542 | - | - | - | - | - | - |
| logging_observability.md::39::需要检查的指标和日志 | 14 | 3.846659 | - | - | - | - | - | - |
| high_ttft.md::31::常见信号 | 15 | 3.756172 | - | - | - | - | - | - |
| continuous_batching.md::7::修复建议与验证 | 16 | 3.734573 | - | - | - | - | - | - |
| multi_instance_load_balance.md::46::故障现象 | 17 | 3.327042 | - | - | - | - | - | - |
| multi_instance_load_balance.md::49::修复建议与验证 | 18 | 3.316141 | - | - | - | - | - | - |
| service_5xx_timeout.md::95::排查步骤 | 19 | 3.147763 | - | - | - | - | - | - |
| high_ttft.md::33::建议动作 | 20 | 3.120696 | - | - | - | - | - | - |

### Dense 候选

| chunk_id | BM rank | BM score | Dense rank | Cosine | RRF rank | RRF score | Cross rank | Cross score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill_decode_latency.md::70::故障现象 | - | - | 1 | 0.730290 | - | - | - | - |
| vllm_queue_batching.md::115::故障现象 | - | - | 2 | 0.714514 | - | - | - | - |
| high_ttft.md::31::常见信号 | - | - | 3 | 0.697203 | - | - | - | - |
| scheduler_queue.md::92::建议动作 | - | - | 4 | 0.677143 | - | - | - | - |
| vllm_queue_batching.md::118::修复建议与验证 | - | - | 5 | 0.676414 | - | - | - | - |
| scheduler_queue.md::90::常见信号 | - | - | 6 | 0.674700 | - | - | - | - |
| rate_limit_retry.md::86::故障现象 | - | - | 7 | 0.648478 | - | - | - | - |
| ttft_tpot_p99.md::105::故障现象 | - | - | 8 | 0.646631 | - | - | - | - |
| ncaching_prefix_kv.md::50::故障现象 | - | - | 9 | 0.642501 | - | - | - | - |
| high_ttft.md::34::风险提示 | - | - | 10 | 0.640159 | - | - | - | - |
| autoscaling_capacity.md::0::故障现象 | - | - | 11 | 0.636604 | - | - | - | - |
| gpu_utilization_anomaly.md::26::修复建议与验证 | - | - | 12 | 0.635629 | - | - | - | - |
| high_ttft.md::33::建议动作 | - | - | 13 | 0.624988 | - | - | - | - |
| continuous_batching.md::7::修复建议与验证 | - | - | 14 | 0.622224 | - | - | - | - |
| throughput_regression.md::101::故障现象 | - | - | 15 | 0.620905 | - | - | - | - |
| high_ttft.md::32::可能根因 | - | - | 16 | 0.614652 | - | - | - | - |
| multi_instance_load_balance.md::46::故障现象 | - | - | 17 | 0.614492 | - | - | - | - |
| continuous_batching.md::4::故障现象 | - | - | 18 | 0.611184 | - | - | - | - |
| prefill_decode_latency.md::73::修复建议与验证 | - | - | 19 | 0.608791 | - | - | - | - |
| gpu_utilization_anomaly.md::23::故障现象 | - | - | 20 | 0.597366 | - | - | - | - |

### RRF 融合

| chunk_id | BM rank | BM score | Dense rank | Cosine | RRF rank | RRF score | Cross rank | Cross score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| prefill_decode_latency.md::70::故障现象 | 2 | 8.672922 | 1 | 0.730290 | 1 | 0.032522 | - | - |
| vllm_queue_batching.md::115::故障现象 | 1 | 8.975357 | 2 | 0.714514 | 2 | 0.032522 | - | - |
| scheduler_queue.md::92::建议动作 | 6 | 6.218017 | 4 | 0.677143 | 3 | 0.030777 | - | - |
| rate_limit_retry.md::86::故障现象 | 5 | 6.411361 | 7 | 0.648478 | 4 | 0.030310 | - | - |
| autoscaling_capacity.md::0::故障现象 | 4 | 6.925619 | 11 | 0.636604 | 5 | 0.029710 | - | - |
| high_ttft.md::31::常见信号 | 15 | 3.756172 | 3 | 0.697203 | 6 | 0.029206 | - | - |
| gpu_utilization_anomaly.md::23::故障现象 | 3 | 7.073209 | 20 | 0.597366 | 7 | 0.028373 | - | - |
| high_ttft.md::34::风险提示 | 12 | 4.221894 | 10 | 0.640159 | 8 | 0.028175 | - | - |
| continuous_batching.md::4::故障现象 | 8 | 5.634264 | 18 | 0.611184 | 9 | 0.027526 | - | - |
| continuous_batching.md::7::修复建议与验证 | 16 | 3.734573 | 14 | 0.622224 | 10 | 0.026671 | - | - |
| high_ttft.md::33::建议动作 | 20 | 3.120696 | 13 | 0.624988 | 11 | 0.026199 | - | - |
| multi_instance_load_balance.md::46::故障现象 | 17 | 3.327042 | 17 | 0.614492 | 12 | 0.025974 | - | - |
| vllm_queue_batching.md::118::修复建议与验证 | - | - | 5 | 0.676414 | 13 | 0.015385 | - | - |
| scheduler_queue.md::90::常见信号 | - | - | 6 | 0.674700 | 14 | 0.015152 | - | - |
| multi_instance_load_balance.md::47::需要检查的指标和日志 | 7 | 6.193233 | - | - | 15 | 0.014925 | - | - |
| ttft_tpot_p99.md::105::故障现象 | - | - | 8 | 0.646631 | 16 | 0.014706 | - | - |
| ncaching_prefix_kv.md::50::故障现象 | - | - | 9 | 0.642501 | 17 | 0.014493 | - | - |
| ttft_tpot_p99.md::106::需要检查的指标和日志 | 9 | 5.614938 | - | - | 18 | 0.014493 | - | - |
| vllm_queue_batching.md::116::需要检查的指标和日志 | 10 | 5.147069 | - | - | 19 | 0.014286 | - | - |
| ncuda_oom_fragmentation.md::60::排查步骤 | 11 | 5.007074 | - | - | 20 | 0.014085 | - | - |
| gpu_utilization_anomaly.md::26::修复建议与验证 | - | - | 12 | 0.635629 | 21 | 0.013889 | - | - |
| vllm_paged_attention.md::111::故障现象 | 13 | 4.131542 | - | - | 22 | 0.013699 | - | - |
| logging_observability.md::39::需要检查的指标和日志 | 14 | 3.846659 | - | - | 23 | 0.013514 | - | - |
| throughput_regression.md::101::故障现象 | - | - | 15 | 0.620905 | 24 | 0.013333 | - | - |
| high_ttft.md::32::可能根因 | - | - | 16 | 0.614652 | 25 | 0.013158 | - | - |
| multi_instance_load_balance.md::49::修复建议与验证 | 18 | 3.316141 | - | - | 26 | 0.012821 | - | - |
| prefill_decode_latency.md::73::修复建议与验证 | - | - | 19 | 0.608791 | 27 | 0.012658 | - | - |
| service_5xx_timeout.md::95::排查步骤 | 19 | 3.147763 | - | - | 28 | 0.012658 | - | - |

### Cross-Encoder 最终 Top-3

| chunk_id | BM rank | BM score | Dense rank | Cosine | RRF rank | RRF score | Cross rank | Cross score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| vllm_queue_batching.md::115::故障现象 | 1 | 8.975357 | 2 | 0.714514 | 2 | 0.032522 | 1 | 0.892369 |
| scheduler_queue.md::90::常见信号 | - | - | 6 | 0.674700 | 14 | 0.015152 | 2 | 0.847270 |
| prefill_decode_latency.md::70::故障现象 | 2 | 8.672922 | 1 | 0.730290 | 1 | 0.032522 | 3 | 0.832330 |

### vLLM 请求排队与批处理 / 故障现象

```json
{
  "chunk_id": "vllm_queue_batching.md::115::故障现象",
  "text": "vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT，常见于 max_num_batched_tokens 过小或 decode token 被长 prompt 挤压。",
  "source": "vllm_queue_batching.md",
  "title": "vLLM 请求排队与批处理",
  "section": "故障现象",
  "metadata": {
    "source": "vllm_queue_batching.md",
    "title": "vLLM 请求排队与批处理",
    "section": "故障现象"
  },
  "bm25_score": 8.975357099415039,
  "bm25_rank": 1,
  "rrf_score": 0.03252247488101534,
  "dense_score": 0.714513897895813,
  "dense_rank": 2,
  "rrf_rank": 2,
  "reranker_score": 0.8923693895339966,
  "reranker_rank": 1
}
```

### Scheduler queue backlog / 常见信号

```json
{
  "source": "scheduler_queue.md",
  "section": "常见信号",
  "title": "Scheduler queue backlog",
  "chunk_id": "scheduler_queue.md::90::常见信号",
  "text": "- waiting requests 持续增长，TTFT P95 同步变差。\n- 日志出现 scheduler queue timeout 或 batch waiting。",
  "dense_score": 0.6747003197669983,
  "dense_rank": 6,
  "metadata": {
    "source": "scheduler_queue.md",
    "section": "常见信号",
    "title": "Scheduler queue backlog"
  },
  "rrf_score": 0.015151515151515152,
  "bm25_score": null,
  "bm25_rank": null,
  "rrf_rank": 14,
  "reranker_score": 0.8472697138786316,
  "reranker_rank": 2
}
```

### Prefill 与 Decode 延迟 / 故障现象

```json
{
  "chunk_id": "prefill_decode_latency.md::70::故障现象",
  "text": "TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用、KV cache 或通信瓶颈。",
  "source": "prefill_decode_latency.md",
  "title": "Prefill 与 Decode 延迟",
  "section": "故障现象",
  "metadata": {
    "source": "prefill_decode_latency.md",
    "title": "Prefill 与 Decode 延迟",
    "section": "故障现象"
  },
  "bm25_score": 8.67292230641757,
  "bm25_rank": 2,
  "rrf_score": 0.03252247488101534,
  "dense_score": 0.7302896976470947,
  "dense_rank": 1,
  "rrf_rank": 1,
  "reranker_score": 0.8323298692703247,
  "reranker_rank": 3
}
```

## LLM 与工具调用时间线

按原始事件顺序展开。每轮请求的 JSON content 已格式化，便于查看实际 RAG context 和工具证据；原始序列化消息保留在 JSONL 中。

### 第 1 轮请求：tool_selection

#### system

```text
你是 LLM serving SRE 诊断助手。
你只能使用已经提供的只读工具，不得修改线上配置。
先结合排障手册判断还缺少什么证据；证据不足时调用 get_metrics_snapshot 或 search_logs。
工具参数中的 service_id 必须使用当前请求的 service_id。证据足够后停止调用工具。
最终报告必须只基于请求、排障手册和工具返回的实际证据；证据不足时选择 unknown，不能把推测写成事实。

```

#### human

```json
{
  "request": {
    "service_id": "demo-high-ttft",
    "symptom": "vLLM 请求排队增长，TTFT 和 P99 延迟升高"
  },
  "runbook": [
    {
      "chunk_id": "vllm_queue_batching.md::115::故障现象",
      "text": "vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT，常见于 max_num_batched_tokens 过小或 decode token 被长 prompt 挤压。",
      "source": "vllm_queue_batching.md",
      "title": "vLLM 请求排队与批处理",
      "section": "故障现象",
      "metadata": {
        "source": "vllm_queue_batching.md",
        "title": "vLLM 请求排队与批处理",
        "section": "故障现象"
      },
      "bm25_score": 8.975357099415039,
      "bm25_rank": 1,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.714513897895813,
      "dense_rank": 2,
      "rrf_rank": 2,
      "reranker_score": 0.8923693895339966,
      "reranker_rank": 1
    },
    {
      "source": "scheduler_queue.md",
      "section": "常见信号",
      "title": "Scheduler queue backlog",
      "chunk_id": "scheduler_queue.md::90::常见信号",
      "text": "- waiting requests 持续增长，TTFT P95 同步变差。\n- 日志出现 scheduler queue timeout 或 batch waiting。",
      "dense_score": 0.6747003197669983,
      "dense_rank": 6,
      "metadata": {
        "source": "scheduler_queue.md",
        "section": "常见信号",
        "title": "Scheduler queue backlog"
      },
      "rrf_score": 0.015151515151515152,
      "bm25_score": null,
      "bm25_rank": null,
      "rrf_rank": 14,
      "reranker_score": 0.8472697138786316,
      "reranker_rank": 2
    },
    {
      "chunk_id": "prefill_decode_latency.md::70::故障现象",
      "text": "TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用、KV cache 或通信瓶颈。",
      "source": "prefill_decode_latency.md",
      "title": "Prefill 与 Decode 延迟",
      "section": "故障现象",
      "metadata": {
        "source": "prefill_decode_latency.md",
        "title": "Prefill 与 Decode 延迟",
        "section": "故障现象"
      },
      "bm25_score": 8.67292230641757,
      "bm25_rank": 2,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.7302896976470947,
      "dense_rank": 1,
      "rrf_rank": 1,
      "reranker_score": 0.8323298692703247,
      "reranker_rank": 3
    }
  ],
  "tool_results": [],
  "instruction": "选择必要的只读工具。已有足够证据时不要再调用工具。"
}
```

### 第 1 轮：原始模型消息（含工具调用）

```json
{
  "content": "I'll gather evidence from the metrics snapshot and logs for the service.",
  "additional_kwargs": {
    "tool_calls": [
      {
        "id": "call_00_uiJFIbDje1lQVw4TqtvT4797",
        "function": {
          "arguments": "{\"service_id\": \"demo-high-ttft\"}",
          "name": "get_metrics_snapshot"
        },
        "type": "function",
        "index": 0
      },
      {
        "id": "call_01_wHxiM7j9elc3ASyjzqpL2258",
        "function": {
          "arguments": "{\"service_id\": \"demo-high-ttft\", \"keyword\": \"scheduler\", \"limit\": 20}",
          "name": "search_logs"
        },
        "type": "function",
        "index": 1
      },
      {
        "id": "call_02_2rdDEDqojIzdGNASeQcj9179",
        "function": {
          "arguments": "{\"service_id\": \"demo-high-ttft\", \"keyword\": \"waiting\", \"limit\": 20}",
          "name": "search_logs"
        },
        "type": "function",
        "index": 2
      }
    ],
    "refusal": null
  },
  "response_metadata": {
    "token_usage": {
      "completion_tokens": 444,
      "prompt_tokens": 1301,
      "total_tokens": 1745,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": null,
        "reasoning_tokens": 240,
        "rejected_prediction_tokens": null
      },
      "prompt_tokens_details": {
        "audio_tokens": null,
        "cache_write_tokens": null,
        "cached_tokens": 512
      },
      "prompt_cache_hit_tokens": 512,
      "prompt_cache_miss_tokens": 789
    },
    "model_name": "deepseek-v4-flash",
    "system_fingerprint": "a26a7955944dc5c60445bff77fac9c8e",
    "id": "22309ca4-7777-4ed2-946f-7dfb5aeb3a65",
    "service_tier": null,
    "finish_reason": "tool_calls",
    "logprobs": null
  },
  "type": "ai",
  "name": null,
  "id": "run--01a06d28-843c-7e72-885c-860dfecf854e-0",
  "example": false,
  "tool_calls": [
    {
      "name": "get_metrics_snapshot",
      "args": {
        "service_id": "demo-high-ttft"
      },
      "id": "call_00_uiJFIbDje1lQVw4TqtvT4797",
      "type": "tool_call"
    },
    {
      "name": "search_logs",
      "args": {
        "service_id": "demo-high-ttft",
        "keyword": "scheduler",
        "limit": 20
      },
      "id": "call_01_wHxiM7j9elc3ASyjzqpL2258",
      "type": "tool_call"
    },
    {
      "name": "search_logs",
      "args": {
        "service_id": "demo-high-ttft",
        "keyword": "waiting",
        "limit": 20
      },
      "id": "call_02_2rdDEDqojIzdGNASeQcj9179",
      "type": "tool_call"
    }
  ],
  "invalid_tool_calls": [],
  "usage_metadata": {
    "input_tokens": 1301,
    "output_tokens": 444,
    "total_tokens": 1745,
    "input_token_details": {
      "cache_read": 512
    },
    "output_token_details": {
      "reasoning": 240
    }
  }
}
```

### MCP 返回：get_metrics_snapshot

```json
{
  "name": "get_metrics_snapshot",
  "arguments": {
    "service_id": "demo-high-ttft"
  },
  "result": {
    "service_id": "demo-high-ttft",
    "gpu_memory_utilization": 0.78,
    "waiting_requests": 64,
    "ttft_p95_ms": 1850,
    "prefix_cache_hit_rate": 0.42,
    "request_success_rate": 0.995,
    "kv_cache_free_blocks": 320
  },
  "tool_message": {
    "role": "tool",
    "name": "get_metrics_snapshot",
    "content": {
      "service_id": "demo-high-ttft",
      "gpu_memory_utilization": 0.78,
      "waiting_requests": 64,
      "ttft_p95_ms": 1850,
      "prefix_cache_hit_rate": 0.42,
      "request_success_rate": 0.995,
      "kv_cache_free_blocks": 320
    }
  }
}
```

### MCP 返回：search_logs

```json
{
  "name": "search_logs",
  "arguments": {
    "service_id": "demo-high-ttft",
    "keyword": "scheduler",
    "limit": 20
  },
  "result": [
    "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
  ],
  "tool_message": {
    "role": "tool",
    "name": "search_logs",
    "content": [
      "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
    ]
  }
}
```

### MCP 返回：search_logs

```json
{
  "name": "search_logs",
  "arguments": {
    "service_id": "demo-high-ttft",
    "keyword": "waiting",
    "limit": 20
  },
  "result": [
    "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
  ],
  "tool_message": {
    "role": "tool",
    "name": "search_logs",
    "content": [
      "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
    ]
  }
}
```

### 第 2 轮请求：tool_selection

#### system

```text
你是 LLM serving SRE 诊断助手。
你只能使用已经提供的只读工具，不得修改线上配置。
先结合排障手册判断还缺少什么证据；证据不足时调用 get_metrics_snapshot 或 search_logs。
工具参数中的 service_id 必须使用当前请求的 service_id。证据足够后停止调用工具。
最终报告必须只基于请求、排障手册和工具返回的实际证据；证据不足时选择 unknown，不能把推测写成事实。

```

#### human

```json
{
  "request": {
    "service_id": "demo-high-ttft",
    "symptom": "vLLM 请求排队增长，TTFT 和 P99 延迟升高"
  },
  "runbook": [
    {
      "chunk_id": "vllm_queue_batching.md::115::故障现象",
      "text": "vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT，常见于 max_num_batched_tokens 过小或 decode token 被长 prompt 挤压。",
      "source": "vllm_queue_batching.md",
      "title": "vLLM 请求排队与批处理",
      "section": "故障现象",
      "metadata": {
        "source": "vllm_queue_batching.md",
        "title": "vLLM 请求排队与批处理",
        "section": "故障现象"
      },
      "bm25_score": 8.975357099415039,
      "bm25_rank": 1,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.714513897895813,
      "dense_rank": 2,
      "rrf_rank": 2,
      "reranker_score": 0.8923693895339966,
      "reranker_rank": 1
    },
    {
      "source": "scheduler_queue.md",
      "section": "常见信号",
      "title": "Scheduler queue backlog",
      "chunk_id": "scheduler_queue.md::90::常见信号",
      "text": "- waiting requests 持续增长，TTFT P95 同步变差。\n- 日志出现 scheduler queue timeout 或 batch waiting。",
      "dense_score": 0.6747003197669983,
      "dense_rank": 6,
      "metadata": {
        "source": "scheduler_queue.md",
        "section": "常见信号",
        "title": "Scheduler queue backlog"
      },
      "rrf_score": 0.015151515151515152,
      "bm25_score": null,
      "bm25_rank": null,
      "rrf_rank": 14,
      "reranker_score": 0.8472697138786316,
      "reranker_rank": 2
    },
    {
      "chunk_id": "prefill_decode_latency.md::70::故障现象",
      "text": "TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用、KV cache 或通信瓶颈。",
      "source": "prefill_decode_latency.md",
      "title": "Prefill 与 Decode 延迟",
      "section": "故障现象",
      "metadata": {
        "source": "prefill_decode_latency.md",
        "title": "Prefill 与 Decode 延迟",
        "section": "故障现象"
      },
      "bm25_score": 8.67292230641757,
      "bm25_rank": 2,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.7302896976470947,
      "dense_rank": 1,
      "rrf_rank": 1,
      "reranker_score": 0.8323298692703247,
      "reranker_rank": 3
    }
  ],
  "tool_results": [
    {
      "name": "get_metrics_snapshot",
      "arguments": {
        "service_id": "demo-high-ttft"
      },
      "result": {
        "service_id": "demo-high-ttft",
        "gpu_memory_utilization": 0.78,
        "waiting_requests": 64,
        "ttft_p95_ms": 1850,
        "prefix_cache_hit_rate": 0.42,
        "request_success_rate": 0.995,
        "kv_cache_free_blocks": 320
      }
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "scheduler",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
      ]
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "waiting",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
      ]
    }
  ],
  "instruction": "选择必要的只读工具。已有足够证据时不要再调用工具。"
}
```

### 第 2 轮：原始模型消息（含工具调用）

```json
{
  "content": "I'll first check what additional evidence distinguishes the candidate causes in the runbook (prefill queueing vs. decode contention). I already have TTFT (1850ms) and queue growth, but no TPOT/decode signal, so I'll search logs for decode/TPOT and throughput-related messages.",
  "additional_kwargs": {
    "tool_calls": [
      {
        "id": "call_00_4rKsj3o6S4rFm8jQvZZf9317",
        "function": {
          "arguments": "{\"service_id\": \"demo-high-ttft\", \"keyword\": \"TPOT\", \"limit\": 20}",
          "name": "search_logs"
        },
        "type": "function",
        "index": 0
      },
      {
        "id": "call_01_BwivjGpnV5MgSZxaPzu67312",
        "function": {
          "arguments": "{\"service_id\": \"demo-high-ttft\", \"keyword\": \"decode\", \"limit\": 20}",
          "name": "search_logs"
        },
        "type": "function",
        "index": 1
      },
      {
        "id": "call_02_RdvV8ijKxPKZIcZPxtGX1065",
        "function": {
          "arguments": "{\"service_id\": \"demo-high-ttft\", \"keyword\": \"throughput\", \"limit\": 20}",
          "name": "search_logs"
        },
        "type": "function",
        "index": 2
      }
    ],
    "refusal": null
  },
  "response_metadata": {
    "token_usage": {
      "completion_tokens": 2032,
      "prompt_tokens": 1534,
      "total_tokens": 3566,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": null,
        "reasoning_tokens": 1751,
        "rejected_prediction_tokens": null
      },
      "prompt_tokens_details": {
        "audio_tokens": null,
        "cache_write_tokens": null,
        "cached_tokens": 512
      },
      "prompt_cache_hit_tokens": 512,
      "prompt_cache_miss_tokens": 1022
    },
    "model_name": "deepseek-v4-flash",
    "system_fingerprint": "a26a7955944dc5c60445bff77fac9c8e",
    "id": "d5a6287f-4196-4fc4-94ef-4a9c36580bf1",
    "service_tier": null,
    "finish_reason": "tool_calls",
    "logprobs": null
  },
  "type": "ai",
  "name": null,
  "id": "run--01a06d28-9cfc-7cb0-9aed-969f505d6844-0",
  "example": false,
  "tool_calls": [
    {
      "name": "search_logs",
      "args": {
        "service_id": "demo-high-ttft",
        "keyword": "TPOT",
        "limit": 20
      },
      "id": "call_00_4rKsj3o6S4rFm8jQvZZf9317",
      "type": "tool_call"
    },
    {
      "name": "search_logs",
      "args": {
        "service_id": "demo-high-ttft",
        "keyword": "decode",
        "limit": 20
      },
      "id": "call_01_BwivjGpnV5MgSZxaPzu67312",
      "type": "tool_call"
    },
    {
      "name": "search_logs",
      "args": {
        "service_id": "demo-high-ttft",
        "keyword": "throughput",
        "limit": 20
      },
      "id": "call_02_RdvV8ijKxPKZIcZPxtGX1065",
      "type": "tool_call"
    }
  ],
  "invalid_tool_calls": [],
  "usage_metadata": {
    "input_tokens": 1534,
    "output_tokens": 2032,
    "total_tokens": 3566,
    "input_token_details": {
      "cache_read": 512
    },
    "output_token_details": {
      "reasoning": 1751
    }
  }
}
```

### MCP 返回：search_logs

```json
{
  "name": "search_logs",
  "arguments": {
    "service_id": "demo-high-ttft",
    "keyword": "TPOT",
    "limit": 20
  },
  "result": [],
  "tool_message": {
    "role": "tool",
    "name": "search_logs",
    "content": []
  }
}
```

### MCP 返回：search_logs

```json
{
  "name": "search_logs",
  "arguments": {
    "service_id": "demo-high-ttft",
    "keyword": "decode",
    "limit": 20
  },
  "result": [
    "2026-08-27T09:10:04Z INFO decode workers healthy"
  ],
  "tool_message": {
    "role": "tool",
    "name": "search_logs",
    "content": [
      "2026-08-27T09:10:04Z INFO decode workers healthy"
    ]
  }
}
```

### MCP 返回：search_logs

```json
{
  "name": "search_logs",
  "arguments": {
    "service_id": "demo-high-ttft",
    "keyword": "throughput",
    "limit": 20
  },
  "result": [],
  "tool_message": {
    "role": "tool",
    "name": "search_logs",
    "content": []
  }
}
```

### 第 3 轮请求：tool_selection

#### system

```text
你是 LLM serving SRE 诊断助手。
你只能使用已经提供的只读工具，不得修改线上配置。
先结合排障手册判断还缺少什么证据；证据不足时调用 get_metrics_snapshot 或 search_logs。
工具参数中的 service_id 必须使用当前请求的 service_id。证据足够后停止调用工具。
最终报告必须只基于请求、排障手册和工具返回的实际证据；证据不足时选择 unknown，不能把推测写成事实。

```

#### human

```json
{
  "request": {
    "service_id": "demo-high-ttft",
    "symptom": "vLLM 请求排队增长，TTFT 和 P99 延迟升高"
  },
  "runbook": [
    {
      "chunk_id": "vllm_queue_batching.md::115::故障现象",
      "text": "vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT，常见于 max_num_batched_tokens 过小或 decode token 被长 prompt 挤压。",
      "source": "vllm_queue_batching.md",
      "title": "vLLM 请求排队与批处理",
      "section": "故障现象",
      "metadata": {
        "source": "vllm_queue_batching.md",
        "title": "vLLM 请求排队与批处理",
        "section": "故障现象"
      },
      "bm25_score": 8.975357099415039,
      "bm25_rank": 1,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.714513897895813,
      "dense_rank": 2,
      "rrf_rank": 2,
      "reranker_score": 0.8923693895339966,
      "reranker_rank": 1
    },
    {
      "source": "scheduler_queue.md",
      "section": "常见信号",
      "title": "Scheduler queue backlog",
      "chunk_id": "scheduler_queue.md::90::常见信号",
      "text": "- waiting requests 持续增长，TTFT P95 同步变差。\n- 日志出现 scheduler queue timeout 或 batch waiting。",
      "dense_score": 0.6747003197669983,
      "dense_rank": 6,
      "metadata": {
        "source": "scheduler_queue.md",
        "section": "常见信号",
        "title": "Scheduler queue backlog"
      },
      "rrf_score": 0.015151515151515152,
      "bm25_score": null,
      "bm25_rank": null,
      "rrf_rank": 14,
      "reranker_score": 0.8472697138786316,
      "reranker_rank": 2
    },
    {
      "chunk_id": "prefill_decode_latency.md::70::故障现象",
      "text": "TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用、KV cache 或通信瓶颈。",
      "source": "prefill_decode_latency.md",
      "title": "Prefill 与 Decode 延迟",
      "section": "故障现象",
      "metadata": {
        "source": "prefill_decode_latency.md",
        "title": "Prefill 与 Decode 延迟",
        "section": "故障现象"
      },
      "bm25_score": 8.67292230641757,
      "bm25_rank": 2,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.7302896976470947,
      "dense_rank": 1,
      "rrf_rank": 1,
      "reranker_score": 0.8323298692703247,
      "reranker_rank": 3
    }
  ],
  "tool_results": [
    {
      "name": "get_metrics_snapshot",
      "arguments": {
        "service_id": "demo-high-ttft"
      },
      "result": {
        "service_id": "demo-high-ttft",
        "gpu_memory_utilization": 0.78,
        "waiting_requests": 64,
        "ttft_p95_ms": 1850,
        "prefix_cache_hit_rate": 0.42,
        "request_success_rate": 0.995,
        "kv_cache_free_blocks": 320
      }
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "scheduler",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
      ]
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "waiting",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
      ]
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "TPOT",
        "limit": 20
      },
      "result": []
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "decode",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:04Z INFO decode workers healthy"
      ]
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "throughput",
        "limit": 20
      },
      "result": []
    }
  ],
  "instruction": "选择必要的只读工具。已有足够证据时不要再调用工具。"
}
```

### 第 3 轮：原始模型消息（含工具调用）

```json
{
  "content": "## 诊断报告：demo-high-ttft\n\n### 一、证据汇总\n基于只读证据（metrics snapshot 与日志检索）：\n\n| 指标/日志 | 值 | 含义 |\n|---|---|---|\n| waiting_requests | 64（日志 WARN 同步确认） | 存在 scheduler queue backlog，且有增长迹象 |\n| ttft_p95_ms | 1850 ms | TTFT P95 明显升高 |\n| gpu_memory_utilization | 0.78 | 尚未逼近显存上限 |\n| kv_cache_free_blocks | 320 | KV cache 未耗尽，仍有空闲块 |\n| prefix_cache_hit_rate | 0.42 | 前缀命中中等 |\n| request_success_rate | 0.995 | 成功率正常，无明显报错/超时丢弃 |\n| 日志 scheduler / waiting | WARN scheduler queue waiting_requests=64 | 确认队列堆积 |\n| 日志 decode | INFO decode workers healthy | decode worker 正常 |\n| 日志 TPOT / throughput | 无结果 | 缺少这两个维度的观测 |\n\n### 二、结论（基于证据）\n\n**高置信度判定：vLLM 处于 scheduler queue backlog 状态，表现为请求长期 waiting、队列堆积，TTFT P95（1850ms）显著劣化。** 这与排障手册《scheduler_queue.md》\"waiting requests 持续增长、TTFT P95 同步变差\" 以及《vllm_queue_batching.md》\"请求持续 waiting、queue length 增长、短请求也出现高 TTFT\" 的现象完全吻合。\n\n**可以排除的因素（证据充分）：**\n- KV cache / 显存不足：`gpu_memory_utilization=0.78`、`kv_cache_free_blocks=320`，资源未打满，不是显存/缓存瓶颈。\n- 服务整体故障或大面积失败：`request_success_rate=0.995`，无超时丢请求。\n- decode 侧异常：`decode workers healthy`，且未检索到 decode 相关告警。\n\n**无法确认、标记为 unknown 的根因细节（证据不足）：**\n- TTFT 升高的具体阶段归因：排障手册《prefill_decode_latency.md》要求用 \"TTFT 高而 TPOT 正常 → prefill 排队/长 prompt\" vs \"TPOT 高而 TTFT 正常 → decode 争用\" 来区分，但当前**没有 TPOT 数据**（日志检索为空），无法验证 TPOT 是否正常，因此不能确认是否仅 prefill 阶段受影响。\n- 具体触发源（`max_num_batched_tokens` 过小 / decode token 被长 prompt 挤压 / 并发超载）：手册指出这类问题可通过\"吞吐未随并发提升\"及 token 统计判断，但当前**缺少 throughput 指标与 prompt/num_batched_tokens 观测**，日志检索 throughput 为空，无法归因到上述任一配置或负载原因。\n\n### 三、建议\n1. 结论方向：问题属于 **prefill/调度队列侧瓶颈导致 TTFT 劣化**，而非 decode、KV cache 或显存瓶颈（这些可基于现有证据排除）。\n2. 补齐证据后再定根因：采集 TPOT（确认是否正常以区分 prefill/decode）、吞吐-并发曲线（验证\"吞吐是否随并发提升\"）、以及 num_batched_tokens / 长 prompt 占比统计，才能确认是 `max_num_batched_tokens` 过小、长 prompt 挤压还是并发超载。\n\n> 说明：上述报告仅基于请求、排障手册与工具返回的实际证据；TPOT、吞吐等缺失项均按 unknown 处理，未做推测性归因。",
  "additional_kwargs": {
    "refusal": null
  },
  "response_metadata": {
    "token_usage": {
      "completion_tokens": 3654,
      "prompt_tokens": 1671,
      "total_tokens": 5325,
      "completion_tokens_details": {
        "accepted_prediction_tokens": null,
        "audio_tokens": null,
        "reasoning_tokens": 2849,
        "rejected_prediction_tokens": null
      },
      "prompt_tokens_details": {
        "audio_tokens": null,
        "cache_write_tokens": null,
        "cached_tokens": 1408
      },
      "prompt_cache_hit_tokens": 1408,
      "prompt_cache_miss_tokens": 263
    },
    "model_name": "deepseek-v4-flash",
    "system_fingerprint": "a26a7955944dc5c60445bff77fac9c8e",
    "id": "95e3b17d-dede-4c51-b9eb-71f291df423b",
    "service_tier": null,
    "finish_reason": "stop",
    "logprobs": null
  },
  "type": "ai",
  "name": null,
  "id": "run--01a06d28-e61b-7cc3-b15c-2ce74544b800-0",
  "example": false,
  "tool_calls": [],
  "invalid_tool_calls": [],
  "usage_metadata": {
    "input_tokens": 1671,
    "output_tokens": 3654,
    "total_tokens": 5325,
    "input_token_details": {
      "cache_read": 1408
    },
    "output_token_details": {
      "reasoning": 2849
    }
  }
}
```

### 第 4 轮请求：final_user_response

#### system

```text
你是 LLM serving SRE 诊断助手。
你只能使用已经提供的只读工具，不得修改线上配置。
先结合排障手册判断还缺少什么证据；证据不足时调用 get_metrics_snapshot 或 search_logs。
工具参数中的 service_id 必须使用当前请求的 service_id。证据足够后停止调用工具。
最终报告必须只基于请求、排障手册和工具返回的实际证据；证据不足时选择 unknown，不能把推测写成事实。

```

#### human

```json
{
  "request": {
    "service_id": "demo-high-ttft",
    "symptom": "vLLM 请求排队增长，TTFT 和 P99 延迟升高"
  },
  "runbook": [
    {
      "chunk_id": "vllm_queue_batching.md::115::故障现象",
      "text": "vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT，常见于 max_num_batched_tokens 过小或 decode token 被长 prompt 挤压。",
      "source": "vllm_queue_batching.md",
      "title": "vLLM 请求排队与批处理",
      "section": "故障现象",
      "metadata": {
        "source": "vllm_queue_batching.md",
        "title": "vLLM 请求排队与批处理",
        "section": "故障现象"
      },
      "bm25_score": 8.975357099415039,
      "bm25_rank": 1,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.714513897895813,
      "dense_rank": 2,
      "rrf_rank": 2,
      "reranker_score": 0.8923693895339966,
      "reranker_rank": 1
    },
    {
      "source": "scheduler_queue.md",
      "section": "常见信号",
      "title": "Scheduler queue backlog",
      "chunk_id": "scheduler_queue.md::90::常见信号",
      "text": "- waiting requests 持续增长，TTFT P95 同步变差。\n- 日志出现 scheduler queue timeout 或 batch waiting。",
      "dense_score": 0.6747003197669983,
      "dense_rank": 6,
      "metadata": {
        "source": "scheduler_queue.md",
        "section": "常见信号",
        "title": "Scheduler queue backlog"
      },
      "rrf_score": 0.015151515151515152,
      "bm25_score": null,
      "bm25_rank": null,
      "rrf_rank": 14,
      "reranker_score": 0.8472697138786316,
      "reranker_rank": 2
    },
    {
      "chunk_id": "prefill_decode_latency.md::70::故障现象",
      "text": "TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用、KV cache 或通信瓶颈。",
      "source": "prefill_decode_latency.md",
      "title": "Prefill 与 Decode 延迟",
      "section": "故障现象",
      "metadata": {
        "source": "prefill_decode_latency.md",
        "title": "Prefill 与 Decode 延迟",
        "section": "故障现象"
      },
      "bm25_score": 8.67292230641757,
      "bm25_rank": 2,
      "rrf_score": 0.03252247488101534,
      "dense_score": 0.7302896976470947,
      "dense_rank": 1,
      "rrf_rank": 1,
      "reranker_score": 0.8323298692703247,
      "reranker_rank": 3
    }
  ],
  "tool_results": [
    {
      "name": "get_metrics_snapshot",
      "arguments": {
        "service_id": "demo-high-ttft"
      },
      "result": {
        "service_id": "demo-high-ttft",
        "gpu_memory_utilization": 0.78,
        "waiting_requests": 64,
        "ttft_p95_ms": 1850,
        "prefix_cache_hit_rate": 0.42,
        "request_success_rate": 0.995,
        "kv_cache_free_blocks": 320
      }
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "scheduler",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
      ]
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "waiting",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:01Z WARN scheduler queue waiting_requests=64"
      ]
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "TPOT",
        "limit": 20
      },
      "result": []
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "decode",
        "limit": 20
      },
      "result": [
        "2026-08-27T09:10:04Z INFO decode workers healthy"
      ]
    },
    {
      "name": "search_logs",
      "arguments": {
        "service_id": "demo-high-ttft",
        "keyword": "throughput",
        "limit": 20
      },
      "result": []
    }
  ],
  "instruction": "严格按照下面的 IncidentReport JSON Schema 输出一个对象，只允许 schema 中的字段，不要输出 service_id、symptom、status、supporting_evidence 或其他额外字段。incident_type 只能是 cuda_oom、high_ttft、low_prefix_cache_hit、unknown；severity 只能是 low、medium、high。evidence 必须引用实际工具或文档来源，不能编造指标、日志或操作结果。\nJSON Schema：{\"$defs\": {\"Evidence\": {\"properties\": {\"source\": {\"maxLength\": 200, \"minLength\": 1, \"title\": \"Source\", \"type\": \"string\"}, \"detail\": {\"maxLength\": 1000, \"minLength\": 1, \"title\": \"Detail\", \"type\": \"string\"}}, \"required\": [\"source\", \"detail\"], \"title\": \"Evidence\", \"type\": \"object\"}}, \"additionalProperties\": false, \"properties\": {\"incident_type\": {\"enum\": [\"cuda_oom\", \"high_ttft\", \"low_prefix_cache_hit\", \"unknown\"], \"title\": \"Incident Type\", \"type\": \"string\"}, \"severity\": {\"enum\": [\"low\", \"medium\", \"high\"], \"title\": \"Severity\", \"type\": \"string\"}, \"root_cause\": {\"maxLength\": 1000, \"minLength\": 1, \"title\": \"Root Cause\", \"type\": \"string\"}, \"evidence\": {\"items\": {\"$ref\": \"#/$defs/Evidence\"}, \"maxItems\": 12, \"minItems\": 1, \"title\": \"Evidence\", \"type\": \"array\"}, \"recommended_actions\": {\"items\": {\"type\": \"string\"}, \"maxItems\": 8, \"minItems\": 1, \"title\": \"Recommended Actions\", \"type\": \"array\"}, \"tools_used\": {\"items\": {\"type\": \"string\"}, \"maxItems\": 8, \"title\": \"Tools Used\", \"type\": \"array\"}, \"confidence\": {\"maximum\": 1, \"minimum\": 0, \"title\": \"Confidence\", \"type\": \"number\"}}, \"required\": [\"incident_type\", \"severity\", \"root_cause\", \"evidence\", \"recommended_actions\", \"confidence\"], \"title\": \"IncidentReport\", \"type\": \"object\"}\n模型草稿：## 诊断报告：demo-high-ttft\n\n### 一、证据汇总\n基于只读证据（metrics snapshot 与日志检索）：\n\n| 指标/日志 | 值 | 含义 |\n|---|---|---|\n| waiting_requests | 64（日志 WARN 同步确认） | 存在 scheduler queue backlog，且有增长迹象 |\n| ttft_p95_ms | 1850 ms | TTFT P95 明显升高 |\n| gpu_memory_utilization | 0.78 | 尚未逼近显存上限 |\n| kv_cache_free_blocks | 320 | KV cache 未耗尽，仍有空闲块 |\n| prefix_cache_hit_rate | 0.42 | 前缀命中中等 |\n| request_success_rate | 0.995 | 成功率正常，无明显报错/超时丢弃 |\n| 日志 scheduler / waiting | WARN scheduler queue waiting_requests=64 | 确认队列堆积 |\n| 日志 decode | INFO decode workers healthy | decode worker 正常 |\n| 日志 TPOT / throughput | 无结果 | 缺少这两个维度的观测 |\n\n### 二、结论（基于证据）\n\n**高置信度判定：vLLM 处于 scheduler queue backlog 状态，表现为请求长期 waiting、队列堆积，TTFT P95（1850ms）显著劣化。** 这与排障手册《scheduler_queue.md》\"waiting requests 持续增长、TTFT P95 同步变差\" 以及《vllm_queue_batching.md》\"请求持续 waiting、queue length 增长、短请求也出现高 TTFT\" 的现象完全吻合。\n\n**可以排除的因素（证据充分）：**\n- KV cache / 显存不足：`gpu_memory_utilization=0.78`、`kv_cache_free_blocks=320`，资源未打满，不是显存/缓存瓶颈。\n- 服务整体故障或大面积失败：`request_success_rate=0.995`，无超时丢请求。\n- decode 侧异常：`decode workers healthy`，且未检索到 decode 相关告警。\n\n**无法确认、标记为 unknown 的根因细节（证据不足）：**\n- TTFT 升高的具体阶段归因：排障手册《prefill_decode_latency.md》要求用 \"TTFT 高而 TPOT 正常 → prefill 排队/长 prompt\" vs \"TPOT 高而 TTFT 正常 → decode 争用\" 来区分，但当前**没有 TPOT 数据**（日志检索为空），无法验证 TPOT 是否正常，因此不能确认是否仅 prefill 阶段受影响。\n- 具体触发源（`max_num_batched_tokens` 过小 / decode token 被长 prompt 挤压 / 并发超载）：手册指出这类问题可通过\"吞吐未随并发提升\"及 token 统计判断，但当前**缺少 throughput 指标与 prompt/num_batched_tokens 观测**，日志检索 throughput 为空，无法归因到上述任一配置或负载原因。\n\n### 三、建议\n1. 结论方向：问题属于 **prefill/调度队列侧瓶颈导致 TTFT 劣化**，而非 decode、KV cache 或显存瓶颈（这些可基于现有证据排除）。\n2. 补齐证据后再定根因：采集 TPOT（确认是否正常以区分 prefill/decode）、吞吐-并发曲线（验证\"吞吐是否随并发提升\"）、以及 num_batched_tokens / 长 prompt 占比统计，才能确认是 `max_num_batched_tokens` 过小、长 prompt 挤压还是并发超载。\n\n> 说明：上述报告仅基于请求、排障手册与工具返回的实际证据；TPOT、吞吐等缺失项均按 unknown 处理，未做推测性归因。"
}
```

### 第 4 轮：结构化解析结果

```json
{
  "incident_type": "high_ttft",
  "severity": "medium",
  "root_cause": "vLLM scheduler queue 存在持续 backlog（waiting_requests=64），导致请求等待 prefill slot 时间增加，TTFT P95 升至 1850ms。GPU 内存利用率 0.78、KV cache 空闲块 320、decode workers healthy，排除显存/KV cache 或 decode 侧瓶颈；但具体触发因素（如 max_num_batched_tokens 过小、长 prompt 挤压、并发超载）因缺少 TPOT 与 throughput 数据，尚无法确认。",
  "evidence": [
    {
      "source": "get_metrics_snapshot",
      "detail": "waiting_requests=64, ttft_p95_ms=1850"
    },
    {
      "source": "get_metrics_snapshot",
      "detail": "gpu_memory_utilization=0.78, kv_cache_free_blocks=320, request_success_rate=0.995"
    },
    {
      "source": "search_logs",
      "detail": "WARN scheduler queue waiting_requests=64"
    },
    {
      "source": "search_logs",
      "detail": "INFO decode workers healthy"
    },
    {
      "source": "runbook scheduler_queue.md",
      "detail": "waiting requests 持续增长，TTFT P95 同步变差；日志出现 scheduler queue timeout 或 batch waiting"
    },
    {
      "source": "runbook vllm_queue_batching.md",
      "detail": "vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT"
    },
    {
      "source": "runbook prefill_decode_latency.md",
      "detail": "TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用"
    }
  ],
  "recommended_actions": [
    "采集 TPOT 指标，确认 TPOT 是否正常，以区分 prefill 排队/长 prompt 与 decode 侧问题。",
    "监控 scheduler queue length 与 TTFT 变化趋势，确认队列是否持续增长。",
    "检查 vLLM 批处理参数（如 max_num_batched_tokens、max_num_seqs）是否过小，必要时调优以提升 prefill 吞吐。",
    "分析是否存在超长 prompt 或高并发请求挤压短请求的 prefill，考虑拆分 prompt 或优化调度策略。",
    "根据实际负载评估是否需要扩容或增加 vLLM 实例数。"
  ],
  "tools_used": [
    "get_metrics_snapshot",
    "search_logs"
  ],
  "confidence": 0.78
}
```

## 最终用户报告

```json
{
  "incident_type": "high_ttft",
  "severity": "medium",
  "root_cause": "vLLM scheduler queue 存在持续 backlog（waiting_requests=64），导致请求等待 prefill slot 时间增加，TTFT P95 升至 1850ms。GPU 内存利用率 0.78、KV cache 空闲块 320、decode workers healthy，排除显存/KV cache 或 decode 侧瓶颈；但具体触发因素（如 max_num_batched_tokens 过小、长 prompt 挤压、并发超载）因缺少 TPOT 与 throughput 数据，尚无法确认。",
  "evidence": [
    {
      "source": "get_metrics_snapshot",
      "detail": "waiting_requests=64, ttft_p95_ms=1850"
    },
    {
      "source": "get_metrics_snapshot",
      "detail": "gpu_memory_utilization=0.78, kv_cache_free_blocks=320, request_success_rate=0.995"
    },
    {
      "source": "search_logs",
      "detail": "WARN scheduler queue waiting_requests=64"
    },
    {
      "source": "search_logs",
      "detail": "INFO decode workers healthy"
    },
    {
      "source": "runbook scheduler_queue.md",
      "detail": "waiting requests 持续增长，TTFT P95 同步变差；日志出现 scheduler queue timeout 或 batch waiting"
    },
    {
      "source": "runbook vllm_queue_batching.md",
      "detail": "vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT"
    },
    {
      "source": "runbook prefill_decode_latency.md",
      "detail": "TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用"
    }
  ],
  "recommended_actions": [
    "采集 TPOT 指标，确认 TPOT 是否正常，以区分 prefill 排队/长 prompt 与 decode 侧问题。",
    "监控 scheduler queue length 与 TTFT 变化趋势，确认队列是否持续增长。",
    "检查 vLLM 批处理参数（如 max_num_batched_tokens、max_num_seqs）是否过小，必要时调优以提升 prefill 吞吐。",
    "分析是否存在超长 prompt 或高并发请求挤压短请求的 prefill，考虑拆分 prompt 或优化调度策略。",
    "根据实际负载评估是否需要扩容或增加 vLLM 实例数。"
  ],
  "tools_used": [
    "get_metrics_snapshot",
    "search_logs",
    "search_logs",
    "search_logs",
    "search_logs",
    "search_logs"
  ],
  "confidence": 0.78
}
```
