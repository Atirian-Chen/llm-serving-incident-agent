# RAG 检索评测

- 模式：`real`
- 知识库 chunk：119
- Query：24
- Embedding：`BAAI/bge-small-zh-v1.5`；Reranker：`BAAI/bge-reranker-v2-m3`；Device：`cuda`

| 方法 | Recall@1 | Recall@3 | Recall@10 | MRR | NDCG@3 | 平均 ms | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| keyword | 0.142 | 0.175 | 0.192 | 0.803 | 0.690 | 0.246 | 0.460 |
| bm25 | 0.133 | 0.175 | 0.200 | 0.793 | 0.655 | 0.367 | 0.607 |
| dense | 0.133 | 0.175 | 0.200 | 0.786 | 0.631 | 10.133 | 13.355 |
| rrf | 0.133 | 0.183 | 0.200 | 0.802 | 0.721 | 8.716 | 12.516 |
| hybrid_reranker | 0.108 | 0.158 | 0.200 | 0.717 | 0.629 | 180.484 | 215.168 |

## Query 明细

| Query | 方法 | 首个相关 rank | NDCG@3 | Top 来源 |
| --- | --- | ---: | ---: | --- |
| q01 CUDA out of memory KV Cache 分配失败 | keyword | 2 | 0.387 | gpu_oom_recovery.md, cuda_oom.md, ncuda_oom_fragmentation.md |
| q01 CUDA out of memory KV Cache 分配失败 | bm25 | 1 | 0.613 | cuda_oom.md, ncuda_oom_fragmentation.md, gpu_oom_recovery.md |
| q01 CUDA out of memory KV Cache 分配失败 | dense | 4 | 0.000 | ncuda_oom_fragmentation.md, vllm_paged_attention.md, kv_cache_capacity.md |
| q01 CUDA out of memory KV Cache 分配失败 | rrf | 2 | 0.387 | ncuda_oom_fragmentation.md, cuda_oom.md, vllm_paged_attention.md |
| q01 CUDA out of memory KV Cache 分配失败 | hybrid_reranker | 4 | 0.000 | ncuda_oom_fragmentation.md, kv_cache_capacity.md, vllm_paged_attention.md |
| q02 gpu_memory_utilization 接近 1 显存不足 | keyword | 3 | 0.307 | cuda_kernel_errors.md, cuda_oom.md, ncuda_oom_fragmentation.md |
| q02 gpu_memory_utilization 接近 1 显存不足 | bm25 | 2 | 0.387 | cuda_oom.md, gpu_memory.md, kv_cache_capacity.md |
| q02 gpu_memory_utilization 接近 1 显存不足 | dense | 1 | 0.613 | gpu_memory.md, cuda_oom.md, vllm_paged_attention.md |
| q02 gpu_memory_utilization 接近 1 显存不足 | rrf | 2 | 0.387 | cuda_oom.md, gpu_memory.md, kv_cache_capacity.md |
| q02 gpu_memory_utilization 接近 1 显存不足 | hybrid_reranker | 2 | 0.387 | cuda_oom.md, gpu_memory.md, gpu_utilization_anomaly.md |
| q03 vLLM max_num_seqs 增大后 OOM | keyword | 1 | 0.613 | cuda_oom.md, cuda_oom.md, gpu_oom_recovery.md |
| q03 vLLM max_num_seqs 增大后 OOM | bm25 | 1 | 0.613 | vllm_queue_batching.md, service_5xx_timeout.md, vllm_queue_batching.md |
| q03 vLLM max_num_seqs 增大后 OOM | dense | 1 | 0.920 | cuda_oom.md, high_ttft.md, vllm_queue_batching.md |
| q03 vLLM max_num_seqs 增大后 OOM | rrf | 1 | 1.000 | cuda_oom.md, vllm_queue_batching.md, high_ttft.md |
| q03 vLLM max_num_seqs 增大后 OOM | hybrid_reranker | 1 | 1.000 | cuda_oom.md, vllm_queue_batching.md, high_ttft.md |
| q04 GPU 显存碎片 reserved allocated | keyword | 1 | 1.000 | ncuda_oom_fragmentation.md, gpu_memory.md, gpu_utilization_anomaly.md |
| q04 GPU 显存碎片 reserved allocated | bm25 | 1 | 1.000 | ncuda_oom_fragmentation.md, ncuda_oom_fragmentation.md, gpu_memory.md |
| q04 GPU 显存碎片 reserved allocated | dense | 2 | 0.631 | vllm_paged_attention.md, ncuda_oom_fragmentation.md, gpu_memory.md |
| q04 GPU 显存碎片 reserved allocated | rrf | 1 | 1.000 | ncuda_oom_fragmentation.md, gpu_memory.md, vllm_paged_attention.md |
| q04 GPU 显存碎片 reserved allocated | hybrid_reranker | 1 | 1.000 | ncuda_oom_fragmentation.md, ncuda_oom_fragmentation.md, gpu_oom_recovery.md |
| q05 KV Cache 使用率过高 block table | keyword | 1 | 1.000 | kv_cache_capacity.md, ncuda_oom_kv_cache.md, vllm_paged_attention.md |
| q05 KV Cache 使用率过高 block table | bm25 | 4 | 0.000 | cuda_oom.md, ncuda_oom_fragmentation.md, ncaching_prefix_kv.md |
| q05 KV Cache 使用率过高 block table | dense | 3 | 0.307 | vllm_paged_attention.md, ncuda_oom_fragmentation.md, kv_cache_capacity.md |
| q05 KV Cache 使用率过高 block table | rrf | 4 | 0.000 | ncuda_oom_fragmentation.md, cuda_oom.md, vllm_paged_attention.md |
| q05 KV Cache 使用率过高 block table | hybrid_reranker | 2 | 0.387 | cuda_oom.md, kv_cache_capacity.md, ncaching_prefix_kv.md |
| q06 PagedAttention KV cache vLLM | keyword | 1 | 1.000 | vllm_paged_attention.md, kv_cache_capacity.md, ncuda_oom_kv_cache.md |
| q06 PagedAttention KV cache vLLM | bm25 | 1 | 1.000 | vllm_paged_attention.md, gpu_memory.md, vllm_paged_attention.md |
| q06 PagedAttention KV cache vLLM | dense | 1 | 1.000 | vllm_paged_attention.md, vllm_paged_attention.md, ncuda_oom_kv_cache.md |
| q06 PagedAttention KV cache vLLM | rrf | 1 | 1.000 | vllm_paged_attention.md, vllm_paged_attention.md, cuda_oom.md |
| q06 PagedAttention KV cache vLLM | hybrid_reranker | 1 | 1.000 | vllm_paged_attention.md, gpu_memory.md, gpu_memory.md |
| q07 TTFT P95 首 token 延迟升高 | keyword | 1 | 0.920 | ttft_tpot_p99.md, vllm_queue_batching.md, high_ttft.md |
| q07 TTFT P95 首 token 延迟升高 | bm25 | 2 | 0.693 | gpu_utilization_anomaly.md, high_ttft.md, ttft_tpot_p99.md |
| q07 TTFT P95 首 token 延迟升高 | dense | 1 | 0.613 | ttft_tpot_p99.md, prefill_decode_latency.md, vllm_queue_batching.md |
| q07 TTFT P95 首 token 延迟升高 | rrf | 1 | 0.613 | ttft_tpot_p99.md, prefill_decode_latency.md, vllm_queue_batching.md |
| q07 TTFT P95 首 token 延迟升高 | hybrid_reranker | 4 | 0.000 | gpu_utilization_anomaly.md, vllm_queue_batching.md, continuous_batching.md |
| q08 TPOT decode 变慢 P99 长尾 | keyword | 1 | 1.000 | ttft_tpot_p99.md, prefill_decode_latency.md, prefill_decode_latency.md |
| q08 TPOT decode 变慢 P99 长尾 | bm25 | 1 | 0.920 | ttft_tpot_p99.md, continuous_batching.md, prefill_decode_latency.md |
| q08 TPOT decode 变慢 P99 长尾 | dense | 1 | 0.613 | ttft_tpot_p99.md, scheduler_queue.md, scheduler_queue.md |
| q08 TPOT decode 变慢 P99 长尾 | rrf | 1 | 1.000 | ttft_tpot_p99.md, prefill_decode_latency.md, scheduler_queue.md |
| q08 TPOT decode 变慢 P99 长尾 | hybrid_reranker | 1 | 0.920 | ttft_tpot_p99.md, service_5xx_timeout.md, prefill_decode_latency.md |
| q09 vLLM waiting requests scheduler queue 排队 | keyword | 1 | 1.000 | scheduler_queue.md, vllm_queue_batching.md, vllm_queue_batching.md |
| q09 vLLM waiting requests scheduler queue 排队 | bm25 | 1 | 1.000 | scheduler_queue.md, vllm_queue_batching.md, vllm_queue_batching.md |
| q09 vLLM waiting requests scheduler queue 排队 | dense | 1 | 1.000 | vllm_queue_batching.md, scheduler_queue.md, vllm_queue_batching.md |
| q09 vLLM waiting requests scheduler queue 排队 | rrf | 1 | 1.000 | scheduler_queue.md, vllm_queue_batching.md, vllm_queue_batching.md |
| q09 vLLM waiting requests scheduler queue 排队 | hybrid_reranker | 1 | 0.920 | vllm_queue_batching.md, vllm_queue_batching.md, scheduler_queue.md |
| q10 continuous batching max_num_batched_tokens | keyword | 2 | 0.631 | prefill_decode_latency.md, continuous_batching.md, vllm_queue_batching.md |
| q10 continuous batching max_num_batched_tokens | bm25 | 2 | 0.631 | prefill_decode_latency.md, continuous_batching.md, high_ttft.md |
| q10 continuous batching max_num_batched_tokens | dense | 2 | 0.631 | prefill_decode_latency.md, continuous_batching.md, vllm_queue_batching.md |
| q10 continuous batching max_num_batched_tokens | rrf | 3 | 0.500 | prefill_decode_latency.md, vllm_queue_batching.md, continuous_batching.md |
| q10 continuous batching max_num_batched_tokens | hybrid_reranker | 4 | 0.000 | prefill_decode_latency.md, vllm_queue_batching.md, vllm_queue_batching.md |
| q11 SGLang runtime 服务 5xx | keyword | 1 | 1.000 | sglang_runtime.md, service_5xx_timeout.md, service_5xx_timeout.md |
| q11 SGLang runtime 服务 5xx | bm25 | 1 | 0.613 | sglang_runtime.md, health_check_readiness.md, multi_instance_load_balance.md |
| q11 SGLang runtime 服务 5xx | dense | 1 | 0.613 | sglang_runtime.md, sglang_runtime.md, high_ttft.md |
| q11 SGLang runtime 服务 5xx | rrf | 1 | 0.920 | sglang_runtime.md, sglang_runtime.md, service_5xx_timeout.md |
| q11 SGLang runtime 服务 5xx | hybrid_reranker | 1 | 0.613 | sglang_runtime.md, health_check_readiness.md, multi_instance_load_balance.md |
| q12 SGLang tokenizer backend crash | keyword | 1 | 1.000 | sglang_runtime.md, model_loading_tokenizer.md, model_loading_tokenizer.md |
| q12 SGLang tokenizer backend crash | bm25 | 1 | 0.613 | sglang_runtime.md, quantization_dtype.md, throughput_regression.md |
| q12 SGLang tokenizer backend crash | dense | 1 | 0.613 | sglang_runtime.md, vllm_queue_batching.md, gpu_utilization_anomaly.md |
| q12 SGLang tokenizer backend crash | rrf | 1 | 0.613 | sglang_runtime.md, prefix_cache.md, throughput_regression.md |
| q12 SGLang tokenizer backend crash | hybrid_reranker | 1 | 0.920 | sglang_runtime.md, sglang_runtime.md, model_loading_tokenizer.md |
| q13 NCCL tensor parallel communication timeout | keyword | 4 | 0.000 | prefill_decode_latency.md, throughput_regression.md, vllm_paged_attention.md |
| q13 NCCL tensor parallel communication timeout | bm25 | 4 | 0.000 | prefill_decode_latency.md, throughput_regression.md, vllm_paged_attention.md |
| q13 NCCL tensor parallel communication timeout | dense | 3 | 0.500 | node_failure_network.md, logging_observability.md, ncuda_nccl_tensor_parallel.md |
| q13 NCCL tensor parallel communication timeout | rrf | 2 | 0.631 | node_failure_network.md, ncuda_nccl_tensor_parallel.md, logging_observability.md |
| q13 NCCL tensor parallel communication timeout | hybrid_reranker | 2 | 0.631 | node_failure_network.md, ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md |
| q14 NCCL network rank desync | keyword | 1 | 0.613 | ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md |
| q14 NCCL network rank desync | bm25 | 1 | 0.613 | ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md, logging_observability.md |
| q14 NCCL network rank desync | dense | 1 | 0.613 | ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md |
| q14 NCCL network rank desync | rrf | 1 | 0.613 | ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md |
| q14 NCCL network rank desync | hybrid_reranker | 1 | 0.613 | ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md, ncuda_nccl_tensor_parallel.md |
| q15 CUDA kernel launch illegal memory access | keyword | 1 | 1.000 | cuda_kernel_errors.md, cuda_kernel_errors.md, cuda_kernel_errors.md |
| q15 CUDA kernel launch illegal memory access | bm25 | 1 | 1.000 | cuda_kernel_errors.md, cuda_kernel_errors.md, cuda_oom.md |
| q15 CUDA kernel launch illegal memory access | dense | 1 | 1.000 | cuda_kernel_errors.md, ncuda_oom_fragmentation.md, cuda_oom.md |
| q15 CUDA kernel launch illegal memory access | rrf | 1 | 1.000 | cuda_kernel_errors.md, cuda_oom.md, cuda_kernel_errors.md |
| q15 CUDA kernel launch illegal memory access | hybrid_reranker | 1 | 1.000 | cuda_kernel_errors.md, cuda_kernel_errors.md, cuda_kernel_errors.md |
| q16 Prometheus gpu_util ttft_p95_ms 指标 | keyword | 2 | 0.631 | gpu_utilization_anomaly.md, prometheus_metrics.md, prometheus_metrics.md |
| q16 Prometheus gpu_util ttft_p95_ms 指标 | bm25 | 1 | 1.000 | prometheus_metrics.md, prometheus_metrics.md, logging_observability.md |
| q16 Prometheus gpu_util ttft_p95_ms 指标 | dense | 4 | 0.000 | gpu_utilization_anomaly.md, multi_instance_load_balance.md, high_ttft.md |
| q16 Prometheus gpu_util ttft_p95_ms 指标 | rrf | 1 | 1.000 | prometheus_metrics.md, gpu_utilization_anomaly.md, high_ttft.md |
| q16 Prometheus gpu_util ttft_p95_ms 指标 | hybrid_reranker | 5 | 0.000 | high_ttft.md, multi_instance_load_balance.md, prefill_decode_latency.md |
| q17 日志 trace_id scheduler preempt | keyword | 1 | 0.613 | vllm_queue_batching.md, scheduler_queue.md, continuous_batching.md |
| q17 日志 trace_id scheduler preempt | bm25 | 1 | 0.920 | vllm_queue_batching.md, vllm_paged_attention.md, logging_observability.md |
| q17 日志 trace_id scheduler preempt | dense | 1 | 0.613 | logging_observability.md, high_ttft.md, prefill_decode_latency.md |
| q17 日志 trace_id scheduler preempt | rrf | 1 | 1.000 | logging_observability.md, vllm_queue_batching.md, prefill_decode_latency.md |
| q17 日志 trace_id scheduler preempt | hybrid_reranker | 2 | 0.693 | service_5xx_timeout.md, logging_observability.md, vllm_queue_batching.md |
| q18 服务超时 retry storm 限流 | keyword | 1 | 0.613 | rate_limit_retry.md, rate_limit_retry.md, ttft_tpot_p99.md |
| q18 服务超时 retry storm 限流 | bm25 | 1 | 0.613 | rate_limit_retry.md, rate_limit_retry.md, rate_limit_retry.md |
| q18 服务超时 retry storm 限流 | dense | 1 | 0.613 | service_5xx_timeout.md, ttft_tpot_p99.md, ncuda_oom_fragmentation.md |
| q18 服务超时 retry storm 限流 | rrf | 1 | 1.000 | rate_limit_retry.md, service_5xx_timeout.md, ttft_tpot_p99.md |
| q18 服务超时 retry storm 限流 | hybrid_reranker | 1 | 1.000 | rate_limit_retry.md, service_5xx_timeout.md, service_5xx_timeout.md |
| q19 多实例负载不均 consistent hashing | keyword | 未命中 | 0.000 | autoscaling_capacity.md, autoscaling_capacity.md, autoscaling_capacity.md |
| q19 多实例负载不均 consistent hashing | bm25 | 3 | 0.500 | node_failure_network.md, throughput_regression.md, multi_instance_load_balance.md |
| q19 多实例负载不均 consistent hashing | dense | 2 | 0.631 | node_failure_network.md, multi_instance_load_balance.md, kv_cache_capacity.md |
| q19 多实例负载不均 consistent hashing | rrf | 2 | 0.631 | node_failure_network.md, multi_instance_load_balance.md, throughput_regression.md |
| q19 多实例负载不均 consistent hashing | hybrid_reranker | 2 | 0.631 | throughput_regression.md, multi_instance_load_balance.md, gpu_oom_recovery.md |
| q20 节点故障网络丢包 readiness | keyword | 1 | 0.613 | health_check_readiness.md, health_check_readiness.md, health_check_readiness.md |
| q20 节点故障网络丢包 readiness | bm25 | 2 | 0.387 | ncuda_nccl_tensor_parallel.md, node_failure_network.md, node_failure_network.md |
| q20 节点故障网络丢包 readiness | dense | 1 | 0.613 | node_failure_network.md, ncuda_nccl_tensor_parallel.md, node_failure_network.md |
| q20 节点故障网络丢包 readiness | rrf | 2 | 0.387 | ncuda_nccl_tensor_parallel.md, node_failure_network.md, node_failure_network.md |
| q20 节点故障网络丢包 readiness | hybrid_reranker | 1 | 1.000 | node_failure_network.md, health_check_readiness.md, node_failure_network.md |
| q21 模型加载失败 tokenizer vocab mismatch | keyword | 1 | 1.000 | model_loading_tokenizer.md, model_loading_tokenizer.md, model_loading_tokenizer.md |
| q21 模型加载失败 tokenizer vocab mismatch | bm25 | 1 | 1.000 | model_loading_tokenizer.md, health_check_readiness.md, quantization_dtype.md |
| q21 模型加载失败 tokenizer vocab mismatch | dense | 1 | 1.000 | model_loading_tokenizer.md, sglang_runtime.md, health_check_readiness.md |
| q21 模型加载失败 tokenizer vocab mismatch | rrf | 1 | 1.000 | model_loading_tokenizer.md, health_check_readiness.md, quantization_dtype.md |
| q21 模型加载失败 tokenizer vocab mismatch | hybrid_reranker | 1 | 1.000 | model_loading_tokenizer.md, sglang_runtime.md, health_check_readiness.md |
| q22 throughput tokens per second regression | keyword | 1 | 1.000 | throughput_regression.md, autoscaling_capacity.md, gpu_utilization_anomaly.md |
| q22 throughput tokens per second regression | bm25 | 1 | 1.000 | throughput_regression.md, gpu_utilization_anomaly.md, gpu_utilization_anomaly.md |
| q22 throughput tokens per second regression | dense | 1 | 1.000 | throughput_regression.md, ttft_tpot_p99.md, throughput_regression.md |
| q22 throughput tokens per second regression | rrf | 1 | 1.000 | throughput_regression.md, continuous_batching.md, gpu_utilization_anomaly.md |
| q22 throughput tokens per second regression | hybrid_reranker | 4 | 0.000 | gpu_utilization_anomaly.md, autoscaling_capacity.md, gpu_utilization_anomaly.md |
| q23 tensor parallel quantization dtype memory | keyword | 5 | 0.000 | prefill_decode_latency.md, throughput_regression.md, vllm_paged_attention.md |
| q23 tensor parallel quantization dtype memory | bm25 | 5 | 0.000 | prefill_decode_latency.md, throughput_regression.md, vllm_paged_attention.md |
| q23 tensor parallel quantization dtype memory | dense | 5 | 0.000 | vllm_paged_attention.md, cuda_oom.md, ncuda_oom_fragmentation.md |
| q23 tensor parallel quantization dtype memory | rrf | 6 | 0.000 | vllm_paged_attention.md, cuda_oom.md, model_loading_tokenizer.md |
| q23 tensor parallel quantization dtype memory | hybrid_reranker | 2 | 0.387 | vllm_paged_attention.md, quantization_dtype.md, ncuda_oom_kv_cache.md |
| q24 prefix cache hit rate low | keyword | 1 | 0.613 | ncaching_prefix_kv.md, ncaching_prefix_kv.md, ncuda_oom_kv_cache.md |
| q24 prefix cache hit rate low | bm25 | 1 | 0.613 | ncaching_prefix_kv.md, ncuda_oom_kv_cache.md, ncaching_prefix_kv.md |
| q24 prefix cache hit rate low | dense | 1 | 1.000 | ncaching_prefix_kv.md, prefix_cache.md, ncaching_prefix_kv.md |
| q24 prefix cache hit rate low | rrf | 1 | 0.613 | ncaching_prefix_kv.md, ncaching_prefix_kv.md, ncuda_oom_kv_cache.md |
| q24 prefix cache hit rate low | hybrid_reranker | 1 | 1.000 | ncaching_prefix_kv.md, prefix_cache.md, ncaching_prefix_kv.md |
