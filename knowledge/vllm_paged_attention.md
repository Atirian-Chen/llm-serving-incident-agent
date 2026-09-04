# vLLM PagedAttention

## 故障现象
PagedAttention 相关请求延迟抖动，KV block 分配失败或出现频繁 preemption。GPU 利用率可能不高，但显存已被 KV cache 和权重占用。

## 需要检查的指标和日志
检查 kv_cache_usage_perc、num_preemptions、gpu_memory_utilization、free_memory_bytes 以及 block_manager、KV cache、preempt 日志。

## 排查步骤
确认模型层数、head 数、上下文长度和并发估算出的 KV cache 需求；核对 tensor parallel 分片后每卡显存；区分权重占用、CUDA graph workspace 和 KV cache 占用。

## 修复建议与验证
降低最大上下文或并发，调整 gpu_memory_utilization 并保留 CUDA graph 余量；升级到包含已知 PagedAttention 修复的版本。验证连续压测无 block 分配错误且 preemption 下降。
