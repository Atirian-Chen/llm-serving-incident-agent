# vLLM 请求排队与批处理

## 故障现象
vLLM 请求持续处于 waiting，scheduler queue length 增长，吞吐没有随并发提升。短请求也出现高 TTFT，常见于 max_num_batched_tokens 过小或 decode token 被长 prompt 挤压。

## 需要检查的指标和日志
检查 vllm:num_requests_waiting、vllm:num_requests_running、request_queue_time、ttft_ms、tpot_ms、tokens_per_second 和 gpu_util。日志搜索 scheduler、preempt、swap、waiting。

## 排查步骤
按时间对齐队列长度、prefill/decode token 数和 GPU 显存；确认是否有单个超长 prompt 占满 batch；检查 max_num_seqs、max_num_batched_tokens、gpu_memory_utilization 以及是否频繁 preemption。

## 修复建议与验证
在显存余量允许时提高批处理 token 上限，限制异常 prompt 长度并分离长短请求。修复后观察 15 分钟，队列应回落，TTFT P95 与吞吐恢复且无 OOM。
