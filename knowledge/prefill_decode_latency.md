# Prefill 与 Decode 延迟

## 故障现象
TTFT 显著升高而 TPOT 正常，通常是 prefill 排队或长 prompt；TPOT 升高而 TTFT 正常，通常是 decode 阶段 GPU 争用、KV cache 或通信瓶颈。

## 需要检查的指标和日志
拆分 prefill_ms、decode_ms、queue_ms、ttft_ms、tpot_ms、input_tokens、output_tokens、batch_size 和 scheduler 日志。

## 排查步骤
按输入长度分桶比较 P50/P95/P99；检查 chunked prefill、continuous batching、max_num_batched_tokens 和 decode batch；核对是否启用 tensor parallel。

## 修复建议与验证
为长 prompt 启用合理的 chunked prefill，调整 batch token 预算并限制输出长度。使用短、长两组基准分别验证 TTFT 与 TPOT，不以单一平均值判断。
