# KV Cache 容量不足

## 故障现象
服务启动成功但高并发后拒绝请求，日志显示 KV cache capacity、block allocation 或 maximum sequence length 不足。

## 需要检查的指标和日志
检查 kv_cache_capacity_tokens、kv_cache_usage_perc、active_sequences、max_model_len、num_preemptions 和 block allocation 日志。

## 排查步骤
按模型 hidden size、层数、KV heads、dtype 和 TP 大小估算每 token 字节数；比较预留给权重与 KV cache 的显存；确认 prefix cache 是否占用容量。

## 修复建议与验证
降低 max_model_len、并发或 prefix cache 保留时间，或增加 GPU/TP 分片；不要仅提高显存比例掩盖权重空间不足。验证高峰流量下无拒绝和 preemption。
