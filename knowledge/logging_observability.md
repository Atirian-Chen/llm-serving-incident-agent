# 日志与可观测性排障

## 故障现象
无法从日志定位一次请求，日志缺少 request_id、rank 或时间戳，或者错误被采样丢失。

## 需要检查的指标和日志
检查结构化日志字段、trace_id/request_id、service_id、instance_id、model、rank、latency_ms、error_type 和日志采集延迟。

## 排查步骤
先按时间和 request_id 聚合网关、调度器、worker、CUDA/NCCL 日志，再关联 Prometheus 指标；保留首次错误和完整 traceback，避免只看最后一行。

## 修复建议与验证
统一 JSON 日志 schema 和时区，针对错误关闭采样并设置敏感字段脱敏；用一次故障演练验证从告警链接到原始日志可追溯。
