# Prometheus 指标排障

## 故障现象
告警触发但面板无数据、指标名称变化或 scrape 延迟导致诊断误判。高基数标签也可能拖慢 Prometheus。

## 需要检查的指标和日志
检查 scrape_up、scrape_duration_seconds、sample ingestion、target health、query error，以及服务导出的 gpu_util、queue、latency 指标。

## 排查步骤
确认指标端点、端口、TLS 和服务发现；用 PromQL 查询原始时间序列并核对 label；检查重启前后 metric name 与版本变更。

## 修复建议与验证
固定 recording rules 和 dashboard 版本，降低高基数标签，设置合理 scrape interval。用一条告警从触发到根因查询完整走通。
