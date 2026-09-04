# Serving 5xx 与请求超时

## 故障现象
客户端收到 5xx、504 或 timeout，可能集中发生在模型加载、长 prompt、GPU OOM 或 upstream 连接中断期间。

## 需要检查的指标和日志
检查 HTTP status、upstream_response_time、queue_time、request_duration、timeout_count、retry_count、worker exit code 和完整 traceback。

## 排查步骤
用 trace_id 串起网关、serving、MCP/上游日志；判断超时发生在排队、prefill、decode、网络还是客户端；检查重试是否放大流量。

## 修复建议与验证
统一端到端 deadline，限制重试次数并使用指数退避；对 OOM、模型加载和参数错误返回可区分状态码。验证故障恢复后 5xx、超时和重试曲线归零。
