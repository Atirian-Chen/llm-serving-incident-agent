# 健康检查与就绪状态

## 故障现象
服务端口可连接但请求全部失败，或模型尚未加载就被负载均衡器接收，导致启动期间 5xx。

## 需要检查的指标和日志
检查 liveness、readiness、model_loaded、worker_ready、startup_duration、active_connections 和探针响应日志。

## 排查步骤
确认 readiness 只在模型、tokenizer、CUDA context 和工具依赖就绪后返回成功；区分进程存活与可接收推理请求。

## 修复建议与验证
增加启动宽限时间和真实推理 readiness，故障时快速摘除但避免重启风暴。滚动发布时验证旧实例排空、新实例预热和探针切换。
