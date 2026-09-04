# GPU OOM 恢复流程

## 故障现象
worker 因 CUDA out of memory 退出，健康检查失败，重启后服务短暂正常但高峰再次故障。

## 需要检查的指标和日志
保留 OOM 前 10 分钟的显存、KV cache、请求 token、并发、重试和 worker 生命周期；搜索 CUDA out of memory、allocation、preempt。

## 排查步骤
先摘除故障实例避免反复重启，再判断泄漏、碎片、单请求超长或容量不足；核对重启前后的模型与参数完全一致。

## 修复建议与验证
实施上下文/并发 admission control，修正 allocator 或模型配置，滚动替换异常 worker；至少完成一次高峰流量演练后再恢复全部权重。
