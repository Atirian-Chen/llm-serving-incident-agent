# GPU memory investigation

## 常见信号
- GPU memory utilization 长时间接近上限。
- 显存使用在请求峰值时明显抖动。

## 可能根因
- 权重、激活、KV Cache 和 CUDA Graph 共同占用显存。

## 建议动作
- 同时采集显存、请求长度、并发、KV Cache 与错误日志，避免只凭单一指标判断。

