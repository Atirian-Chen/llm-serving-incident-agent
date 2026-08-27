# High TTFT and P95 latency

## 常见信号
- `ttft_p95_ms` 持续超过服务目标，等待队列增长。
- 日志包含 scheduler queue、batching delay 或 prefill backlog。

## 可能根因
- Prefill 阶段过载，`max_num_seqs` 与请求长度不匹配。
- 调度器等待批次或 GPU 利用率达到瓶颈。

## 建议动作
- 对 TTFT、queue time、prefill time 分段观测。
- 调整 batching 和调度参数，并用固定流量回归 P95。

## 风险提示
- 不能只看平均延迟；要同时观察 P95、吞吐和错误率。

