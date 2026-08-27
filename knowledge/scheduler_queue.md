# Scheduler queue backlog

## 常见信号
- waiting requests 持续增长，TTFT P95 同步变差。
- 日志出现 scheduler queue timeout 或 batch waiting。

## 可能根因
- 到达率超过服务处理能力。
- 调度批次大小、并发上限或流量突增不匹配。

## 建议动作
- 先区分 prefill 和 decode 的排队时间，再调整并发和批处理。
- 以固定负载测试吞吐、TTFT 和 P95 的联合变化。

