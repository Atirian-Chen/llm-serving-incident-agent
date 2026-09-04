# Continuous Batching

## 故障现象
开启 continuous batching 后吞吐提高但短请求 TTFT 变差，或者 batch 频繁重组造成延迟抖动。

## 需要检查的指标和日志
检查 batch_size、batch_tokens、join/leave rate、prefill/decode 比例、TTFT/TPOT 和 scheduler decision 日志。

## 排查步骤
按短请求、长请求和输出长度分组，观察 batch 是否被单个长序列拖住；核对 max_num_seqs、token budget 和 chunked prefill 配置。

## 修复建议与验证
调整 token budget 和调度公平性，必要时隔离长请求队列；用混合负载压测比较吞吐、P99 和短请求 SLO，而不是只看平均吞吐。
