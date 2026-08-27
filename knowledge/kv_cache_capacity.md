# KV Cache capacity

## 常见信号
- 显存尚未完全占满但可用 KV blocks 不足。
- 长上下文请求触发分配失败或频繁 eviction。

## 可能根因
- 上下文长度和并发的乘积超过 KV Cache 预算。
- 量化、block size 或缓存保留策略不适合当前流量。

## 建议动作
- 估算每 token KV 开销，降低并发或最大上下文长度并回归。

