# Prefix Cache 命中率低

## 常见信号
- `prefix_cache_hit_rate` 长时间低于目标。
- 相同 system prompt 的请求仍反复执行 prefill。

## 可能根因
- Prefix 文本不稳定，动态字段放在共享前缀中。
- 请求路由或 tokenizer 配置导致缓存键不一致。

## 建议动作
- 检查共享前缀是否稳定、tokenizer 版本是否一致。
- 记录 cache hit/miss、前缀长度和路由分布后再调整。

## 风险提示
- 提升命中率不能以错误复用用户私有上下文为代价。

