# Prefix Cache 与 KV Cache 命中

## 故障现象
重复系统提示没有带来预期加速，prefix cache hit rate 低，TTFT 随重复请求仍很高；KV cache 使用率却持续上升。

## 需要检查的指标和日志
检查 prefix_cache_hits、prefix_cache_lookups、hit_rate、evictions、cached_tokens、kv_cache_usage_perc 和 cache key 日志。

## 排查步骤
确认模板、tokenizer、空格和系统提示完全一致；检查租户隔离、缓存 TTL、最大缓存容量和 eviction 原因；区分 prefix 命中与 decode KV 占用。

## 修复建议与验证
规范化可缓存前缀，扩大合理的 cache 容量并设置 TTL，避免把用户私有内容放入共享键。用完全相同前缀的 A/B 请求验证 hit rate 与 TTFT 改善。
