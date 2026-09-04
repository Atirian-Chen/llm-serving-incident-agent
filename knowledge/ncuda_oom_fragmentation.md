# CUDA OOM 与显存碎片

## 故障现象
free memory 看似仍有空间却分配失败，错误包含 CUDA out of memory、fragmentation 或无法分配连续 block。重启 worker 后短暂恢复。

## 需要检查的指标和日志
检查 allocated/reserved bytes、largest_free_block、gpu_memory_utilization、KV cache 使用率、请求最大上下文和 OOM traceback。

## 排查步骤
区分权重、KV cache、CUDA graph、临时 workspace 和 allocator reserved；观察长短请求混合压测下碎片随时间增长；确认是否存在泄漏。

## 修复建议与验证
降低并发或上下文上限，给 graph/workspace 留余量，升级 allocator 或 serving 版本；必要时周期性滚动重启。验证 24 小时压测无 OOM 且 largest_free_block 稳定。
