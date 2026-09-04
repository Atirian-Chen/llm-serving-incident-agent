# SGLang 运行时故障

## 故障现象
SGLang 服务返回 5xx、请求卡住或 runtime worker 重启，常伴随 tokenizer、RadixAttention 或 scheduler 报错。

## 需要检查的指标和日志
检查 HTTP 5xx、active requests、queue latency、TTFT、GPU 显存和 worker restart count。日志搜索 RuntimeError、tokenizer、RadixCache、scheduler、traceback。

## 排查步骤
先用同一模型和 tokenizer 复现单请求，再逐步增加并发；确认启动参数、模型 revision、量化配置和 CUDA/PyTorch 版本匹配；查看 worker 是否因 NCCL 或 CUDA 错误退出。

## 修复建议与验证
固定模型与 tokenizer 版本，移除未经验证的量化或 attention 后端，必要时滚动重启异常 worker。验证健康检查、流式输出、长上下文和并发场景均无 5xx。
