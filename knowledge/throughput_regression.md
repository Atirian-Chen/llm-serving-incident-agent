# Throughput 下降

## 故障现象
并发增加后 tokens/s 下降，GPU 仍有余量，或部署升级后吞吐回退。请求数 QPS 与 token throughput 需要分开分析。

## 需要检查的指标和日志
检查 input_tokens/s、output_tokens/s、requests/s、batch token 数、GPU memory bandwidth、SM occupancy、CPU tokenizer 时间和版本变更日志。

## 排查步骤
用固定 prompt/output 长度做基准，分别测试单实例和多实例；比较量化、attention backend、CUDA graph、tensor parallel 与 batch 参数；排除上游限流。

## 修复建议与验证
回滚已知性能回退的镜像或 kernel，重新调优 batch 和并行度。用相同硬件和数据集复测，并保留基线与新版本的 P50/P99。
