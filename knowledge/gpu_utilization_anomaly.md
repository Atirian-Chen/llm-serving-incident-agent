# GPU 利用率异常

## 故障现象
GPU utilization 长时间低于预期但请求延迟升高，或 utilization 100% 而 tokens/s 很低。单看 utilization 不能直接判断瓶颈。

## 需要检查的指标和日志
同时查看 gpu_util、sm_util、mem_util、power_w、clock、PCIe/NVLink throughput、running/waiting requests、TTFT、TPOT 和 CPU 利用率。

## 排查步骤
区分 prefill 与 decode；低利用率时检查队列、网络、tokenizer 和 CPU 调度，高利用率时检查 kernel、显存带宽和 batch 形状；比较每个实例和每张卡。

## 修复建议与验证
修复请求分发、tokenizer 线程或批处理参数，避免只通过提高并发掩盖瓶颈。用固定输入集验证 tokens/s、P95 延迟与 GPU 指标同时改善。
