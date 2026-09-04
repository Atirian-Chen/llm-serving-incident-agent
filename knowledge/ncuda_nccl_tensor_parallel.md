# NCCL 与 Tensor Parallel 通信异常

## 故障现象
多卡服务启动卡住、吞吐骤降或 worker 退出，日志包含 NCCL timeout、unhandled system error、connection closed 或 rank mismatch。

## 需要检查的指标和日志
检查每个 rank 的启动日志、NCCL_DEBUG、NCCL_SOCKET_IFNAME、GPU 拓扑、NVLink/PCIe 错误计数、通信耗时和节点网络丢包。

## 排查步骤
确认所有 rank 使用相同模型、dtype、TP size 和 world size；运行 nccl-tests 验证硬件链路；检查容器共享内存、端口、防火墙与网卡选择。

## 修复建议与验证
修正网卡和 NCCL 环境变量，增加通信超时仅用于诊断，不作为根治；隔离故障 GPU 或节点。完成 all-reduce 基准和实际推理压测后再恢复流量。
