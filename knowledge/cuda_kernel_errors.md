# CUDA Kernel 错误

## 故障现象
服务突然出现 illegal memory access、device-side assert、CUBLAS_STATUS_EXECUTION_FAILED 或 kernel launch failure，随后 worker 不再接收请求。

## 需要检查的指标和日志
记录 GPU Xid、ECC、温度、功耗、driver reset 和进程退出码；日志搜索 CUDA、kernel、assert、illegal memory、CUBLAS、Xid。

## 排查步骤
保存首次错误前后的完整日志，开启 CUDA_LAUNCH_BLOCKING=1 在隔离环境复现；确认驱动、CUDA runtime、PyTorch 与编译扩展 ABI 一致；排除单一输入触发的越界。

## 修复建议与验证
升级或回滚到兼容的驱动与 serving 镜像，重建自定义 CUDA 扩展并隔离坏卡。清理进程后执行短压测和长压测，确认不再出现 Xid 或 kernel error。
