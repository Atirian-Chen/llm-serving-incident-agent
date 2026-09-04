# 扩缩容与容量规划

## 故障现象
扩容滞后导致排队和 P99 告警，或频繁扩缩容造成模型加载风暴与吞吐抖动。

## 需要检查的指标和日志
检查 queue length、pending pods、startup duration、GPU availability、scale-up/down events、冷启动失败和实例利用率。

## 排查步骤
估算每实例 tokens/s 与最大并发，比较扩容触发信号和真实 SLO；检查节点池 GPU 配额、镜像拉取与模型缓存命中。

## 修复建议与验证
使用队列和 token 速率的复合指标，设置冷却窗口与最小容量，预热模型缓存；用流量阶跃测试验证扩容时间覆盖 SLO。
