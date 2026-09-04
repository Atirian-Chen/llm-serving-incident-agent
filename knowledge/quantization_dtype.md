# 量化与数据类型兼容

## 故障现象
量化模型启动失败、生成质量下降或吞吐异常，日志包含 dtype、kernel unsupported、bitsandbytes 或 dequantization 错误。

## 需要检查的指标和日志
检查模型量化格式、compute dtype、显卡架构、显存、kernel backend、加载耗时和错误 traceback。

## 排查步骤
在单卡固定输入上比较 FP16/BF16 与量化版本，确认 attention、GEMM 和 KV cache dtype；检查量化权重是否完整。

## 修复建议与验证
选择与 GPU 架构匹配的量化 kernel，必要时回退到已验证 dtype；用准确率烟测与 tokens/s、P99 双重验证后再扩容。
