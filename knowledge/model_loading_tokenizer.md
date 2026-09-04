# 模型加载与 Tokenizer 问题

## 故障现象
服务无法启动、首个请求失败或 token 数异常，日志出现 model not found、safetensors、vocab、special token、chat template 或 tokenizer mismatch。

## 需要检查的指标和日志
检查模型 revision、文件校验和、磁盘空间、下载缓存、dtype、quantization、tokenizer 版本以及启动 traceback。

## 排查步骤
在隔离环境校验模型清单与 SHA，单独执行 tokenizer encode/decode 和 chat template；确认模型配置与 serving 版本兼容，避免半下载目录被复用。

## 修复建议与验证
固定 revision，清理损坏缓存并重新下载到 E 盘模型缓存；统一 tokenizer 和模型来源。启动后执行短 prompt、中文、工具调用和长上下文验证。
