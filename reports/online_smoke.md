# 真实 Hybrid 在线单案例核验

本页核验已保存的在线日志，没有再次调用 DeepSeek。该案例与历史 30 条评测分开记录。

- Run ID：`28c990bb-c43e-44bc-afa7-25ab78190a2a`
- 原始运行时间：`2026-09-04T16:02:46.709000+00:00`
- 模型：`deepseek-v4-flash`；RAG device：`cuda`
- 耗时：68.31 秒；MCP 调用：6 次；最终文档：3 条
- 请求：vLLM 请求排队增长，TTFT 和 P99 延迟升高
- 服务：`demo-high-ttft`（本地 fixture）
- [可读完整流程](../example.md) / [原始日志](traces/online_smoke.jsonl) / [核验 JSON](online_smoke.json)
- Trace SHA-256：`57c491a88ddde23de5ec884c6099d04460eec3d7a4c6337f8ae4c98d6b82b58b`

| 检查项 | 结果 |
| --- | --- |
| hybrid_trace_present | 通过 |
| final_top_three | 通过 |
| cuda_recorded | 通过 |
| deepseek_response_metadata_present | 通过 |
| mcp_arguments_valid | 通过 |
| final_report_recorded | 通过 |

核验覆盖日志中的 Hybrid、Top-3、CUDA、DeepSeek 响应、执行工具参数和最终报告；不衡量根因判断是否正确。仅一个成功案例，尚未构成六类故障或 30 条完整 Hybrid 在线回归。

日志未保留全部 Cross-Encoder 候选分数及最终 HTTP 原始响应。工具的展示消息与实际 user JSON context 的区别见 example.md。
