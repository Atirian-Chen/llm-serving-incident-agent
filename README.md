# LLM Serving Incident Agent

一个面向 vLLM/SGLang 推理服务的最小故障诊断 Agent。它把告警、指标、日志和排障手册串成一个可追踪、可评测的闭环：

```text
FastAPI → LangGraph → RAG → Function Calling → MCP Client/Server
        → 指标/日志证据 → Pydantic Structured Output → Trace/Eval
```

项目只覆盖四类故障：CUDA OOM、TTFT/P95 过高、Prefix Cache 命中率低、Unknown。指标和日志使用本地 fixture，工具全部只读，方便学习和复现；没有接入真实生产集群。

## 运行

需要 Python 3.11+。推荐使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[all]"
```

不需要真实模型或 API Key 时，可以使用确定性的离线模型运行评测：

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate.py
```

安装完整依赖后启动 API：

```powershell
Copy-Item .env.example .env
python -m uvicorn incident_agent.api:app --app-dir src --reload
```

请求示例：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/diagnose `
  -Method Post -ContentType "application/json" `
  -Body '{"service_id":"demo-oom","symptom":"请求失败且 GPU 显存接近满载，请判断原因。"}'
```

默认 `MODEL_PROVIDER=rule`，不需要 API Key；它会使用确定性的本地模型替身跑通完整图和评测。使用本地 vLLM/SGLang 的 OpenAI-compatible endpoint 时：

```text
MODEL_PROVIDER=openai
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_API_KEY=local-dev-key
OPENAI_MODEL=你的模型名
```

此时模型通过 Function Calling 自主选择两个只读工具，最终使用 `IncidentReport` Structured Output。

## MCP

`incident_agent.mcp_server.server` 使用官方 MCP Python SDK 的 FastMCP 暴露：

- `get_metrics_snapshot(service_id)`
- `search_logs(service_id, keyword, limit)`

Agent 通过 `MCPToolClient` 作为 MCP Client 调用独立进程，工具服务端校验 service allowlist、限制日志返回数量并且不接受任意文件路径。全新环境尚未安装 SDK 时，客户端会启动同一独立进程的 JSON-RPC fallback，便于先跑通学习闭环；安装依赖后自动走官方 MCP stdio transport。

## RAG

离线默认使用可解释的 `KeywordRetriever`，用于无模型、无网络测试。要使用项目简历中对应的 BGE + Chroma 实现：

```powershell
$env:RAG_PROVIDER = "chroma"
python scripts/build_index.py
```

这会使用 `BAAI/bge-small-zh-v1.5` 生成 embedding 并持久化到 `data/chroma`。首次运行需要下载模型。

## Trace 与评测

每次运行会写入 `data/traces.jsonl`，能看到 `rag.retrieve`、`model.decide`、`mcp.tool_call`、`structured_output` 等事件。配置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 后可额外发送到 LangSmith。

运行 30 个固定案例：

```powershell
$env:PYTHONPATH = "src"
python scripts/evaluate.py
```

结果写入 `data/metrics.json`，包含故障类型准确率、工具选择准确率、Schema 成功率、证据覆盖率、平均延迟和 P95 延迟。简历数字必须来自这个文件或你自己的真实复测。

## 测试

```powershell
python -m pytest
```

测试覆盖 Schema、fixture 安全边界、RAG 检索、MCP 独立进程调用、LangGraph 路由和 API。

## 目录

```text
src/incident_agent/
  api.py             # FastAPI
  graph.py           # LangGraph 状态图和节点
  models.py          # Rule provider / OpenAI-compatible provider
  mcp_client.py      # MCP Client
  mcp_server/        # FastMCP Server
  rag/               # Keyword + BGE/Chroma retriever
  schemas.py         # Pydantic 输入、输出、工具调用 schema
knowledge/           # vLLM/SGLang 排障手册
fixtures/            # 模拟指标和日志
eval/                # 30 个固定案例
scripts/             # 建索引、评测
tests/               # 自动化测试
```

## 面试边界

第一版选择单 Agent、Chroma 和本地 fixture，是为了优先完成可评测闭环。手写 while 循环也能完成这个小任务，LangGraph 的价值在于把状态、条件路由和可恢复节点显式化。MCP 负责工具协议，不负责权限；生产化还需要身份、审计、沙箱、多租户、真实集群适配和更严格的 Prompt Injection 防护。

## 简历使用规则

这是一个可运行的学习项目，但简历中的案例数、准确率、延迟和“提升”必须以你实际执行后的结果为准。没有真实运行前，不要填写虚构百分比，也不要把可选的 BGE/Chroma 或云 Trace 写成已经使用。
