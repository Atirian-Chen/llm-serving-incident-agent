# LLM Serving Incident Agent

一个面向 vLLM/SGLang 推理服务的最小故障诊断 Agent。它把告警、指标、日志和排障手册串成一个可追踪、可评测的闭环：

```text
FastAPI → LangGraph → RAG → Function Calling → MCP Client/Server
        → 指标/日志证据 → Pydantic Structured Output → Trace/Eval
```

项目只覆盖四类故障：CUDA OOM、TTFT/P95 过高、Prefix Cache 命中率低、Unknown。指标和日志使用本地 fixture，工具全部只读，方便学习和复现；没有接入真实生产集群。

## 运行

需要 Python 3.11+。项目提供 E 盘本地环境脚本。首次安装只需要执行一次：

```powershell
.\scripts\setup.ps1
```

脚本会创建 `.venv-standard`，并把 pip、模型下载和临时文件统一放在项目目录下的
`.runtime`，不会把项目依赖安装到 C 盘。`requirements.lock` 固定了当前可复现的核心、
测试和 MCP/LangGraph/DeepSeek 客户端依赖。

配置在线 DeepSeek 模型后启动 API：

```powershell
Copy-Item config/llm.example.toml config/llm.toml
# 编辑 config/llm.toml，填写 deepseek.api_key
.\.venv-standard\Scripts\python.exe -m uvicorn incident_agent.api:app --app-dir src --reload
```

请求示例：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/diagnose `
  -Method Post -ContentType "application/json" `
  -Body '{"service_id":"demo-oom","symptom":"请求失败且 GPU 显存接近满载，请判断原因。"}'
```

模型固定使用 `config/llm.toml` 中的 DeepSeek OpenAI-compatible API，通过
Function Calling 自主选择两个只读工具，最终使用 `IncidentReport` Structured Output。
如果配置文件或 API Key 缺失，`/health` 会返回 `status=degraded`，`/diagnose`
会返回 HTTP 503 和 `LLM unavailable: ...`；项目不会生成离线假报告。

## MCP

`incident_agent.mcp_server.server` 使用官方 MCP Python SDK 的 FastMCP 暴露：

- `get_metrics_snapshot(service_id)`
- `search_logs(service_id, keyword, limit)`

Agent 通过 `MCPToolClient` 作为 MCP Client 调用独立进程。工具服务端校验 service
allowlist、限制日志返回数量并且不接受任意文件路径；客户端和服务端均使用官方
MCP Python SDK 的 stdio transport。

## RAG

生产路径默认是严格的两阶段 Hybrid RAG，不会静默退回 keyword 或离线规则：

```text
Markdown → jieba/BM25 Top-20 ┐
                             ├→ chunk_id 去重 → RRF → BGE Cross-Encoder → Top-3
Markdown → BGE/Chroma Top-20 ┘
```

首次运行需要在有 NVIDIA CUDA 的机器上下载 `BAAI/bge-small-zh-v1.5` 和
`BAAI/bge-reranker-v2-m3`，并建立 Chroma cosine collection：

```powershell
$env:HF_HOME = "$PWD\.runtime\huggingface"
$env:TRANSFORMERS_CACHE = "$PWD\.runtime\huggingface"
$env:TORCH_HOME = "$PWD\.runtime\torch"
.\.venv-standard\Scripts\python.exe scripts/build_index.py
```

`RAG_STRICT=true`、`RAG_PROVIDER=hybrid`、最终返回 Top-3 是默认配置。缺少
`rank-bm25`、`jieba`、Chroma、SentenceTransformers、模型或 CUDA 时会给出可操作的
错误并终止请求；KeywordRetriever 只保留给独立评测基线，不能被生产 Agent 选用。
全部模型、pip 缓存和临时文件由 `scripts/setup.ps1` 放在 E 盘项目目录的 `.runtime` 下。

独立检索评测（不调用 LLM）可以选择真实模型或明确标注的 mock 模式：

```powershell
.\.venv-standard\Scripts\python.exe scripts/rag_eval.py --mode auto
# 无法下载模型时显式使用：
.\.venv-standard\Scripts\python.exe scripts/rag_eval.py --mode mock
```

结果写入 `reports/rag_eval.json` 和 `reports/rag_eval.md`，比较 Keyword、BM25、Dense、
RRF 和 Cross-Encoder 五条路径。

## Trace 与评测

每次运行会写入 `data/traces.jsonl`。除阶段耗时外，日志保留原始用户请求、BM25/Dense/RRF/Reranker
候选及分数、发送给 LLM 的完整消息、LLM 工具调用原消息、MCP 参数和返回的 tool message、
工具结果后的最终 LLM 请求以及最终用户报告。`example.md` 是一条可读的代表性链路记录。
配置 `LANGSMITH_TRACING=true` 和 `LANGSMITH_API_KEY` 后可额外发送到 LangSmith。

运行 30 个固定案例（需要在线 DeepSeek API Key）：

```powershell
.\scripts\evaluate.ps1
```

结果写入 `data/metrics.json`，包含故障类型准确率、工具选择准确率、Schema 成功率、证据覆盖率、平均延迟和 P95 延迟。未配置在线模型时评测会直接以 `LLM unavailable` 退出，不会写入虚假指标。简历数字必须来自这个文件或你自己的真实复测。

## 测试

```powershell
.\scripts\test.ps1
```

后续测试不需要再次安装依赖。若要从头重建 E 盘环境，执行
`.\scripts\setup.ps1 -ForceRecreate`；普通开发只运行 `.\scripts\test.ps1` 即可。

测试覆盖 Schema、fixture 安全边界、RAG 检索、MCP 独立进程调用、LangGraph 路由和 API。

## 目录

```text
src/incident_agent/
  api.py             # FastAPI
  graph.py           # LangGraph 状态图和节点
  models.py          # 在线 DeepSeek + Function Calling + Structured Output
  mcp_client.py      # MCP Client
  mcp_server/        # FastMCP Server
  rag/               # strict BM25 + BGE/Chroma + RRF + Cross-Encoder
  schemas.py         # Pydantic 输入、输出、工具调用 schema
knowledge/           # vLLM/SGLang 排障手册
fixtures/            # 模拟指标和日志
eval/                # 30 个固定案例
scripts/             # 建索引、评测
tests/               # 自动化测试
```

## 面试边界

项目选择单 Agent、严格 Hybrid RAG、Chroma 和本地 fixture，是为了优先完成可评测闭环。手写 while 循环也能完成这个小任务，LangGraph 的价值在于把状态、条件路由和可恢复节点显式化。MCP 负责工具协议，不负责权限；生产化还需要身份、审计、沙箱、多租户、真实集群适配和更严格的 Prompt Injection 防护。在线模型或严格 RAG 不可用时系统明确失败，避免把猜测伪装成诊断结果。

## 简历使用规则

这是一个可运行的学习项目，但简历中的案例数、准确率、延迟和“提升”必须以你实际执行后的结果为准。没有真实运行前，不要填写虚构百分比，也不要把 mock 评测、BGE/Chroma 或云 Trace 写成已经在线使用。
