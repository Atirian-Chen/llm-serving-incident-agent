import pytest

from incident_agent.mcp_client import MCPToolClient


pytest.importorskip("mcp")


@pytest.mark.asyncio
async def test_mcp_client_calls_a_separate_server_process():
    async with MCPToolClient() as client:
        metrics = await client.call_tool("get_metrics_snapshot", {"service_id": "demo-oom"})
        assert metrics["service_id"] == "demo-oom"
        logs = await client.call_tool(
            "search_logs", {"service_id": "demo-oom", "keyword": "out of memory", "limit": 5}
        )
        assert logs and "out of memory" in logs[0].lower()
