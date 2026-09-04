from __future__ import annotations

from typing import Any

from ..fixtures import load_metrics, search_logs as fixture_search_logs

try:
    from mcp.server.fastmcp import FastMCP  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency installation issue
    raise RuntimeError(
        "MCP unavailable: install project dependencies with pip install -e ."
    ) from exc


mcp = FastMCP("serving-diagnostics")


@mcp.tool()
async def get_metrics_snapshot(service_id: str) -> dict[str, Any]:
    """Read a metrics snapshot for one allowlisted demo inference service."""
    return load_metrics(service_id)


@mcp.tool()
async def search_logs(service_id: str, keyword: str, limit: int = 20) -> list[str]:
    """Search read-only logs for one allowlisted demo inference service."""
    return fixture_search_logs(service_id, keyword, limit)


def main() -> None:
    # FastMCP speaks the official MCP stdio transport.
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
