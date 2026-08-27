from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from ..fixtures import FixtureError, load_metrics, search_logs as fixture_search_logs

try:  # The official MCP Python SDK is used when installed.
    from mcp.server.fastmcp import FastMCP  # type: ignore
except ImportError:  # pragma: no cover - exercised only in dependency-free setup
    FastMCP = None  # type: ignore


if FastMCP is not None:
    mcp = FastMCP("serving-diagnostics")

    @mcp.tool()
    async def get_metrics_snapshot(service_id: str) -> dict[str, Any]:
        """Read a metrics snapshot for one allowlisted demo inference service."""
        return load_metrics(service_id)

    @mcp.tool()
    async def search_logs(service_id: str, keyword: str, limit: int = 20) -> list[str]:
        """Search read-only logs for one allowlisted demo inference service."""
        return fixture_search_logs(service_id, keyword, limit)
else:
    mcp = None


def _fallback_result(request_id: int, result: Any = None, error: str | None = None) -> str:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id}
    if error:
        payload["error"] = {"code": -32000, "message": error}
    else:
        payload["result"] = result
    return json.dumps(payload, ensure_ascii=False) + "\n"


async def _fallback_stdio_server() -> None:
    """Tiny JSON-RPC fallback so the demo is runnable before pip install.

    Once the MCP SDK is installed, the normal entrypoint uses FastMCP and the
    client uses the official MCP initialize/tools/call protocol.
    """
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        request_id = int(request.get("id", 0))
        method = request.get("method")
        try:
            if method in {"initialize", "tools/list"}:
                result = {
                    "protocolVersion": "2024-11-05",
                    "tools": [
                        {"name": "get_metrics_snapshot", "description": "Read metrics", "inputSchema": {}},
                        {"name": "search_logs", "description": "Search logs", "inputSchema": {}},
                    ],
                }
            elif method == "tools/call":
                params = request.get("params", {})
                name = params.get("name")
                args = params.get("arguments", {})
                if name == "get_metrics_snapshot":
                    result = load_metrics(str(args.get("service_id", "")))
                elif name == "search_logs":
                    result = fixture_search_logs(
                        str(args.get("service_id", "")),
                        str(args.get("keyword", "")),
                        int(args.get("limit", 20)),
                    )
                else:
                    raise FixtureError(f"unknown tool: {name}")
            else:
                result = {}
            sys.stdout.write(_fallback_result(request_id, result=result))
        except Exception as exc:
            sys.stdout.write(_fallback_result(request_id, error=str(exc)))
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serving diagnostics MCP server")
    parser.add_argument("--fallback", action="store_true", help="run dependency-free JSON-RPC fallback")
    args = parser.parse_args()
    if args.fallback or mcp is None:
        asyncio.run(_fallback_stdio_server())
    else:
        # FastMCP speaks the official MCP stdio transport.
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
