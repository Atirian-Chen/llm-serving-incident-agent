from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from .config import ROOT


class MCPClientError(RuntimeError):
    pass


class MCPToolClient:
    """MCP client that uses the official Python SDK over stdio transport."""

    def __init__(self, server_module: str = "incident_agent.mcp_server.server") -> None:
        self.server_module = server_module

    async def __aenter__(self) -> "MCPToolClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise MCPClientError(
                "MCP unavailable: install project dependencies with pip install -e ."
            ) from exc

    async def aclose(self) -> None:
        # stdio_client owns and closes each short-lived process per call.
        return None

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 5.0) -> Any:
        await self.start()
        return await asyncio.wait_for(self._call_official_sdk(name, arguments), timeout=timeout)

    async def _call_official_sdk(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError as exc:  # pragma: no cover - guarded by start()
            raise MCPClientError("MCP unavailable: official SDK import failed") from exc

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self.server_module],
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "src")
                + os.pathsep
                + os.environ.get("PYTHONPATH", ""),
            },
        )
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments=arguments)
        except Exception as exc:
            raise MCPClientError(f"MCP tool call failed: {exc}") from exc

        if getattr(result, "isError", False):
            raise MCPClientError(str(result))
        structured = getattr(result, "structuredContent", None) or getattr(
            result, "structured_content", None
        )
        if isinstance(structured, dict):
            if "result" in structured:
                return structured["result"]
            if "value" in structured:
                return structured["value"]
        content = getattr(result, "content", result)
        if isinstance(content, list) and content:
            texts = [getattr(item, "text", None) for item in content]
            texts = [text for text in texts if text is not None]
            if texts:
                if len(texts) > 1:
                    return texts
                try:
                    return json.loads(texts[0])
                except json.JSONDecodeError:
                    return texts[0]
        return content
