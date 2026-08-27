from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from .config import ROOT


class MCPClientError(RuntimeError):
    pass


class MCPToolClient:
    """MCP client with official SDK transport and a development fallback.

    The SDK path is the one used in the project demo. The fallback is useful
    for tests on a fresh Python installation and still exercises a separate
    process plus request/response protocol rather than importing tool code.
    """

    def __init__(
        self,
        server_module: str = "incident_agent.mcp_server.server",
        force_fallback: bool = False,
    ) -> None:
        self.server_module = server_module
        self._process: asyncio.subprocess.Process | None = None
        self._reader_lock = asyncio.Lock()
        self._request_id = 0
        self._sdk_available = self._check_sdk() and not force_fallback

    @staticmethod
    def _check_sdk() -> bool:
        try:
            import mcp  # noqa: F401

            return True
        except ImportError:
            return False

    async def __aenter__(self) -> "MCPToolClient":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def start(self) -> None:
        if self._process is not None:
            return
        if self._sdk_available:
            # SDK sessions are opened per call to keep lifecycle simple and
            # avoid leaking stdio handles across FastAPI requests.
            return
        self._process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            self.server_module,
            "--fallback",
            cwd=str(ROOT),
            env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def aclose(self) -> None:
        if self._process is not None:
            self._process.terminate()
            await self._process.wait()
            self._process = None

    async def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 5.0) -> Any:
        if self._sdk_available:
            return await asyncio.wait_for(self._call_official_sdk(name, arguments), timeout=timeout)
        await self.start()
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise MCPClientError("MCP process is not running")
        async with self._reader_lock:
            self._request_id += 1
            request_id = self._request_id
            request = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
            self._process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode())
            await self._process.stdin.drain()
            raw = await asyncio.wait_for(self._process.stdout.readline(), timeout=timeout)
        if not raw:
            raise MCPClientError("MCP server closed stdout")
        response = json.loads(raw.decode())
        if "error" in response:
            raise MCPClientError(response["error"].get("message", "MCP tool error"))
        return response.get("result")

    async def _call_official_sdk(self, name: str, arguments: dict[str, Any]) -> Any:
        try:
            from mcp import ClientSession, StdioServerParameters  # type: ignore
            from mcp.client.stdio import stdio_client  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise MCPClientError("MCP SDK import failed") from exc
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", self.server_module],
            env={**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")},
        )
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments=arguments)
                if getattr(result, "isError", False):
                    raise MCPClientError(str(result))
                structured = getattr(result, "structuredContent", None) or getattr(
                    result, "structured_content", None
                )
                if isinstance(structured, dict):
                    # FastMCP commonly wraps a return value as {"result": value}.
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
                            return texts if name == "search_logs" else texts[0]
                return content
