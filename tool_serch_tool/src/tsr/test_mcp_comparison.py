"""
Test Tool Search Pattern with MCP servers.
Compares token usage with/without tool search using real MCP tools.
"""

import json
import asyncio
import os
from dataclasses import dataclass
from typing import Callable, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# MCP imports
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client, StdioServerParameters

# Reuse from main
from main import Tool, VectorToolSearcher, ToolSearcher


# =============================================================================
# MCP TOOL LOADER
# =============================================================================

async def load_tools_from_mcp_sse(url: str) -> list[Tool]:
    """Load tools from an MCP server via SSE."""
    tools = []

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            result = await session.list_tools()

            for mcp_tool in result.tools:
                # Create a closure to capture the session and tool name
                tool_name = mcp_tool.name

                def make_executor(name: str, sess_url: str):
                    async def execute(**kwargs):
                        # Need to reconnect for each call in sync context
                        async with sse_client(sess_url) as (r, w):
                            async with ClientSession(r, w) as s:
                                await s.initialize()
                                result = await s.call_tool(name, kwargs)
                                return result.content[0].text if result.content else str(result)
                    return lambda **kw: asyncio.get_event_loop().run_until_complete(execute(**kw))

                # Extract parameters from input_schema
                params = {}
                if mcp_tool.inputSchema:
                    params = {
                        "properties": mcp_tool.inputSchema.get("properties", {}),
                        "required": mcp_tool.inputSchema.get("required", [])
                    }

                tool = Tool(
                    name=mcp_tool.name,
                    description=mcp_tool.description or f"Tool: {mcp_tool.name}",
                    parameters=params,
                    function=make_executor(tool_name, url)
                )
                tools.append(tool)
                print(f"  Loaded: {tool.name} - {tool.description[:50]}...")

    return tools


async def load_tools_from_mcp_stdio(command: str, args: list[str]) -> list[Tool]:
    """Load tools from an MCP server via stdio."""
    tools = []

    server_params = StdioServerParameters(command=command, args=args)

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()

            for mcp_tool in result.tools:
                params = {}
                if mcp_tool.inputSchema:
                    params = {
                        "properties": mcp_tool.inputSchema.get("properties", {}),
                        "required": mcp_tool.inputSchema.get("required", [])
                    }

                # Create mock function (actual execution would need session)
                tool = Tool(
                    name=mcp_tool.name,
                    description=mcp_tool.description or f"Tool: {mcp_tool.name}",
                    parameters=params,
                    function=lambda **kw: {"status": "mock", "args": kw}
                )
                tools.append(tool)
                print(f"  Loaded: {tool.name} - {tool.description[:60]}...")

    return tools


# =============================================================================
# OPENAI ADVISOR (same as before)
# =============================================================================

class OpenAIToolSearchAdvisor:
    """Tool Search Tool pattern for OpenAI API."""

    def __init__(
        self,
        client: OpenAI,
        tool_searcher: ToolSearcher,
        model: str = "gpt-4o",
        max_search_results: int = 5,
        max_iterations: int = 10
    ):
        self.client = client
        self.tool_searcher = tool_searcher
        self.model = model
        self.max_search_results = max_search_results
        self.max_iterations = max_iterations
        self.all_tools: dict[str, Tool] = {}
        self.discovered_tools: dict[str, Tool] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

    def register_tools(self, tools: list[Tool]) -> None:
        self.all_tools = {t.name: t for t in tools}
        self.tool_searcher.index(tools)
        print(f"[Advisor] Registered {len(tools)} tools")

    def _tool_to_openai_schema(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters.get("properties", {}),
                    "required": tool.parameters.get("required", [])
                }
            }
        }

    def _get_search_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": "Search for available tools by describing what capability you need.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Description of capability needed"}
                    },
                    "required": ["query"]
                }
            }
        }

    def _get_current_tools(self) -> list[dict]:
        tools = [self._get_search_tool_schema()]
        for tool in self.discovered_tools.values():
            tools.append(self._tool_to_openai_schema(tool))
        return tools

    def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        if tool_name == "search_tools":
            query = tool_input.get("query", "")
            found_tools = self.tool_searcher.search(query, self.max_search_results)
            for tool in found_tools:
                if tool.name not in self.discovered_tools:
                    self.discovered_tools[tool.name] = tool
                    print(f"[Advisor] Discovered: {tool.name}")
            return json.dumps({
                "found_tools": [{"name": t.name, "description": t.description} for t in found_tools]
            })
        elif tool_name in self.discovered_tools:
            tool = self.discovered_tools[tool_name]
            try:
                result = tool.execute(**tool_input)
                print(f"[Advisor] Executed {tool_name}")
                return json.dumps({"result": result})
            except Exception as e:
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Tool '{tool_name}' not found"})

    def chat(self, user_message: str) -> str:
        self.discovered_tools = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Use search_tools to discover available tools."},
            {"role": "user", "content": user_message}
        ]

        for iteration in range(self.max_iterations):
            print(f"\n[Advisor] === Iteration {iteration + 1} ===")
            print(f"[Advisor] Tools: ['search_tools'] + {list(self.discovered_tools.keys())}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._get_current_tools(),
                tool_choice="auto"
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            self.request_count += 1

            print(f"[Advisor] Tokens: in={response.usage.prompt_tokens}, out={response.usage.completion_tokens}")

            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                print(f"\n[Advisor] === COMPLETE ===")
                return assistant_message.content or ""

            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                tool_input = json.loads(tool_call.function.arguments)
                result = self._handle_tool_call(tool_call.function.name, tool_input)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        return "Max iterations reached"


class OpenAIAllToolsAdvisor:
    """Traditional approach: send ALL tools upfront."""

    def __init__(self, client: OpenAI, model: str = "gpt-4o", max_iterations: int = 10):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.all_tools: dict[str, Tool] = {}
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

    def register_tools(self, tools: list[Tool]) -> None:
        self.all_tools = {t.name: t for t in tools}
        print(f"[AllTools] Registered {len(tools)} tools")

    def _tool_to_openai_schema(self, tool: Tool) -> dict:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": tool.parameters.get("properties", {}),
                    "required": tool.parameters.get("required", [])
                }
            }
        }

    def _get_all_tools(self) -> list[dict]:
        return [self._tool_to_openai_schema(t) for t in self.all_tools.values()]

    def _handle_tool_call(self, tool_name: str, tool_input: dict) -> str:
        if tool_name in self.all_tools:
            tool = self.all_tools[tool_name]
            try:
                result = tool.execute(**tool_input)
                print(f"[AllTools] Executed {tool_name}")
                return json.dumps({"result": result})
            except Exception as e:
                return json.dumps({"error": str(e)})
        return json.dumps({"error": f"Tool '{tool_name}' not found"})

    def chat(self, user_message: str) -> str:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to various tools."},
            {"role": "user", "content": user_message}
        ]

        for iteration in range(self.max_iterations):
            print(f"\n[AllTools] === Iteration {iteration + 1} ===")
            print(f"[AllTools] Sending {len(self.all_tools)} tools")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self._get_all_tools(),
                tool_choice="auto"
            )

            self.total_input_tokens += response.usage.prompt_tokens
            self.total_output_tokens += response.usage.completion_tokens
            self.request_count += 1

            print(f"[AllTools] Tokens: in={response.usage.prompt_tokens}, out={response.usage.completion_tokens}")

            assistant_message = response.choices[0].message

            if not assistant_message.tool_calls:
                print(f"\n[AllTools] === COMPLETE ===")
                return assistant_message.content or ""

            messages.append(assistant_message)
            for tool_call in assistant_message.tool_calls:
                tool_input = json.loads(tool_call.function.arguments)
                result = self._handle_tool_call(tool_call.function.name, tool_input)
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})

        return "Max iterations reached"


# =============================================================================
# MAIN TEST
# =============================================================================

async def main():
    client = OpenAI()

    # Try to load tools from public MCP servers
    print("=" * 70)
    print("LOADING TOOLS FROM MCP SERVERS")
    print("=" * 70)

    all_tools = []

    # Option 1: Use @modelcontextprotocol/server-fetch (stdio)
    print("\n[1] Loading from @modelcontextprotocol/server-fetch...")
    try:
        fetch_tools = await load_tools_from_mcp_stdio(
            "npx",
            ["-y", "@modelcontextprotocol/server-fetch"]
        )
        all_tools.extend(fetch_tools)
        print(f"    Loaded {len(fetch_tools)} tools from fetch server")
    except Exception as e:
        print(f"    Failed: {e}")

    # Option 2: Use @modelcontextprotocol/server-memory (stdio)
    print("\n[2] Loading from @modelcontextprotocol/server-memory...")
    try:
        memory_tools = await load_tools_from_mcp_stdio(
            "npx",
            ["-y", "@modelcontextprotocol/server-memory"]
        )
        all_tools.extend(memory_tools)
        print(f"    Loaded {len(memory_tools)} tools from memory server")
    except Exception as e:
        print(f"    Failed: {e}")

    # Option 3: Use @modelcontextprotocol/server-everything (stdio) - has many tools
    print("\n[3] Loading from @modelcontextprotocol/server-everything...")
    try:
        everything_tools = await load_tools_from_mcp_stdio(
            "npx",
            ["-y", "@modelcontextprotocol/server-everything"]
        )
        all_tools.extend(everything_tools)
        print(f"    Loaded {len(everything_tools)} tools from everything server")
    except Exception as e:
        print(f"    Failed: {e}")

    # Option 4: Local server if running
    print("\n[4] Trying local MCP server at http://localhost:8080/sse...")
    try:
        local_tools = await load_tools_from_mcp_sse("http://localhost:8080/sse")
        all_tools.extend(local_tools)
        print(f"    Loaded {len(local_tools)} tools from local server")
    except Exception as e:
        print(f"    Failed (server not running?): {e}")

    if not all_tools:
        print("\n[!] No MCP tools loaded. Using example tools instead.")
        from main import create_example_tools
        all_tools = create_example_tools()

    print(f"\n[TOTAL] Loaded {len(all_tools)} tools")
    print("-" * 70)
    for i, tool in enumerate(all_tools, 1):
        desc = tool.description[:50] + "..." if len(tool.description) > 50 else tool.description
        print(f"  {i}. {tool.name}: {desc}")

    # Test query - uses tools from the loaded MCP servers
    test_query = "Get the user name and then get their latest news"

    print("\n" + "=" * 70)
    print("TEST 1: WITH TOOL SEARCH (Dynamic Discovery)")
    print("=" * 70)

    searcher = VectorToolSearcher()
    
    advisor_with_search = OpenAIToolSearchAdvisor(
        client=client,
        tool_searcher=searcher,
        model="gpt-4o",
        max_search_results=5
    )
    advisor_with_search.register_tools(all_tools)

    response1 = advisor_with_search.chat(test_query)

    with_stats = {
        "input": advisor_with_search.total_input_tokens,
        "output": advisor_with_search.total_output_tokens,
        "requests": advisor_with_search.request_count,
        "total": advisor_with_search.total_input_tokens + advisor_with_search.total_output_tokens
    }

    print("\n" + "=" * 70)
    print("TEST 2: WITHOUT TOOL SEARCH (All Tools)")
    print("=" * 70)

    advisor_all = OpenAIAllToolsAdvisor(client=client, model="gpt-4o")
    advisor_all.register_tools(all_tools)

    response2 = advisor_all.chat(test_query)

    without_stats = {
        "input": advisor_all.total_input_tokens,
        "output": advisor_all.total_output_tokens,
        "requests": advisor_all.request_count,
        "total": advisor_all.total_input_tokens + advisor_all.total_output_tokens
    }

    # Summary
    print("\n" + "=" * 70)
    print(f"TOKEN COMPARISON ({len(all_tools)} MCP tools)")
    print("=" * 70)
    print(f"\n{'Metric':<20} {'With Search':<15} {'Without':<15} {'Savings':<15}")
    print("-" * 65)

    in_save = without_stats["input"] - with_stats["input"]
    in_pct = (in_save / without_stats["input"] * 100) if without_stats["input"] > 0 else 0

    out_diff = without_stats["output"] - with_stats["output"]
    out_pct = (out_diff / without_stats["output"] * 100) if without_stats["output"] > 0 else 0

    tot_save = without_stats["total"] - with_stats["total"]
    tot_pct = (tot_save / without_stats["total"] * 100) if without_stats["total"] > 0 else 0

    print(f"{'Input Tokens':<20} {with_stats['input']:<15} {without_stats['input']:<15} {in_save} ({in_pct:.1f}%)")
    print(f"{'Output Tokens':<20} {with_stats['output']:<15} {without_stats['output']:<15} {out_diff} ({out_pct:.1f}%)")
    print(f"{'Total Tokens':<20} {with_stats['total']:<15} {without_stats['total']:<15} {tot_save} ({tot_pct:.1f}%)")
    print(f"{'API Requests':<20} {with_stats['requests']:<15} {without_stats['requests']:<15}")

    print("\n" + "=" * 70)
    print("RESPONSES")
    print("=" * 70)
    print("\n[WITH SEARCH]:")
    print(response1[:400] + "..." if len(response1) > 400 else response1)
    print("\n[WITHOUT SEARCH]:")
    print(response2[:400] + "..." if len(response2) > 400 else response2)


if __name__ == "__main__":
    asyncio.run(main())
