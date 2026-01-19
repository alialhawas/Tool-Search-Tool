"""
Test script comparing token usage:
1. WITH tool search tool (dynamic discovery)
2. WITHOUT tool search tool (all tools sent upfront)
"""

import json
import os
from dataclasses import dataclass
from typing import Callable, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Reuse Tool and searcher from main
from main import Tool, VectorToolSearcher, ToolSearcher, create_example_tools


# =============================================================================
# OPENAI-COMPATIBLE ADVISOR (WITH TOOL SEARCH)
# =============================================================================

class OpenAIToolSearchAdvisor:
    """
    Tool Search Tool pattern for OpenAI API.
    Sends only search_tools initially, discovers tools on demand.
    """

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

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.request_count = 0

    def register_tools(self, tools: list[Tool]) -> None:
        self.all_tools = {t.name: t for t in tools}
        self.tool_searcher.index(tools)
        print(f"[Advisor] Registered {len(tools)} tools")

    def _tool_to_openai_schema(self, tool: Tool) -> dict:
        """Convert Tool to OpenAI function format."""
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
        """The meta-tool for searching."""
        return {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": (
                    "Search for available tools by describing what capability you need. "
                    "Use this to discover tools before calling them."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Natural language description of the capability"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

    def _get_current_tools(self) -> list[dict]:
        """Get tools: search_tool + discovered tools."""
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
                "found_tools": [
                    {"name": t.name, "description": t.description}
                    for t in found_tools
                ]
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
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to various tools. "
                    "Use the search_tools function to discover what tools are available "
                    "before trying to use them."
                )
            },
            {"role": "user", "content": user_message}
        ]

        for iteration in range(self.max_iterations):
            print(f"\n[Advisor] === Iteration {iteration + 1} ===")
            print(f"[Advisor] Available: ['search_tools'] + {list(self.discovered_tools.keys())}")

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
            breakpoint()

            # Check if done
            if not assistant_message.tool_calls:
                print(f"\n[Advisor] === COMPLETE ===")
                print(f"[Advisor] Total: {self.request_count} requests, "
                      f"{self.total_input_tokens} input, {self.total_output_tokens} output")
                return assistant_message.content or ""

            # Process tool calls
            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                breakpoint()

                tool_input = json.loads(tool_call.function.arguments)
                result = self._handle_tool_call(tool_call.function.name, tool_input)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "Max iterations reached"


# =============================================================================
# OPENAI-COMPATIBLE ADVISOR (WITHOUT TOOL SEARCH - ALL TOOLS)
# =============================================================================

class OpenAIAllToolsAdvisor:
    """
    Traditional approach: send ALL tools upfront.
    No dynamic discovery.
    """

    def __init__(
        self,
        client: OpenAI,
        model: str = "gpt-4o",
        max_iterations: int = 10
    ):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations

        self.all_tools: dict[str, Tool] = {}

        # Token tracking
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
        """Return ALL tools."""
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
            {
                "role": "system",
                "content": "You are a helpful assistant with access to various tools."
            },
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
                print(f"[AllTools] Total: {self.request_count} requests, "
                      f"{self.total_input_tokens} input, {self.total_output_tokens} output")
                return assistant_message.content or ""

            messages.append(assistant_message)

            for tool_call in assistant_message.tool_calls:
                tool_input = json.loads(tool_call.function.arguments)
                result = self._handle_tool_call(tool_call.function.name, tool_input)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

        return "Max iterations reached"


# =============================================================================
# MAIN TEST
# =============================================================================

def main():
    client = OpenAI()
    tools = create_example_tools()

    test_query = (
        "Help me plan what to wear today."
        # "Please suggest clothing shops that are open right now."
    )

    print("=" * 70)
    print("TEST 1: WITH TOOL SEARCH TOOL (Dynamic Discovery)")
    print("=" * 70)

    searcher = VectorToolSearcher()
    
    advisor_with_search = OpenAIToolSearchAdvisor(
        client=client,
        tool_searcher=searcher,
        model="gpt-4o",
        max_search_results=5
    )
    advisor_with_search.register_tools(tools)

    response1 = advisor_with_search.chat(test_query)

    with_search_stats = {
        "input_tokens": advisor_with_search.total_input_tokens,
        "output_tokens": advisor_with_search.total_output_tokens,
        "requests": advisor_with_search.request_count,
        "total_tokens": advisor_with_search.total_input_tokens + advisor_with_search.total_output_tokens
    }

    print("\n" + "=" * 70)
    print("TEST 2: WITHOUT TOOL SEARCH TOOL (All Tools Upfront)")
    print("=" * 70)

    advisor_all_tools = OpenAIAllToolsAdvisor(
        client=client,
        model="gpt-4o"
    )
    advisor_all_tools.register_tools(tools)

    response2 = advisor_all_tools.chat(test_query)

    without_search_stats = {
        "input_tokens": advisor_all_tools.total_input_tokens,
        "output_tokens": advisor_all_tools.total_output_tokens,
        "requests": advisor_all_tools.request_count,
        "total_tokens": advisor_all_tools.total_input_tokens + advisor_all_tools.total_output_tokens
    }

    # Summary
    print("\n" + "=" * 70)
    print("TOKEN USAGE COMPARISON")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'With Search':<15} {'Without Search':<15} {'Savings':<15}")
    print("-" * 70)

    input_savings = without_search_stats["input_tokens"] - with_search_stats["input_tokens"]
    input_pct = (input_savings / without_search_stats["input_tokens"] * 100) if without_search_stats["input_tokens"] > 0 else 0

    output_diff = without_search_stats["output_tokens"] - with_search_stats["output_tokens"]
    output_pct = (output_diff / without_search_stats["output_tokens"] * 100) if without_search_stats["output_tokens"] > 0 else 0

    total_savings = without_search_stats["total_tokens"] - with_search_stats["total_tokens"]
    total_pct = (total_savings / without_search_stats["total_tokens"] * 100) if without_search_stats["total_tokens"] > 0 else 0

    print(f"{'Input Tokens':<25} {with_search_stats['input_tokens']:<15} {without_search_stats['input_tokens']:<15} {input_savings} ({input_pct:.1f}%)")
    print(f"{'Output Tokens':<25} {with_search_stats['output_tokens']:<15} {without_search_stats['output_tokens']:<15} {output_diff} ({output_pct:.1f}%)")
    print(f"{'Total Tokens':<25} {with_search_stats['total_tokens']:<15} {without_search_stats['total_tokens']:<15} {total_savings} ({total_pct:.1f}%)")
    print(f"{'API Requests':<25} {with_search_stats['requests']:<15} {without_search_stats['requests']:<15}")

    print("\n" + "=" * 70)
    print("RESPONSES")
    print("=" * 70)
    print("\n[WITH TOOL SEARCH]:")
    print(response1[:500] + "..." if len(response1) > 500 else response1)
    print("\n[WITHOUT TOOL SEARCH]:")
    print(response2[:500] + "..." if len(response2) > 500 else response2)


if __name__ == "__main__":
    main()
