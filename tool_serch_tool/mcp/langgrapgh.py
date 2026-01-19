"""
LangGraph Agent with Local Model (Ollama) + MCP Tools
Shows token count before and after connecting to MCP server
"""

import os
import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

# Disable LangSmith caching
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()


# =============================================================================
# Token Counting Utilities
# =============================================================================

def count_tokens_approx(text: str) -> int:
    """Approximate token count (~4 chars per token for English)"""
    return len(text) // 4


def get_tool_schema(tool) -> dict:
    """Safely extract tool schema regardless of tool type"""
    schema = {
        "name": getattr(tool, 'name', str(tool)),
        "description": getattr(tool, 'description', ''),
    }
    
    # Try different ways to get the parameters schema
    if hasattr(tool, 'args_schema') and tool.args_schema:
        if hasattr(tool.args_schema, 'model_json_schema'):
            # Pydantic v2 model
            schema["parameters"] = tool.args_schema.model_json_schema()
        elif hasattr(tool.args_schema, 'schema'):
            # Pydantic v1 model
            schema["parameters"] = tool.args_schema.schema()
        elif isinstance(tool.args_schema, dict):
            # Already a dict
            schema["parameters"] = tool.args_schema
        else:
            schema["parameters"] = {}
    elif hasattr(tool, 'args') and tool.args:

        schema["parameters"] = {"properties": tool.args, "type": "object"}
    elif hasattr(tool, 'input_schema') and tool.input_schema:
        schema["parameters"] = tool.input_schema
    else:
        schema["parameters"] = {}
    
    return schema


def get_context_size(system_prompt: str, messages: list, tools: list = None) -> dict:
    """Calculate the context size in tokens"""
    
    # System prompt tokens
    system_tokens = count_tokens_approx(system_prompt)
    
    # Messages tokens
    messages_text = json.dumps([{"role": "user", "content": m} if isinstance(m, str) else str(m) for m in messages])
    messages_tokens = count_tokens_approx(messages_text)
    
    # Tools tokens
    tools_tokens = 0
    tools_json = ""
    if tools:
        tools_schema = []
        for tool in tools:
            schema = get_tool_schema(tool)
            tools_schema.append(schema)
        tools_json = json.dumps(tools_schema, indent=2)
        tools_tokens = count_tokens_approx(tools_json)
    
    total_tokens = system_tokens + messages_tokens + tools_tokens
    
    return {
        "system_prompt_tokens": system_tokens,
        "messages_tokens": messages_tokens,
        "tools_tokens": tools_tokens,
        "total_tokens": total_tokens,
        "tools_json": tools_json if tools else None,
        "tools_count": len(tools) if tools else 0
    }


def print_context_comparison(before: dict, after: dict):
    """Print a nice comparison of context sizes"""
    
    token_increase = after['total_tokens'] - before['total_tokens']
    pct_increase = ((token_increase) / before['total_tokens'] * 100) if before['total_tokens'] > 0 else 0
    
    print("\n" + "=" * 70)
    print("📊 CONTEXT LENGTH COMPARISON")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                     LangGraph Context Comparison                    │
├─────────────────────────────────────────────────────────────────────┤
│  BEFORE (no MCP tools)                                              │
│    • System prompt:     ~{before['system_prompt_tokens']:>5} tokens                                │
│    • Messages:          ~{before['messages_tokens']:>5} tokens                               │
│    • Tools:             ~{before['tools_tokens']:>5} tokens ({before['tools_count']} tools)                       │
│    • TOTAL:             ~{before['total_tokens']:>5} tokens                               │
├─────────────────────────────────────────────────────────────────────┤
│  AFTER (with MCP tools)                                             │
│    • System prompt:     ~{after['system_prompt_tokens']:>5} tokens                              │
│    • Messages:          ~{after['messages_tokens']:>5} tokens                               │
│    • Tools:             ~{after['tools_tokens']:>5} tokens ({after['tools_count']} tools)                       │
│    • TOTAL:             ~{after['total_tokens']:>5} tokens                             │
├─────────────────────────────────────────────────────────────────────┤
│  IMPACT                                                             │
│    • Token increase:    +{token_increase:>5} tokens                              │
│    • Percentage:        +{pct_increase:>5.1f}%                                 │
└─────────────────────────────────────────────────────────────────────┘
""")
    
    if after['tools_json']:
        print("\n📋 MCP Tools Schema (what gets added to context):")
        print("-" * 70)
        print(after['tools_json'])


# =============================================================================
# LangGraph Agent Setup
# =============================================================================

def create_langgraph_agent(model, tools):
    """Create a LangGraph ReAct-style agent"""
    
    # Bind tools to the model
    model_with_tools = model.bind_tools(tools)
    
    # Define the function that calls the model
    def call_model(state: MessagesState):
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Define the function that determines whether to continue or end
    def should_continue(state: MessagesState):
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        # Otherwise, end
        return END
    
    # Create the graph
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    
    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    
    # Compile the graph
    return workflow.compile()


# =============================================================================
# Main
# =============================================================================

async def main():
    print("=" * 70)
    print("🦜 LangGraph Agent with Local Model + MCP Tools")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Setup Local Model (Ollama)
    # -------------------------------------------------------------------------
    print("\n🤖 Setting up local model (Ollama)...")
    
    # You can change this to any Ollama model you have installed
    # Popular options: llama3.2, mistral, qwen2.5, deepseek-r1
    MODEL_NAME = "qwen2.5:7b"  # Change this to your preferred model
    
    try:
        model = ChatOpenAI(
            model="gpt-4o-mini",  # or "gpt-4o" for best results
            temperature=0,
        )

        print(f"   ✅ Using model: {MODEL_NAME}")
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print("   Make sure Ollama is running: ollama serve")
        print(f"   And the model is installed: ollama pull {MODEL_NAME}")
        return
    
    # -------------------------------------------------------------------------
    # System prompt and user message
    # -------------------------------------------------------------------------
    system_prompt = """You are a helpful assistant with access to tools.
Use the available tools to help answer user questions.
Always use tools when they can help provide accurate information."""

    user_message = "Get the user name and then get their latest news"
    
    # -------------------------------------------------------------------------
    # BEFORE: Calculate context WITHOUT MCP tools
    # -------------------------------------------------------------------------
    print("\n📊 Calculating context size BEFORE MCP connection...")
    
    context_before = get_context_size(
        system_prompt=system_prompt,
        messages=[user_message],
        tools=[]
    )
    
    print(f"   • System prompt: ~{context_before['system_prompt_tokens']} tokens")
    print(f"   • Messages: ~{context_before['messages_tokens']} tokens")
    print(f"   • Tools: ~{context_before['tools_tokens']} tokens (0 tools)")
    print(f"   • TOTAL: ~{context_before['total_tokens']} tokens")
    
    # -------------------------------------------------------------------------
    # Connect to MCP Server
    # -------------------------------------------------------------------------
    print("\n🔌 Connecting to MCP Server...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Load MCP tools
            tools = await load_mcp_tools(session)
            
            print(f"   ✅ Connected! Loaded {len(tools)} tools:")
            for tool in tools:
                tool_name = getattr(tool, 'name', str(tool))
                tool_desc = getattr(tool, 'description', '')[:50]
                print(f"      - {tool_name}: {tool_desc}...")
            
            # -----------------------------------------------------------------
            # AFTER: Calculate context WITH MCP tools
            # -----------------------------------------------------------------
            print("\n📊 Calculating context size AFTER MCP connection...")
            
            context_after = get_context_size(
                system_prompt=system_prompt,
                messages=[user_message],
                tools=tools
            )
            
            print(f"   • System prompt: ~{context_after['system_prompt_tokens']} tokens")
            print(f"   • Messages: ~{context_after['messages_tokens']} tokens")
            print(f"   • Tools: ~{context_after['tools_tokens']} tokens ({context_after['tools_count']} tools)")
            print(f"   • TOTAL: ~{context_after['total_tokens']} tokens")
            
            # -----------------------------------------------------------------
            # Print Comparison
            # -----------------------------------------------------------------
            print_context_comparison(context_before, context_after)
            
            # -----------------------------------------------------------------
            # Run the Agent
            # -----------------------------------------------------------------
            print("\n" + "=" * 70)
            print("🚀 Running LangGraph Agent...")
            print("=" * 70)
            
            # Create the LangGraph agent
            agent = create_langgraph_agent(model, tools)
            
            # Prepare messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            print(f"\n📝 User Query: {user_message}\n")
            print("-" * 70)
            
            # Run the agent
            result = await agent.ainvoke({"messages": messages})
            
            # -----------------------------------------------------------------
            # Display Results
            # -----------------------------------------------------------------
            print("\n" + "-" * 70)
            print("📜 Agent Execution Trace:")
            print("-" * 70)
            
            for i, msg in enumerate(result["messages"]):
                if isinstance(msg, SystemMessage):
                    print(f"\n[{i}] 🔧 SYSTEM: {msg.content[:100]}...")
                elif isinstance(msg, HumanMessage):
                    print(f"\n[{i}] 👤 USER: {msg.content}")
                elif isinstance(msg, AIMessage):
                    if hasattr(msg, 'tool_calls') and msg.tool_calls:
                        print(f"\n[{i}] 🤖 AI: (calling tools)")
                        for tc in msg.tool_calls:
                            print(f"      → {tc['name']}({tc['args']})")
                    else:
                        print(f"\n[{i}] 🤖 AI: {msg.content}")
                else:
                    # ToolMessage
                    tool_name = getattr(msg, 'name', 'unknown')
                    content = str(msg.content)[:100] if msg.content else ''
                    print(f"\n[{i}] 🔨 TOOL [{tool_name}]: {content}...")
            
            # Final answer
            final_message = result["messages"][-1]
            print("\n" + "=" * 70)
            print("✅ FINAL ANSWER:")
            print("=" * 70)
            print(f"\n{final_message.content}\n")
            
            # -----------------------------------------------------------------
            # Save Report
            # -----------------------------------------------------------------
            output_dir = Path.cwd() / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Safely parse tools_json
            tools_schema_list = []
            if context_after['tools_json']:
                try:
                    tools_schema_list = json.loads(context_after['tools_json'])
                except:
                    tools_schema_list = []
            
            report = {
                "model": MODEL_NAME,
                "framework": "LangGraph",
                "context_before": {
                    "system_prompt_tokens": context_before['system_prompt_tokens'],
                    "messages_tokens": context_before['messages_tokens'],
                    "tools_tokens": context_before['tools_tokens'],
                    "tools_count": context_before['tools_count'],
                    "total_tokens": context_before['total_tokens'],
                },
                "context_after": {
                    "system_prompt_tokens": context_after['system_prompt_tokens'],
                    "messages_tokens": context_after['messages_tokens'],
                    "tools_tokens": context_after['tools_tokens'],
                    "tools_count": context_after['tools_count'],
                    "total_tokens": context_after['total_tokens'],
                    "tools_schema": tools_schema_list
                },
                "impact": {
                    "token_increase": context_after['total_tokens'] - context_before['total_tokens'],
                    "percentage_increase": round(((context_after['total_tokens'] - context_before['total_tokens']) / context_before['total_tokens'] * 100) if context_before['total_tokens'] > 0 else 0, 2)
                },
                "final_answer": final_message.content
            }
            
            report_path = output_dir / "langgraph_mcp_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            
            print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())