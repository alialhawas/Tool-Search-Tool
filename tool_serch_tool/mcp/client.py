# import os
# import asyncio
# from dotenv import load_dotenv
# from mcp import ClientSession, StdioServerParameters
# from mcp.client.stdio import stdio_client
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI

# # Disable LangSmith caching
# os.environ["LANGCHAIN_TRACING_V2"] = "false"

# load_dotenv()

# model = ChatOpenAI(
#     base_url="http://localhost:8000/v1",
#     model="Qwen/Qwen2.5-1.5B-Instruct",
#     api_key="not-needed",
#     temperature=0,
# )

# server_params = StdioServerParameters(
#     command="python",
#     args=["server.py"],
# )

# async def run_agent():
#     async with stdio_client(server_params) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()

#             tools = await load_mcp_tools(session)
            
#             print(f"Loaded {len(tools)} tools:")
#             for tool in tools:
#                 print(f"  - {tool.name}: {tool.description}")
#             breakpoint()


#             agent = create_react_agent(model, tools)
            
#             agent_response = await agent.ainvoke({
#                 "messages": [("user", "what is the client name and their latest news")]
#             })
#             return agent_response

# if __name__ == "__main__":
#     result = asyncio.run(run_agent())
#     print("\n--- Agent Response ---")
#     print(result)
#     print(result["messages"][-1].content)
#     with open ('res.txt', 'w') as file:
#         file.write(result["messages"][-1].content)



# import os
# import asyncio
# from dotenv import load_dotenv
# from mcp import ClientSession
# from mcp.client.sse import sse_client  # Changed from stdio_client
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI

# # Disable LangSmith caching
# os.environ["LANGCHAIN_TRACING_V2"] = "false"

# load_dotenv()

# # model = ChatOpenAI(
# #     base_url="http://localhost:8000/v1",
# #     model="Qwen/Qwen2.5-1.5B-Instruct",
# #     api_key="not-needed",
# #     temperature=0,
# # )

# model = ChatOpenAI(
#     model="gpt-4o-mini",  # or "gpt-4o" for best results
#     temperature=0,
# )

# MCP_SERVER_URL = "http://localhost:8080/sse"  


# async def run_agent():
#     async with sse_client(MCP_SERVER_URL) as (read, write):
#         async with ClientSession(read, write) as session:
#             await session.initialize()

#             tools = await load_mcp_tools(session)
            
#             print(f"Loaded {len(tools)} tools:")
#             for tool in tools:
#                 print(f"  - {tool.name}: {tool.description}")

#             agent = create_react_agent(model, tools)
            
#             agent_response = await agent.ainvoke({
#                 "messages": [("user", "Get the user name and then get their latest news")]
#             })
#             return agent_response


# if __name__ == "__main__":
#     result = asyncio.run(run_agent())
#     print("\n--- Agent Response ---")
#     # print(result)
#     print(result["messages"][-1].content)
#     with open ('res.txt', 'w') as file:
#         file.write(result["messages"][-1].content)




from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import tiktoken

from dotenv import load_dotenv

load_dotenv()


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def count_tool_tokens(tools: list, model: str = "gpt-4o") -> int:
    """Estimate tokens for tool definitions"""
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for tool in tools:
        # Convert tool schema to string representation
        breakpoint()
        tool_str = str(tool.args_schema if hasattr(tool, 'args_schema') else tool)
        tool_str += tool.name + (tool.description or "")
        total += len(encoding.encode(tool_str))
    return total


async def main():
    client = MultiServerMCPClient(
        {
            "user": {
                "url": "http://localhost:8080/sse",
                "transport": "sse"
            },
            "postgres": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://user:pass@localhost:5432/mcp_db"],
                "transport": "stdio"
            },
        }
    )

    llm = ChatOpenAI(model="gpt-4o")
    
    # --- BEFORE TOOLS ---
    base_message = "What's the latest news about the user? and see if it match user_news table with the news"
    tokens_before = count_tokens(base_message)
    print(f"Tokens BEFORE tools: {tokens_before}")
    
    # --- GET TOOLS ---
    tools = await client.get_tools()
    num_tools = len(tools)
    print(f"Number of tools: {num_tools}")
    print("Tool names:")
    for i, tool in enumerate(tools, 1):
        print(f"  {i}. {tool.name}")
    
    # --- AFTER TOOLS ---
    tool_tokens = count_tool_tokens(tools)
    tokens_after = tokens_before + tool_tokens
    print(f"\nTokens for tool definitions: {tool_tokens}")
    print(f"Tokens AFTER tools (estimated): {tokens_after}")
    print(f"Token increase: {tool_tokens} ({(tool_tokens/tokens_before)*100:.1f}% of base)")
    
    # --- More accurate: Use LLM's token counting ---
    agent = create_react_agent(llm, tools)
    
    # Get actual token count from a dry run
    messages = [{"role": "user", "content": base_message}]
    
    # Bind tools to see the actual prompt
    llm_with_tools = llm.bind_tools(tools)
    
    # Get token usage from API (requires actual call)
    # Or inspect the formatted prompt:
    print("\n--- Tool Schemas ---")
    for tool in tools:
        print(f"\n{tool.name}:")
        if hasattr(tool, 'args_schema'):
            print(f"  Schema: {tool.args_schema}")
    
    result = await agent.ainvoke({"messages": messages})
    print("\n" + result["messages"][-1].content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())