# from mcp.server.fastmcp import FastMCP

# mcp = FastMCP("user")

# @mcp.tool()
# def get_client_name(name: str) -> dict:
#     """Get the client name and their ID. Returns a dict with 'name' and 'id' keys."""
#     return {"id": 1}

# @mcp.tool()
# def get_latest_news(id: int) -> str:
#     """Get the latest news about a client. Requires the client's ID from get_client_name."""
#     return f"Latest news: ahmed is trying again (client_id: {id})"

# if __name__ == "__main__":
#     mcp.run(transport="stdio")


import uvicorn
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("user")


@mcp.tool()
def get_user_name() -> dict:
    """Get the user name and their ID. Returns a dict with 'name' and 'id' keys."""
    return {"name": "ali", "id": 1}


@mcp.tool()
def get_latest_news(id: int) -> str:
    """Get the latest news about a user. Requires the user's ID from get_user_name."""
    return f"Latest news: ali is trying again (user_id: {id})"


if __name__ == "__main__":

    app = mcp.sse_app()
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )