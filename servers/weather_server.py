from mcp.server.fastmcp import FastMCP

mcp = FastMCP()


@mcp.tool()
async def check_weather(location: str) -> str:
    """check the current weather for the specified {location}"""
    return f"The current weather for the {location} is SuperHot"


if __name__ == "__main__":
    mcp.run("streamable-http")
