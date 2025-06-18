from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

@mcp.tool()
def get_weather(city: str) -> dict:
    """
    Get current weather for a city (mocked).
    Args:
        city: The name of the city
    Returns:
        dict: Weather information
    """
    return {
        "city": city,
        "temperature": "22°C",
        "condition": "Partly Cloudy",
        "mock": True
    }

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
