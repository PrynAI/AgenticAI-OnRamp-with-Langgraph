import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

load_dotenv()

llm = ChatOpenAI(model="gpt-5-nano", temperature=0)

stdio_server_params = StdioServerParameters(
    command="python",
    args=[
        "<fill with your file path>\\AgenticAI-OnRamp-with-Langgraph\\servers\\math_server.py"
    ],
)


async def stdioclient():
    async with stdio_client(stdio_server_params) as (read, write):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            print("session initialized")
            print("==" * 60)
            tools = await load_mcp_tools(session)
            print(tools)
            agent = create_agent(llm, tools)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="what is the value of 8*5+5*3?")]}
            )
            print(result["messages"][-1].content)


weather_url = "http://127.0.0.1:8000/mcp"


async def streamablehttpclient():
    async with streamable_http_client(weather_url) as (read, write, get_session_id):
        async with ClientSession(read_stream=read, write_stream=write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_agent(llm, tools)
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="what is the weather in Guntur?")]}
            )
            print(result["messages"][-1].content)


if __name__ == "__main__":
    # asyncio.run(stdioclient())
    asyncio.run(streamablehttpclient())
