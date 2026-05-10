from deepagents import create_deep_agent
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv()


def get_weather(location: str) -> str:
    """Get weather in degree centigrade for the {location}

    Args:
    Location:str:Location to fetech weather information


    """
    return f"Current weather in the {location} is:SuperHot "


# agent=create_deep_agent(model="openai:gpt-5.4-nano",
#                         tools=[get_weather],
#                         system_prompt="you are expert assistant providing weather information")

agent = create_agent(
    model="openai:gpt-5.4-nano",
    tools=[get_weather],
    system_prompt="you are expert assistant providing weather information",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Guntur"}]}
)


## Conclusion : for simple tasks just shallow agent create_agent(ReAct) is best for consuming less tokens and speed not necessarily you need Deep agent that calls unnecessary chains in the process of bringing output

# Comparision : Deep agent call took : 10.5k tokens, speed: 7365sec with the cose of 0.0005$. create_agent took : 395 tokens , speed : 4.56sec and cost : 0.0002$

if __name__ == "__main__":
    print(result["messages"][-1])
