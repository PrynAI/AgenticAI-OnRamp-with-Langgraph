from graph import app, LAST
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    print("Hello ReAct LangGraph with function calling")

    # This sample prompt exercises both available tools: Tavily for live search
    # and the custom triple tool for the arithmetic step.
    res = app.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What is the temperature in Guntur? List it and then triple it"
                )
            ]
        }
    )
    print(res["messages"][LAST].content)
