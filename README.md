# DeepAgents

A small Python project for experimenting with LangChain agents and
DeepAgents. The current example wires a mock weather tool into an agent,
invokes it with a sample user message, and prints the final response.

This branch focuses on comparing a deeper agent workflow with a simpler
LangChain `create_agent` ReAct-style workflow.

## What This Project Shows

- Loading local environment variables with `python-dotenv`
- Defining a simple callable tool for agent use
- Creating an agent with LangChain's `create_agent`
- Showing how to switch the same tool into `deepagents.create_deep_agent`
- Comparing the practical tradeoff between deeper agent orchestration and a
  shallow agent for a simple task

## Tech Stack

- Python 3.13
- LangChain
- DeepAgents
- LangChain OpenAI integration
- uv for dependency management
- Ruff and Black for development formatting/linting

## Project Structure

```text
.
|-- main.py             # Agent setup, mock weather tool, and sample invocation
|-- pyproject.toml      # Project metadata and dependencies
|-- uv.lock             # Locked dependency versions
|-- .python-version     # Python version used by the project
`-- README.md           # Project documentation
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.13 installed
- uv installed
- An OpenAI API key with access to the model configured in `main.py`

## Setup

Install the project dependencies:

```powershell
uv sync
```

Create a local `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Do not commit `.env` files or API keys.

## Running the Example

Run the main script:

```powershell
uv run python main.py
```

The script sends this sample user message to the agent:

```text
what is the weather in Guntur
```

The weather function is intentionally mocked, so it returns a fixed response
instead of calling a real weather API.

## Switching Between Agent Implementations

`main.py` currently keeps the DeepAgents implementation as a commented block
and runs the simpler LangChain agent by default.

DeepAgents example:

```python
agent = create_deep_agent(
    model="openai:gpt-5.4-nano",
    tools=[get_weather],
    system_prompt="you are expert assistant providing weather information",
)
```

LangChain `create_agent` example:

```python
agent = create_agent(
    model="openai:gpt-5.4-nano",
    tools=[get_weather],
    system_prompt="you are expert assistant providing weather information",
)
```

To test DeepAgents, uncomment the `create_deep_agent` block and comment out
the `create_agent` block.

## Current Observation

For this simple weather-tool task, the shallow LangChain agent is more suitable
because it uses fewer tokens and completes faster. The comparison note in
`main.py` records one local run where DeepAgents consumed significantly more
tokens than `create_agent`.

Use DeepAgents when the task benefits from deeper planning, longer-running
workflows, or multi-step tool use. For simple one-tool requests, a shallow agent
is usually enough.

## Development Commands

Format code:

```powershell
uv run black .
```

Lint code:

```powershell
uv run ruff check .
```

## Notes

- The `get_weather` tool is a demo stub, not a production weather integration.
- Model name, provider, and API key configuration are controlled by `main.py`
  and environment variables.
- Token usage, latency, and cost depend on the selected model, provider
  settings, prompt, and runtime environment.
