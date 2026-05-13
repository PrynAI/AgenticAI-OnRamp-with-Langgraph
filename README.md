# Agentic AI On-Ramp with LangGraph

This repository is a branch-based learning collection for practical Agentic AI
and LLM engineering patterns. The `main` branch is the public landing page. Each
implementation branch is a focused module with its own runnable code,
documentation, dependencies, and examples.

Use this README to decide which branch to explore first. Then check out the
branch that matches the concept you want to learn.

```bash
git clone https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph.git
cd AgenticAI-OnRamp-with-Langgraph
git checkout <branch-name>
```

Most branches use Python `3.13`, `uv`, LangGraph or LangChain, and environment
variables loaded from a local `.env` file. Each branch README contains the exact
setup and run instructions for that module.

## Learning Modules

Each branch demonstrates a different Agentic AI or LLM application pattern:

- Tool-using agents with ReAct-style reasoning loops
- Agentic RAG with routing, grading, web search, and retry control
- Reflection and Reflexion patterns for self-improving model outputs
- Human-in-the-loop workflows with persisted graph state
- Model Context Protocol tool servers and LangChain MCP clients
- DeepAgents versus shallow agent orchestration tradeoffs

## Branch Comparison

| Branch | Main Pattern | Core Problem Solved | Best For |
| --- | --- | --- | --- |
| [`ReAct-Agent-Function-Calling`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/ReAct-Agent-Function-Calling) | ReAct agent with tool calling | Decide when to call tools, execute them, and continue reasoning | Learning tool loops and OpenAI function calling with LangGraph |
| [`agenticRAG`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/agenticRAG) | Adaptive / Corrective / Self-RAG | Route questions, retrieve context, search the web, grade answers, and retry | Building more reliable RAG systems |
| [`deepagents`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/deepagents) | DeepAgents comparison | Compare deeper agent orchestration with a simpler `create_agent` workflow | Understanding when deeper agent planning is useful |
| [`humaninthelooppersistancememory`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/humaninthelooppersistancememory) | Human-in-the-loop and persistence | Pause graph execution, save state, collect feedback, and resume | Approval flows, checkpoints, memory, and branching examples |
| [`mcp`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/mcp) | Model Context Protocol tools | Expose Python functions as MCP tools and call them from an agent | Learning MCP server/client integration |
| [`reflection-agent`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/reflection-agent) | Reflection loop | Generate content, critique it, and revise over multiple iterations | Content improvement and critique-driven generation |
| [`reflexion-agent`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/reflexion-agent) | Reflexion research agent | Draft, self-critique, search, revise, and return cited answers | Search-backed answer refinement and structured tool outputs |

## Repository Map

### ReAct-Agent-Function-Calling

**Branch:** [`ReAct-Agent-Function-Calling`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/ReAct-Agent-Function-Calling)

**High-level overview:**  
This branch implements a compact ReAct-style agent using LangGraph,
LangChain, OpenAI tool calling, Tavily search, and a custom Python tool. The
agent reasons over a user request, decides whether a tool is needed, executes
the tool, and loops until it can produce a final answer.

**Key features implemented:**

- LangGraph `StateGraph` using `MessagesState`
- Reasoning node with tools bound to an OpenAI chat model
- LangGraph `ToolNode` for tool execution
- Conditional routing based on whether the latest model response contains tool calls
- Custom `triple` tool for numeric computation
- Tavily search integration for live web-backed information
- Generated graph diagram stored as `flow.png`

**Technologies/frameworks used:**

- Python, `uv`, LangGraph, LangChain, LangChain OpenAI, OpenAI tool calling,
  LangChain Tavily, `python-dotenv`

**Learning outcomes:**

- How ReAct-style loops work in LangGraph
- How to bind tools to an LLM and let the model request tool calls
- How to route between reasoning and acting nodes
- How to combine live search with deterministic Python tools

**Suggested use cases:**

- Tool-calling agent prototypes
- Search plus calculation workflows
- Beginner-friendly LangGraph control-flow examples
- Demonstrations of OpenAI function calling inside an agent loop

### agenticRAG

**Branch:** [`agenticRAG`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/agenticRAG)

**High-level overview:**  
This branch implements an agentic Retrieval-Augmented Generation workflow using
LangGraph. It routes questions to either a Weaviate vector store or Tavily web
search, grades retrieved context, generates an answer, checks grounding and
usefulness, and retries through explicit graph control flow.

**Key features implemented:**

- LLM-powered router for vector store versus web search
- Weaviate ingestion and retrieval pipeline
- Document relevance grading before generation
- Tavily web search fallback and query rewriting
- RAG generation with a LangSmith Hub prompt
- Hallucination grading against retrieved documents
- Answer usefulness grading against the original question
- Bounded retry logic for generation and web search
- LangGraph CLI configuration for LangSmith Studio testing

**Technologies/frameworks used:**

- Python, `uv`, LangGraph, LangChain, LangChain OpenAI, OpenAI embeddings,
  Weaviate, Tavily, LangSmith Hub, LangGraph CLI, Pytest, `python-dotenv`

**Learning outcomes:**

- How adaptive RAG differs from a basic retrieve-and-generate pipeline
- How to use structured LLM outputs for routing and grading
- How to combine vector search with web search repair
- How to add grounding checks and answer-quality gates
- How to model RAG as an inspectable state machine

**Suggested use cases:**

- Advanced RAG learning projects
- Knowledge-base assistants with web fallback
- RAG quality-control experiments
- LangGraph Studio demonstrations
- Prototypes for grounded answer generation

### deepagents

**Branch:** [`deepagents`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/deepagents)

**High-level overview:**  
This branch compares a deeper agent workflow with a simpler LangChain
`create_agent` ReAct-style workflow. The example uses a mock weather tool and
shows that shallow agent orchestration can be more appropriate for simple
single-tool tasks.

**Key features implemented:**

- Mock weather tool for agent use
- LangChain `create_agent` implementation
- Commented `create_deep_agent` implementation for comparison
- Simple side-by-side evaluation notes around token use, speed, and fit
- Environment loading through `python-dotenv`

**Technologies/frameworks used:**

- Python, `uv`, LangChain agents, DeepAgents, LangChain OpenAI, `python-dotenv`,
  Ruff, Black

**Learning outcomes:**

- How to define a small callable tool for an agent
- How to switch between shallow and deeper agent orchestration
- Why agent complexity should match task complexity
- How to reason about token, latency, and cost tradeoffs

**Suggested use cases:**

- Agent architecture comparison
- Teaching when not to over-engineer an agent
- Simple tool-calling demos
- Evaluating DeepAgents for larger multi-step workflows

### humaninthelooppersistancememory

**Branch:** [`humaninthelooppersistancememory`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/humaninthelooppersistancememory)

**High-level overview:**  
This branch demonstrates persisted human-in-the-loop execution with LangGraph
checkpoints, plus separate examples for parallel branch execution and
conditional routing. The main workflow pauses before a human-feedback node,
saves state to SQLite, accepts terminal feedback, updates the checkpointed
state, and resumes execution.

**Key features implemented:**

- Typed graph state for persisted workflows
- SQLite checkpointing with `SqliteSaver`
- Interrupt-before-human-feedback execution
- Manual state update and graph resume flow
- Thread-based checkpoint namespaces
- Parallel fan-out and join example
- Conditional branch selection example
- Reducer-backed state aggregation with `Annotated[..., operator.add]`
- Mermaid graph image generation

**Technologies/frameworks used:**

- Python, `uv`, LangGraph, LangGraph SQLite checkpointing, SQLite,
  LangGraph checkpoint packages, `python-dotenv`, Ruff, Black

**Learning outcomes:**

- How to pause and resume a LangGraph workflow
- How checkpointed state enables human approval and correction
- How thread IDs isolate persisted runs
- How LangGraph merges shared state across parallel branches
- How conditional routing can choose multiple downstream nodes

**Suggested use cases:**

- Approval workflows
- Human review checkpoints
- Long-running agent processes
- Stateful workflow demos
- Learning parallel execution and branch joins in LangGraph

### mcp

**Branch:** [`mcp`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/mcp)

**High-level overview:**  
This branch demonstrates how to expose Python functions as Model Context
Protocol tools and consume those tools from a LangChain agent. It includes a
stdio math server started by the client and a Streamable HTTP weather server
that runs as a standalone service.

**Key features implemented:**

- FastMCP stdio server with `add` and `multiply` tools
- FastMCP Streamable HTTP server with a mock weather tool
- MCP client sessions using stdio and HTTP transports
- MCP initialization handshake through `ClientSession.initialize()`
- Tool adaptation with `load_mcp_tools`
- LangChain agent invocation over MCP-provided tools
- Async client workflows for both transports

**Technologies/frameworks used:**

- Python, `uv`, MCP, FastMCP, LangChain MCP adapters, LangChain agents,
  LangChain OpenAI, LangGraph, `python-dotenv`

**Learning outcomes:**

- How MCP servers expose typed Python functions as model-callable tools
- How stdio and Streamable HTTP MCP transports differ
- How LangChain can consume MCP tools through adapter packages
- How to separate tool-serving infrastructure from agent orchestration

**Suggested use cases:**

- MCP learning and demos
- Agent tool server prototypes
- Local tool execution through stdio
- HTTP-hosted tool services for agents
- Experiments with MCP-compatible tool ecosystems

### reflection-agent

**Branch:** [`reflection-agent`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/reflection-agent)

**High-level overview:**  
This branch implements a basic reflection agent in LangGraph. It generates a
LinkedIn post, critiques the draft with a second chain, feeds the critique back
as user-style feedback, and repeats until the graph reaches its stopping
condition.

**Key features implemented:**

- Two-node LangGraph workflow with `generate` and `reflect`
- Generation chain for writing or revising LinkedIn posts
- Reflection chain for critique around clarity, style, length, and virality
- Feedback loop that converts critique into a `HumanMessage`
- Message-count stopping condition
- Mermaid, ASCII, and PNG graph visualization

**Technologies/frameworks used:**

- Python, `uv`, LangGraph, LangChain, LangChain Core, LangChain OpenAI,
  OpenAI chat models, Grandalf, `python-dotenv`

**Learning outcomes:**

- How reflection loops improve generated content
- How to split generation and critique into separate chains
- How LangGraph message state accumulates draft and feedback history
- How to implement a simple bounded improvement loop

**Suggested use cases:**

- Content rewriting workflows
- Critique-driven generation demos
- Prompt iteration experiments
- Teaching reflection as a simple agentic pattern

### reflexion-agent

**Branch:** [`reflexion-agent`](https://github.com/PrynAI/AgenticAI-OnRamp-with-Langgraph/tree/reflexion-agent)

**High-level overview:**  
This branch implements a Reflexion-style research agent. It drafts an answer,
self-critiques the response, generates search queries, executes Tavily search,
and revises the answer with references. The loop can repeat for a configured
number of search-and-revision rounds.

**Key features implemented:**

- LangGraph workflow with `draft`, `execute_tools`, and `revise` nodes
- `MessagesState` conversation state
- Pydantic schemas for structured LLM outputs
- Forced structured tool calls for initial answers and revised answers
- Tavily batch search from model-generated queries
- LangGraph `ToolNode` execution layer
- Citation-backed revision schema
- Loop control based on completed tool-search rounds

**Technologies/frameworks used:**

- Python, `uv`, LangGraph, LangChain, LangChain Core, LangChain OpenAI,
  LangChain Tavily, Pydantic, OpenAI chat models, `python-dotenv`

**Learning outcomes:**

- How Reflexion extends basic reflection with external evidence
- How to use structured outputs instead of fragile text parsing
- How to turn model-generated search plans into tool execution
- How to revise answers with citations after tool feedback
- How to control iterative research loops in LangGraph

**Suggested use cases:**

- Research assistant prototypes
- Search-augmented answer refinement
- Structured-output agent examples
- Citation-aware generation experiments
- Learning the difference between Reflection and Reflexion patterns

## How To Choose A Branch

- Start with `ReAct-Agent-Function-Calling` if you are new to tool-calling
  agents.
- Use `agenticRAG` if your goal is reliable retrieval, grounding, and answer
  verification.
- Use `reflection-agent` if you want the simplest self-improvement loop.
- Use `reflexion-agent` if you want self-critique plus web research and
  citations.
- Use `humaninthelooppersistancememory` if you need approval points,
  checkpointing, or resumable state.
- Use `mcp` if you want to learn how external tool servers connect to agents.
- Use `deepagents` if you want to compare shallow and deeper agent
  orchestration patterns.

## Common Prerequisites

The exact requirements vary by branch, but most modules expect:

- Python `3.13`
- `uv` for dependency installation
- An `OPENAI_API_KEY`
- Additional service keys for specific branches, such as `TAVILY_API_KEY`,
  `WEAVIATE_URL`, `WEAVIATE_API_KEY`, or `LANGSMITH_API_KEY`

After checking out a branch, read that branch's README first, then install and
run the module with the commands documented there.

## Project Goal

The goal of this repository is to make Agentic AI patterns easier to learn by
keeping each concept isolated in its own branch. Instead of one large
application, the repository provides small, inspectable examples that show how
LLM systems can reason, call tools, retrieve context, reflect, revise, pause for
humans, persist state, and integrate with external tool protocols.
