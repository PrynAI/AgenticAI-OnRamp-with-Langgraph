# Agentic AI OnRamp with LangGraph

Small LangGraph example showing how to pause a graph for human input, persist the
paused state in SQLite, update that state, and then resume execution.

## What this project demonstrates

- A typed LangGraph state object in `main.py`
- A simple linear graph: `START -> step_1 -> human_feedback -> step_3 -> END`
- A `SqliteSaver` checkpointer for local long-term memory
- `interrupt_before=["human_feedback"]` to pause before the human feedback node
- Manual state updates with `graph.update_state(...)`
- Graph visualization output to `graph.png`

## Project files

- `main.py` - builds and runs the LangGraph workflow
- `pyproject.toml` - Python project metadata and dependencies
- `uv.lock` - locked dependency versions for `uv`
- `LTM_PersistanceDB.sqlite` - generated local checkpoint database
- `graph.png` - generated graph visualization

## Setup

This project targets Python 3.13 and uses `uv` for dependency management.

```powershell
uv sync
```

If you are not using `uv`, create a virtual environment and install the project
dependencies from `pyproject.toml`.

## Run

```powershell
uv run python main.py
```

The script will:

1. Load environment variables from `.env` if one exists.
2. Build the LangGraph workflow.
3. Save checkpoints to `LTM_PersistanceDB.sqlite`.
4. Render the graph to `graph.png`.
5. Start a thread with `thread_id` set to `1`.
6. Run until the `human_feedback` interrupt.
7. Ask for feedback in the terminal.
8. Update the saved state with that feedback.
9. Resume the graph and complete execution.

## Notes

- `LTM_PersistanceDB.sqlite`, `LTM_PersistanceDB.sqlite-wal`, and
  `LTM_PersistanceDB.sqlite-shm` are runtime SQLite files.
- Running the script again with the same `thread_id` reuses the same checkpoint
  namespace.
- Change the `thread_id` value in `main.py` when you want a fresh independent
  run without clearing the SQLite database.
