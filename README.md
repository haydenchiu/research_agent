# Research Agent

Multi-agent AI research workflow orchestrated by LangGraph. Six specialized agents collaborate to produce polished research reports from a single question. All runs are traced to W&B/Weave for observability and evaluation.

## Architecture

```
User Query -> Planner -> Searcher -> Analyst -> Critic -+-> Writer -> Format Checker -> Output (MD + PDF)
                                       ^           |
                                       +-- (gaps) -+
```

| Agent | Role | Model |
|-------|------|-------|
| Planner | Decomposes the question into sub-questions | Claude Sonnet |
| Searcher | Web search via Tavily, deduplicates findings | GPT-4o-mini |
| Analyst | Synthesizes findings into a coherent analysis | Claude Sonnet |
| Critic | Reviews for gaps, bias, and logical errors | GPT-4o |
| Writer | Produces the final structured report | Claude Sonnet |
| Format Checker | Validates Markdown structure and citations | GPT-4o-mini |

When the Critic finds gaps it routes back to Searcher for another research loop (up to `--max-revisions`). The Format Checker can also send the draft back to the Writer for formatting corrections.

## Setup

```bash
# Install dependencies (requires Python >= 3.11)
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your keys: OPENAI_API_KEY, ANTHROPIC_API_KEY, TAVILY_API_KEY, WANDB_API_KEY
```

## Usage

```bash
# Basic research query
python main.py "What are the economic impacts of AI on the labor market?"

# With options
python main.py "Your question here" --max-revisions 5 --output-dir ./reports --verbose

# Save per-agent state snapshots for building eval datasets
python main.py "Your question here" --save-trace
```

Options:
- `--max-revisions N` -- maximum critic revision loops (default: 3)
- `--output-dir DIR` -- where to save reports (default: `output/`)
- `--verbose` -- print the full audit trail after completion
- `--save-trace` -- dump per-agent state snapshots to `output/traces/`

## Evaluation

Each agent has a suite of scorers organized in a 2x2 matrix: with/without ground truth crossed with code-based/LLM-judged. Evaluations are powered by `weave.Evaluation` and results are tracked on W&B.

```bash
# Run evals for a single agent (requires a dataset in evals/datasets/)
python -m evals.run --agent planner

# Run evals for all agents
python -m evals.run --all
```

### Building datasets from W&B traces

Traces from agent runs are stored on Weave automatically. You can pull them into eval datasets and run ground-truth-free scorers (structural checks + LLM rubrics) without any manual labeling.

```bash
# Build datasets from Weave traces (gated on a minimum run count)
python -m evals.build_dataset --all --min-runs 5

# Only include traces after a specific date
python -m evals.build_dataset --all --since 2025-06-01

# Run GT-free evals on trace-derived datasets
python -m evals.run --all --gt-free
```

Dataset files are written to `evals/datasets/` (gitignored). Hand-curated datasets use `<agent>.json`; trace-derived datasets use `<agent>_traces.json`.

## Project Structure

```
main.py                  # CLI entry point, Weave-traced workflow runner
agents/                  # One module per agent (each subclasses weave.Model)
graph/
  workflow.py            # LangGraph StateGraph wiring
  state.py               # ResearchState TypedDict
prompts/                 # Markdown system prompts (editable without code changes)
tools/
  search.py              # Tavily web search (@weave.op)
  pdf_export.py          # Markdown-to-PDF via fpdf2
config/settings.py       # Paths, API key helpers
evals/
  run.py                 # Weave evaluation runner (--gt-free support)
  build_dataset.py       # Build eval datasets from W&B/Weave traces
  scorers/               # Per-agent scorer modules
tests/
  test_workflow_dry_run.py
```

## Configuration

Each agent declares its own `model_name`, `provider`, and `temperature` in its class definition under `agents/`. System prompts live in `prompts/*.md` and can be edited without code changes. Global paths and API key helpers are in `config/settings.py`.

## Testing

```bash
python -m tests.test_workflow_dry_run
```
