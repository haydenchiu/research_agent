# Research Agent

Multi-agent AI research workflow orchestrated by LangGraph. Eight specialized agents collaborate to produce polished research reports from a single question.

## Architecture

```
User Query -> Planner -> Searcher -> Analyst -> Critic -+-> Writer -> Format Checker -> Output (MD + PDF)
                  ^                                |    |
                  +--- (gaps found) ---------------+    |
                                                        |
                  Data Analyst <-> Chart Reviewer ------+
```

| Agent | Role | Model |
|-------|------|-------|
| Planner | Decomposes the question into sub-questions | Claude |
| Searcher | Web search via Tavily, deduplicates findings | GPT-4o-mini |
| Analyst | Synthesizes findings into a coherent analysis | Claude |
| Critic | Reviews for gaps, bias, and logical errors | GPT-4o |
| Data Analyst | Runs Python/pandas code for quantitative analysis | GPT-4o |
| Chart Reviewer | Visually inspects charts using vision capabilities | GPT-4o (vision) |
| Writer | Produces the final structured report | Claude |
| Format Checker | Validates Markdown structure and citations | GPT-4o-mini |

## Setup

```bash
# Clone and install
uv sync

# Configure API keys
cp .env.example .env
# Edit .env with your keys for OpenAI, Anthropic, and Tavily
```

## Usage

```bash
# Basic research query
python main.py "What are the economic impacts of AI on the labor market?"

# With options
python main.py "Your question here" --max-revisions 5 --output-dir ./reports --verbose
```

Options:
- `--max-revisions N` -- maximum critic revision loops (default: 3)
- `--output-dir DIR` -- where to save reports (default: `output/`)
- `--verbose` -- print the full audit trail after completion

## Configuration

Model assignments are in `config/settings.py`. Agent system prompts are in `prompts/*.md` -- edit them without code changes.

## Testing

```bash
python -m tests.test_workflow_dry_run
```
