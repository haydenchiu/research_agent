#!/usr/bin/env python3
"""Multi-agent AI research workflow orchestrated by LangGraph."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import weave
from dotenv import load_dotenv


def _print_progress(messages: list[dict]) -> None:
    """Print the latest agent message as a progress update."""
    if not messages:
        return
    latest = messages[-1]
    agent = latest.get("agent", "unknown")
    msg = latest.get("message", "")
    print(f"  [{agent}] {msg}")


TRACE_FIELDS_PER_AGENT = {
    "planner": ["research_query", "sub_questions"],
    "searcher": ["sub_questions", "search_results"],
    "analyst": ["research_query", "search_results", "analysis"],
    "critic": ["research_query", "analysis", "revision_count", "max_revisions", "critique"],
    "writer": ["research_query", "analysis", "format_issues", "draft_report"],
    "format_checker": ["draft_report", "final_report", "format_issues"],
}


def _save_trace(state: dict, trace_dir: Path) -> None:
    """Save per-agent state snapshots for building eval datasets."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    traces = {}
    for agent_name, fields in TRACE_FIELDS_PER_AGENT.items():
        snapshot = {}
        for field in fields:
            value = state.get(field)
            if value is not None:
                snapshot[field] = value
        traces[agent_name] = snapshot

    slug = state.get("research_query", "unknown")[:60].replace(" ", "_").replace("/", "_")
    trace_path = trace_dir / f"{slug}_trace.json"
    with open(trace_path, "w") as f:
        json.dump(traces, f, indent=2, default=str)
    print(f"\nTrace saved: {trace_path}")


@weave.op
def run(
    query: str,
    *,
    max_revisions: int = 3,
    output_dir: str = "output",
    verbose: bool = False,
    save_trace: bool = False,
) -> Path:
    """Execute the full research workflow and return the output path."""
    from config.settings import OUTPUT_DIR
    from graph.workflow import compile_workflow
    from tools.pdf_export import export_pdf

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nResearch query: {query}")
    print(f"Max revisions: {max_revisions}")
    print(f"Output directory: {out.resolve()}\n")

    workflow = compile_workflow()

    initial_state = {
        "research_query": query,
        "sub_questions": [],
        "search_results": [],
        "analysis": "",
        "critique": {},
        "draft_report": "",
        "format_issues": [],
        "final_report": "",
        "revision_count": 0,
        "max_revisions": max_revisions,
        "format_revision_count": 0,
        "messages": [],
    }

    print("Starting research workflow...\n")
    start_time = time.time()

    prev_message_count = 0
    final_state = None
    with weave.attributes({"research_query": query, "max_revisions": max_revisions}):
        for event in workflow.stream(initial_state, stream_mode="values"):
            final_state = event
            messages = event.get("messages", [])
            new_messages = messages[prev_message_count:]
            for msg in new_messages:
                agent = msg.get("agent", "unknown")
                text = msg.get("message", "")
                print(f"  [{agent}] {text}")
            prev_message_count = len(messages)

    elapsed = time.time() - start_time
    print(f"\nWorkflow completed in {elapsed:.1f}s")

    if final_state is None:
        print("Error: workflow produced no output.")
        sys.exit(1)

    if save_trace:
        _save_trace(final_state, out / "traces")

    final_report = final_state.get("final_report", "") or final_state.get("draft_report", "")

    if not final_report:
        print("Warning: no report was generated.")
        sys.exit(1)

    slug = query[:60].replace(" ", "_").replace("/", "_")
    md_path = out / f"{slug}.md"
    md_path.write_text(final_report)
    print(f"\nMarkdown report saved: {md_path}")

    pdf_path = out / f"{slug}.pdf"
    try:
        export_pdf(final_report, pdf_path)
        print(f"PDF report saved:      {pdf_path}")
    except Exception as exc:
        print(f"Warning: PDF export failed: {exc}")

    if verbose:
        print("\n--- Audit Trail ---")
        for msg in final_state.get("messages", []):
            print(f"  [{msg.get('agent', '?')}] {msg.get('message', '')}")

    return md_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-agent AI research workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "What are the economic impacts of AI on the labor market?"\n'
            '  python main.py "..." --max-revisions 5 --output-dir ./reports --verbose\n'
            '  python main.py "..." --save-trace   # dump per-agent state for eval datasets\n'
        ),
    )
    parser.add_argument("query", help="The research question to investigate")
    parser.add_argument(
        "--max-revisions",
        type=int,
        default=3,
        help="Maximum number of critic revision loops (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for output files (default: output/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full audit trail after completion",
    )
    parser.add_argument(
        "--save-trace",
        action="store_true",
        help="Save per-agent state snapshots to output/traces/ for building eval datasets",
    )
    args = parser.parse_args()

    load_dotenv()
    weave.init("research-agent")

    run(
        args.query,
        max_revisions=args.max_revisions,
        output_dir=args.output_dir,
        verbose=args.verbose,
        save_trace=args.save_trace,
    )


if __name__ == "__main__":
    main()
