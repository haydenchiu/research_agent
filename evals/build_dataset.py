"""Build eval datasets from W&B/Weave traces.

Queries Weave for per-agent predict traces and writes them as dataset JSON
files suitable for ground-truth-free evaluation.

Usage:
    python -m evals.build_dataset --agent planner
    python -m evals.build_dataset --all
    python -m evals.build_dataset --all --min-runs 5
    python -m evals.build_dataset --all --since 2025-01-01
    python -m evals.build_dataset --agent writer --since 2025-06-01T12:00:00Z
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import weave
from dotenv import load_dotenv
from weave.trace_server.trace_server_interface import CallsFilter

EVALS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVALS_DIR / "datasets"

# Maps each agent to its Weave op class name, the predict-input fields to
# extract, and any extra fields that scorers need but are only available in
# the Weave trace attributes (set via ``weave.attributes(...)`` in main.py).
AGENT_TRACE_CONFIG = {
    "planner": {
        "class_name": "PlannerAgent",
        "input_fields": ["research_query"],
    },
    "searcher": {
        "class_name": "SearcherAgent",
        "input_fields": ["sub_questions"],
        "extra_from_attributes": ["research_query"],
    },
    "analyst": {
        "class_name": "AnalystAgent",
        "input_fields": ["research_query", "search_results"],
    },
    "critic": {
        "class_name": "CriticAgent",
        "input_fields": ["research_query", "analysis", "revision_count", "max_revisions"],
    },
    "writer": {
        "class_name": "WriterAgent",
        "input_fields": ["research_query", "analysis", "format_issues"],
    },
    "format_checker": {
        "class_name": "FormatCheckerAgent",
        "input_fields": ["draft_report"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_since(value: str | None) -> datetime | None:
    """Parse an ISO-format datetime string, defaulting to UTC if naive."""
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _op_name_str(call) -> str:
    return str(getattr(call, "op_name", ""))


def _started_at(call) -> datetime | None:
    return getattr(call, "started_at", None)


def _after_since(call, since: datetime | None) -> bool:
    if since is None:
        return True
    started = _started_at(call)
    return started is not None and started >= since


# ---------------------------------------------------------------------------
# Trace querying
# ---------------------------------------------------------------------------


def count_root_runs(client, since: datetime | None = None) -> int:
    """Count root-level workflow runs, excluding Evaluation traces."""
    root_calls = client.get_calls(filter=CallsFilter(trace_roots_only=True))
    count = 0
    for call in root_calls:
        op = _op_name_str(call)
        if "Evaluation" in op:
            continue
        if not _after_since(call, since):
            continue
        count += 1
    return count


def get_agent_calls(
    client,
    class_name: str,
    since: datetime | None = None,
) -> list:
    """Return all ``<ClassName>.predict`` calls for *class_name*."""
    pattern = f"{class_name}.predict"
    matched = []
    for call in client.get_calls():
        if pattern not in _op_name_str(call):
            continue
        if not _after_since(call, since):
            continue
        matched.append(call)
    return matched


def _extract_row(call, config: dict) -> dict:
    """Turn a single Weave call into a dataset row."""
    raw_inputs = call.inputs if call.inputs else {}
    inputs = {k: v for k, v in raw_inputs.items() if k != "self"}
    attrs = dict(getattr(call, "attributes", None) or {})

    row: dict = {"id": f"trace_{call.id}"}

    for field in config["input_fields"]:
        value = inputs.get(field)
        if value is not None:
            row[field] = value

    for field in config.get("extra_from_attributes", []):
        if field not in row:
            value = attrs.get(field)
            if value is not None:
                row[field] = value

    return row


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------


def build_agent_dataset(
    client,
    agent_name: str,
    since: datetime | None = None,
) -> list[dict]:
    """Build a GT-free dataset for *agent_name* from Weave traces."""
    config = AGENT_TRACE_CONFIG[agent_name]
    calls = get_agent_calls(client, config["class_name"], since=since)
    return [_extract_row(c, config) for c in calls]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build eval datasets from W&B/Weave traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m evals.build_dataset --agent planner\n"
            "  python -m evals.build_dataset --all\n"
            "  python -m evals.build_dataset --all --min-runs 5\n"
            "  python -m evals.build_dataset --all --since 2025-01-01\n"
        ),
    )
    parser.add_argument(
        "--agent",
        choices=list(AGENT_TRACE_CONFIG.keys()),
        help="Agent to build a dataset for",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build datasets for all agents",
    )
    parser.add_argument(
        "--min-runs",
        type=int,
        default=0,
        help="Minimum root workflow runs required before building (0 = always build)",
    )
    parser.add_argument(
        "--since",
        help="Only include traces after this ISO datetime (e.g. 2025-01-01)",
    )
    parser.add_argument(
        "--entity",
        default="haydenchiush",
        help="W&B entity / username (default: haydenchiush)",
    )
    parser.add_argument(
        "--project",
        default="research-agent",
        help="Weave project name (default: research-agent)",
    )
    args = parser.parse_args()

    if not args.agent and not args.all:
        parser.error("Specify --agent <name> or --all")

    load_dotenv()
    since = _parse_since(args.since)
    client = weave.init(f"{args.entity}/{args.project}")

    if args.min_runs > 0:
        root_count = count_root_runs(client, since=since)
        print(f"Root workflow runs found: {root_count} (threshold: {args.min_runs})")
        if root_count < args.min_runs:
            print("Not enough runs yet. Exiting without building datasets.")
            sys.exit(0)

    agents = list(AGENT_TRACE_CONFIG.keys()) if args.all else [args.agent]
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for agent_name in agents:
        rows = build_agent_dataset(client, agent_name, since=since)
        if not rows:
            print(f"  {agent_name}: no matching traces, skipping")
            continue

        output_path = DATASETS_DIR / f"{agent_name}_traces.json"
        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2, default=str)
        total_rows += len(rows)
        print(f"  {agent_name}: {len(rows)} rows -> {output_path}")

    if total_rows == 0:
        print("\nNo traces found. Run the agent first:")
        print('  python main.py "Your research question"')
    else:
        print(f"\n{total_rows} total rows written to {DATASETS_DIR}")
        print("Run GT-free evals with:")
        print("  python -m evals.run --all --gt-free")


if __name__ == "__main__":
    main()
