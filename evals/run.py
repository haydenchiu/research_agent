"""CLI entry point for running Weave evaluations against agent models.

Usage:
    python -m evals.run --agent planner
    python -m evals.run --all
    python -m evals.run --agent writer --dataset path/to/custom.json
    python -m evals.run --agent planner --project my-wandb-project
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from importlib import import_module
from pathlib import Path

import weave
from dotenv import load_dotenv

EVALS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVALS_DIR / "datasets"
# Running `python evals/run.py` puts `evals/` on sys.path[0]; top-level `agents/` lives at repo root.
sys.path.insert(0, str(EVALS_DIR.parent))

AGENT_REGISTRY = {
    "planner": {
        "model_module": "agents.planner",
        "model_class": "PlannerAgent",
        "scorer_module": "evals.scorers.planner",
    },
    "searcher": {
        "model_module": "agents.searcher",
        "model_class": "SearcherAgent",
        "scorer_module": "evals.scorers.searcher",
    },
    "analyst": {
        "model_module": "agents.analyst",
        "model_class": "AnalystAgent",
        "scorer_module": "evals.scorers.analyst",
    },
    "critic": {
        "model_module": "agents.critic",
        "model_class": "CriticAgent",
        "scorer_module": "evals.scorers.critic",
    },
    "writer": {
        "model_module": "agents.writer",
        "model_class": "WriterAgent",
        "scorer_module": "evals.scorers.writer",
    },
    "format_checker": {
        "model_module": "agents.format_checker",
        "model_class": "FormatCheckerAgent",
        "scorer_module": "evals.scorers.format_checker",
    },
}


def _load_model(agent_name: str) -> weave.Model:
    info = AGENT_REGISTRY[agent_name]
    mod = import_module(info["model_module"])
    cls = getattr(mod, info["model_class"])
    return cls()


def _load_scorers(agent_name: str, gt_free: bool = False) -> list:
    info = AGENT_REGISTRY[agent_name]
    mod = import_module(info["scorer_module"])
    scorers = mod.get_scorers()
    if gt_free:
        scorers = [s for s in scorers if _is_gt_free_scorer(s)]
    return scorers


def _is_gt_free_scorer(scorer) -> bool:
    """True when the scorer's call signature has no ``target`` parameter."""
    try:
        if hasattr(scorer, "score") and callable(scorer.score):
            sig = inspect.signature(scorer.score)
        elif callable(scorer):
            sig = inspect.signature(scorer)
        else:
            return True
        return "target" not in sig.parameters
    except (ValueError, TypeError):
        return True


def _load_dataset(
    agent_name: str,
    dataset_path: str | None = None,
    gt_free: bool = False,
) -> list[dict]:
    if dataset_path:
        path = Path(dataset_path)
    elif gt_free:
        trace_path = DATASETS_DIR / f"{agent_name}_traces.json"
        default_path = DATASETS_DIR / f"{agent_name}.json"
        path = trace_path if trace_path.exists() else default_path
    else:
        path = DATASETS_DIR / f"{agent_name}.json"
    if not path.exists():
        print(f"Error: dataset not found at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


async def run_eval(
    agent_name: str,
    dataset_path: str | None = None,
    gt_free: bool = False,
) -> dict:
    """Run evaluation for a single agent and return results."""
    print(f"\n{'=' * 60}")
    print(f"  Evaluating: {agent_name}" + (" (GT-free)" if gt_free else ""))
    print(f"{'=' * 60}")

    model = _load_model(agent_name)
    scorers = _load_scorers(agent_name, gt_free=gt_free)
    dataset = _load_dataset(agent_name, dataset_path, gt_free=gt_free)

    if not scorers:
        print(f"  No {'GT-free ' if gt_free else ''}scorers available, skipping")
        return {}

    print(f"  Model:    {model.__class__.__name__}")
    print(f"  Scorers:  {len(scorers)}")
    print(f"  Dataset:  {len(dataset)} rows")

    evaluation = weave.Evaluation(
        name=f"{agent_name}_eval",
        dataset=dataset,
        scorers=scorers,
    )
    results = await evaluation.evaluate(model)

    print(f"\n  Results for {agent_name}:")
    for key, value in results.items():
        if key == "model_latency":
            mean = value.get("mean", 0)
            print(f"    {key}: {mean:.2f}s (mean)")
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    for metric, metric_value in sub_value.items():
                        if isinstance(metric_value, (int, float)):
                            print(f"    {key}.{sub_key}.{metric}: {metric_value:.3f}")
                elif isinstance(sub_value, (int, float)):
                    print(f"    {key}.{sub_key}: {sub_value:.3f}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Weave evaluations for research agent models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m evals.run --agent planner\n"
            "  python -m evals.run --all\n"
            "  python -m evals.run --agent writer --dataset path/to/custom.json\n"
            "  python -m evals.run --all --gt-free              # GT-free scorers only\n"
            "  python -m evals.run --agent planner --gt-free    # uses *_traces.json if available\n"
            "  python -m evals.batch_trace_eval --agent planner # GT-free on cached trace outputs\n"
        ),
    )
    parser.add_argument(
        "--agent",
        choices=list(AGENT_REGISTRY.keys()),
        help="Which agent to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all agents",
    )
    parser.add_argument(
        "--dataset",
        help="Path to a custom dataset JSON file (overrides default)",
    )
    parser.add_argument(
        "--project",
        default="research-agent",
        help="Weave project name (default: research-agent)",
    )
    parser.add_argument(
        "--gt-free",
        action="store_true",
        help=(
            "Run only ground-truth-free scorers (structural checks + LLM rubrics). "
            "Auto-selects *_traces.json datasets when available."
        ),
    )
    args = parser.parse_args()

    if not args.agent and not args.all:
        parser.error("Specify --agent <name> or --all")

    load_dotenv()
    weave.init(args.project)

    if args.all:
        agents = list(AGENT_REGISTRY.keys())
    else:
        agents = [args.agent]

    all_results = {}
    for agent_name in agents:
        dataset_path = args.dataset if len(agents) == 1 else None
        result = asyncio.run(run_eval(agent_name, dataset_path, gt_free=args.gt_free))
        all_results[agent_name] = result

    print(f"\n{'=' * 60}")
    print(f"  Evaluation complete for {len(agents)} agent(s)")
    print(f"  View results at: https://wandb.ai/home -> Weave project '{args.project}'")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
