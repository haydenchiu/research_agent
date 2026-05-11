"""GT-free batch evaluation using cached Weave trace outputs (no model.predict).

Reads datasets produced by ``python -m evals.build_dataset`` (rows include
``predict_output``). Scores each row with the same GT-free scorers as
``evals.run --gt-free``, but logs results via ``weave.EvaluationLogger`` instead
of ``Evaluation.evaluate(model)``, so agent inference is not re-run.

Usage:
    python -m evals.batch_trace_eval --agent planner
    python -m evals.batch_trace_eval --all
    python -m evals.batch_trace_eval --agent planner --dataset path/to.json
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import weave
from dotenv import load_dotenv
from weave import EvaluationLogger
from weave.flow.scorer import Scorer, prepare_scorer_op_args
from weave.flow.scorer import get_scorer_attributes
from weave.trace.isinstance import weave_isinstance
from weave.trace.op import as_op

from .trace_payload import TRACE_PREDICT_OUTPUT_FIELD

EVALS_DIR = Path(__file__).resolve().parent
DATASETS_DIR = EVALS_DIR / "datasets"
sys.path.insert(0, str(EVALS_DIR.parent))

AGENT_REGISTRY = {
    "planner": {"scorer_module": "evals.scorers.planner"},
    "searcher": {"scorer_module": "evals.scorers.searcher"},
    "analyst": {"scorer_module": "evals.scorers.analyst"},
    "critic": {"scorer_module": "evals.scorers.critic"},
    "writer": {"scorer_module": "evals.scorers.writer"},
    "format_checker": {"scorer_module": "evals.scorers.format_checker"},
}


def _is_gt_free_scorer(scorer) -> bool:
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


def _load_gt_free_scorers(agent_name: str) -> list:
    info = AGENT_REGISTRY[agent_name]
    mod = import_module(info["scorer_module"])
    scorers = mod.get_scorers()
    return [s for s in scorers if _is_gt_free_scorer(s)]


def _load_dataset(agent_name: str, dataset_path: str | None) -> list[dict]:
    if dataset_path:
        path = Path(dataset_path)
    else:
        trace_path = DATASETS_DIR / f"{agent_name}_traces.json"
        default_path = DATASETS_DIR / f"{agent_name}.json"
        path = trace_path if trace_path.exists() else default_path
    if not path.exists():
        print(f"Error: dataset not found at {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def _strip_predict_output(row: dict) -> tuple[dict, object]:
    """Return (example inputs for Weave, cached predict output)."""
    if TRACE_PREDICT_OUTPUT_FIELD not in row:
        raise KeyError(TRACE_PREDICT_OUTPUT_FIELD)
    out = row[TRACE_PREDICT_OUTPUT_FIELD]
    inputs = {k: v for k, v in row.items() if k != TRACE_PREDICT_OUTPUT_FIELD}
    return inputs, out


def _invoke_scorer_result(scorer, example: dict, model_output: object) -> object:
    """Run scorer logic without ``apply_scorer_async`` (avoids duplicate score calls)."""
    score_op, score_args = prepare_scorer_op_args(scorer, example, model_output)
    op = as_op(score_op)
    fn = op.resolve_fn
    kwargs = dict(score_args)
    if weave_isinstance(scorer, Scorer):
        kwargs = {"self": scorer, **kwargs}
    result = fn(**kwargs)
    if inspect.isawaitable(result):
        return asyncio.run(result)
    return result


def _scorer_row_scores(
    scorers: list,
    example: dict,
    model_output: object,
) -> dict[str, object]:
    scores: dict[str, object] = {}
    for scorer in scorers:
        attrs = get_scorer_attributes(scorer)
        scores[attrs.scorer_name] = _invoke_scorer_result(scorer, example, model_output)
    return scores


def run_batch_trace_eval(
    agent_name: str,
    dataset_path: str | None = None,
    eval_display_name: str | None = None,
) -> None:
    scorers = _load_gt_free_scorers(agent_name)
    rows = _load_dataset(agent_name, dataset_path)

    if not scorers:
        print(f"  No GT-free scorers for {agent_name}, nothing to do.")
        return

    usable = [r for r in rows if TRACE_PREDICT_OUTPUT_FIELD in r]
    skipped = len(rows) - len(usable)
    if not usable:
        print(
            f"Error: no rows contain '{TRACE_PREDICT_OUTPUT_FIELD}'. "
            "Rebuild datasets with: python -m evals.build_dataset --agent "
            f"{agent_name}"
        )
        sys.exit(1)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%MZ")
    name = eval_display_name or f"{agent_name}_trace_batch_{ts}"

    ds_name = Path(dataset_path).stem if dataset_path else f"{agent_name}_traces"
    eval_logger = EvaluationLogger(
        name=name,
        model={"name": f"{agent_name}-trace-replay"},
        dataset=ds_name,
        eval_attributes={
            "eval_kind": "trace_batch_gt_free",
            "agent": agent_name,
            "rows_total": len(rows),
            "rows_scored": len(usable),
            "rows_skipped_no_output": skipped,
        },
    )

    print(f"\n{'=' * 60}")
    print(f"  Trace batch eval: {agent_name} (GT-free, no predict)")
    print(f"{'=' * 60}")
    print(f"  Rows in file: {len(rows)}")
    print(f"  Rows with {TRACE_PREDICT_OUTPUT_FIELD}: {len(usable)}")
    if skipped:
        print(f"  Skipped (missing output field): {skipped}")
    print(f"  Scorers: {len(scorers)}")

    for i, row in enumerate(usable):
        example, output = _strip_predict_output(row)
        score_map = _scorer_row_scores(scorers, example, output)
        eval_logger.log_example(inputs=example, output=output, scores=score_map)
        if (i + 1) % 10 == 0 or i + 1 == len(usable):
            print(f"  Logged {i + 1}/{len(usable)} examples")

    eval_logger.log_summary({"agent": agent_name, "mode": "trace_batch"})
    url = eval_logger.ui_url
    print(f"\n  Done. EvaluationLogger name: {name}")
    if url:
        print(f"  Weave: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "GT-free eval on trace datasets with cached predict outputs; "
            "uses EvaluationLogger (does not call the agent model)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Requires JSON rows with a "
            f"'{TRACE_PREDICT_OUTPUT_FIELD}' field (from evals.build_dataset).\n"
            "Examples:\n"
            "  python -m evals.batch_trace_eval --agent planner\n"
            "  python -m evals.batch_trace_eval --all\n"
        ),
    )
    parser.add_argument("--agent", choices=list(AGENT_REGISTRY.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", help="Override dataset JSON path")
    parser.add_argument(
        "--project",
        default="research-agent",
        help="Weave project name (default: research-agent)",
    )
    parser.add_argument(
        "--eval-name",
        help="Custom display name for this evaluation in Weave",
    )
    args = parser.parse_args()

    if not args.agent and not args.all:
        parser.error("Specify --agent <name> or --all")

    load_dotenv()
    weave.init(args.project)

    agents = list(AGENT_REGISTRY.keys()) if args.all else [args.agent]
    for agent_name in agents:
        ds = args.dataset if len(agents) == 1 else None
        run_batch_trace_eval(agent_name, dataset_path=ds, eval_display_name=args.eval_name)

    print(f"\n  View results in Weave for project '{args.project}'")


if __name__ == "__main__":
    main()
