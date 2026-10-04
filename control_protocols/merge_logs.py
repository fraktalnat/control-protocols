"""Merge several runs of the same task into one .eval log.

Usage:
    python -m control_protocols.merge_logs LOG [LOG ...] [-o OUTPUT]

Useful when a task was run in batches (e.g. problems 1-5, then 6-20) and later
steps need a single log. The first log's header is kept; samples from all logs
are combined. Fails if the logs come from different tasks or share sample ids.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log, write_eval_log

from control_protocols import config


def merge_logs(log_paths: list[str]) -> EvalLog:
    logs = [read_eval_log(p) for p in log_paths]
    merged = logs[0]

    for path, log in zip(log_paths, logs):
        if log.status != "success":
            raise ValueError(f"{path} has status {log.status!r}; only merge finished runs.")
        if log.eval.task != merged.eval.task:
            raise ValueError(
                f"{path} is from task {log.eval.task!r}, expected {merged.eval.task!r}."
            )

    samples = [s for log in logs for s in (log.samples or [])]
    keys = [(s.id, s.epoch) for s in samples]
    duplicates = sorted({k for k in keys if keys.count(k) > 1}, key=str)
    if duplicates:
        raise ValueError(f"Sample ids appear in more than one log: {duplicates}")

    merged.samples = sorted(samples, key=lambda s: (str(type(s.id)), s.id, s.epoch))
    merged.eval.dataset.samples = len({s.id for s in samples})
    merged.eval.dataset.sample_ids = [s.id for s in merged.samples if s.epoch == 1]
    if merged.results is not None:
        merged.results.total_samples = len(samples)
        merged.results.completed_samples = len(samples)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", help=".eval logs to merge, in order.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (default: <log dir>/<timestamp>_<task>-merged.eval).",
    )
    args = parser.parse_args()

    merged = merge_logs(args.logs)
    if args.output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-00-00")
        task = merged.eval.task.split("/")[-1].replace("_", "-")
        args.output = str(Path(config.LOG_DIR) / f"{stamp}_{task}-merged.eval")

    write_eval_log(merged, args.output)
    ids = merged.eval.dataset.sample_ids
    print(f"Merged {len(args.logs)} logs ({len(merged.samples)} samples, ids {ids}) -> {args.output}")


if __name__ == "__main__":
    main()
