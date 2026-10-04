import argparse
from pprint import pprint

from dotenv import load_dotenv

from control_protocols import config
from control_protocols.dataset import hf_backdoor_log_path
from control_protocols.logs import (
    analyze_detector_scores,
    assemble_protocol_data,
    latest_eval_log,
)
from control_protocols.plots import plot_protocol_comparison, plot_sus_distribution
from control_protocols.protocols import (
    protocol_defer_to_trusted,
    protocol_trusted_editing,
    protocol_trusted_monitoring,
    protocol_upfront_auditing,
    run_all_protocols,
)
from control_protocols.simulation import simulate_safety


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--backdoor-log",
        default=None,
        help="Backdoor-generation .eval log (default: the Hugging Face 500-problem log).",
    )
    args = parser.parse_args()
    load_dotenv()

    backdoor_log_path = args.backdoor_log or hf_backdoor_log_path()

    detector_log_path = latest_eval_log(config.LOG_DIR, "detect-backdoor")
    trusted_solve_log_path = latest_eval_log(config.LOG_DIR, "trusted-solve")
    trusted_edit_log_path = latest_eval_log(config.LOG_DIR, "trusted-edit")

    records = assemble_protocol_data(
        backdoor_log_path=backdoor_log_path,
        detector_log_path=detector_log_path,
        trusted_solve_log_path=trusted_solve_log_path,
        trusted_edit_log_path=trusted_edit_log_path,
    )
    print(f"Assembled {len(records)} per-problem protocol records.")

    submissions = [(r["u_code"], r["u_is_backdoored"]) for r in records]
    trusted_submissions = [(r["t_code"], False) for r in records]
    edited_submissions = [(r["edited_code"], False) for r in records]
    sus_scores = [r["sus_score_u"] for r in records]

    protocol_results = run_all_protocols(
        submissions=submissions,
        trusted_submissions=trusted_submissions,
        edited_submissions=edited_submissions,
        sus_scores=sus_scores,
        protocol_upfront_auditing=protocol_upfront_auditing,
        protocol_trusted_monitoring=protocol_trusted_monitoring,
        protocol_defer_to_trusted=protocol_defer_to_trusted,
        protocol_trusted_editing=protocol_trusted_editing,
    )
    pprint(protocol_results)

    stats, clean_scores, backdoor_scores = analyze_detector_scores(detector_log_path)
    pprint(stats)

    results = simulate_safety(
        n_problems=30,
        backdoor_rate=0.05,
        sus_score_clean=clean_scores,
        sus_score_backdoor=backdoor_scores,
        n_simulations=200,
    )
    pprint(results)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sus_path = config.RESULTS_DIR / "sus_distribution.png"
    comparison_path = config.RESULTS_DIR / "protocol_comparison.png"
    plot_sus_distribution(
        clean_scores, backdoor_scores, sus_path, stats.get("threshold_2pct_fpr")
    )
    plot_protocol_comparison(results, comparison_path)
    print(f"Saved {sus_path} and {comparison_path}")


if __name__ == "__main__":
    main()
