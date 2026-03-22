from __future__ import annotations

import argparse

from app.services.synthetic_data import (
    SyntheticDisputeDataGenerator,
    write_cases_csv,
    write_cases_jsonl,
    write_eval_queries_jsonl,
)
from app.settings import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()

    parser = argparse.ArgumentParser(
        description="Generate synthetic fintech payment dispute cases and eval queries."
    )
    parser.add_argument(
        "--case-count",
        type=int,
        default=settings.synthetic_case_count,
        help="Number of synthetic dispute cases to generate.",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=settings.eval_query_count,
        help="Number of synthetic eval queries to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.random_seed,
        help="Random seed for deterministic dataset generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    settings.ensure_directories()

    generator = SyntheticDisputeDataGenerator(seed=args.seed)
    cases = generator.generate_cases(case_count=args.case_count)
    eval_queries = generator.generate_eval_queries(
        cases=cases,
        eval_query_count=args.eval_count,
    )

    cases_jsonl_path = settings.synthetic_dir / "dispute_cases.jsonl"
    cases_csv_path = settings.synthetic_dir / "dispute_cases.csv"
    eval_jsonl_path = settings.eval_dir / "eval_queries.jsonl"

    write_cases_jsonl(cases=cases, output_path=cases_jsonl_path)
    write_cases_csv(cases=cases, output_path=cases_csv_path)
    write_eval_queries_jsonl(eval_queries=eval_queries, output_path=eval_jsonl_path)

    print("Synthetic dataset generation complete.")
    print(f"Cases JSONL : {cases_jsonl_path}")
    print(f"Cases CSV   : {cases_csv_path}")
    print(f"Eval JSONL  : {eval_jsonl_path}")
    print(f"Seed        : {args.seed}")
    print(f"Case count  : {len(cases)}")
    print(f"Eval count  : {len(eval_queries)}")


if __name__ == "__main__":
    main()