from __future__ import annotations

import json

from app.services.synthetic_data import (
    SyntheticDisputeDataGenerator,
    write_cases_csv,
    write_cases_jsonl,
    write_eval_queries_jsonl,
)


def test_generation_is_deterministic() -> None:
    generator_one = SyntheticDisputeDataGenerator(seed=42)
    generator_two = SyntheticDisputeDataGenerator(seed=42)

    cases_one = generator_one.generate_cases(case_count=20)
    cases_two = generator_two.generate_cases(case_count=20)

    dumps_one = [case.model_dump(mode="json") for case in cases_one]
    dumps_two = [case.model_dump(mode="json") for case in cases_two]

    assert dumps_one == dumps_two


def test_eval_queries_reference_existing_cases() -> None:
    generator = SyntheticDisputeDataGenerator(seed=99)
    cases = generator.generate_cases(case_count=50)
    eval_queries = generator.generate_eval_queries(cases=cases, eval_query_count=10)

    valid_ids = {case.dispute_id for case in cases}
    valid_families = {case.issue_family for case in cases}

    for query in eval_queries:
        assert query.gold_relevant_dispute_ids
        assert set(query.gold_relevant_dispute_ids).issubset(valid_ids)
        assert query.gold_issue_family in valid_families
        assert query.filters["payment_rail"] in {"card", "wallet", "bank_transfer"}


def test_export_writers_create_valid_outputs(tmp_path) -> None:
    generator = SyntheticDisputeDataGenerator(seed=7)
    cases = generator.generate_cases(case_count=30)
    eval_queries = generator.generate_eval_queries(cases=cases, eval_query_count=5)

    cases_jsonl_path = tmp_path / "dispute_cases.jsonl"
    cases_csv_path = tmp_path / "dispute_cases.csv"
    eval_jsonl_path = tmp_path / "eval_queries.jsonl"

    write_cases_jsonl(cases=cases, output_path=cases_jsonl_path)
    write_cases_csv(cases=cases, output_path=cases_csv_path)
    write_eval_queries_jsonl(eval_queries=eval_queries, output_path=eval_jsonl_path)

    assert cases_jsonl_path.exists()
    assert cases_csv_path.exists()
    assert eval_jsonl_path.exists()

    case_lines = cases_jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    eval_lines = eval_jsonl_path.read_text(encoding="utf-8").strip().splitlines()

    assert len(case_lines) == 30
    assert len(eval_lines) == 5

    first_case = json.loads(case_lines[0])
    first_eval = json.loads(eval_lines[0])

    assert first_case["dispute_id"].startswith("D-")
    assert "issue_family" in first_case
    assert first_eval["query_id"].startswith("Q-")
    assert first_eval["gold_relevant_dispute_ids"]