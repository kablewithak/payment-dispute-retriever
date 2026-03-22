from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

from app.schemas.dispute_case import EvalQuery
from app.schemas.retrieval import QueryFilters, RetrieveRequest, RetrieveResponse, RetrievalMode


@dataclass(frozen=True)
class QueryEvaluationRow:
    query_id: str
    mode: str
    recall_at_5: float
    mrr_at_10: float
    hit_rate_at_3: float
    issue_family_correct: float
    escalation_team_correct: float
    latency_ms: float


@dataclass(frozen=True)
class ModeEvaluationSummary:
    mode: str
    query_count: int
    recall_at_5: float
    mrr_at_10: float
    hit_rate_at_3: float
    issue_family_accuracy: float
    escalation_team_accuracy: float
    avg_latency_ms: float


class RetrievalEvaluator:
    def __init__(self, rerank_candidate_pool_size: int = 15) -> None:
        self.rerank_candidate_pool_size = rerank_candidate_pool_size

    def load_eval_queries(self, input_path: Path) -> list[EvalQuery]:
        queries: list[EvalQuery] = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    queries.append(EvalQuery.model_validate_json(stripped))
        return queries

    def evaluate_mode(
        self,
        *,
        eval_queries: list[EvalQuery],
        mode: RetrievalMode,
        run_request: Callable[[str, RetrieveRequest], RetrieveResponse],
    ) -> tuple[ModeEvaluationSummary, list[QueryEvaluationRow]]:
        rows: list[QueryEvaluationRow] = []

        for eval_query in eval_queries:
            request = self._build_request(eval_query=eval_query, mode=mode)

            started = perf_counter()
            response = run_request(eval_query.query_id, request)
            latency_ms = round((perf_counter() - started) * 1000, 2)

            retrieved_ids = [case.dispute_id for case in response.top_cases]
            gold_ids = eval_query.gold_relevant_dispute_ids

            rows.append(
                QueryEvaluationRow(
                    query_id=eval_query.query_id,
                    mode=mode.value,
                    recall_at_5=self._recall_at_k(retrieved_ids, gold_ids, 5),
                    mrr_at_10=self._mrr_at_k(retrieved_ids, gold_ids, 10),
                    hit_rate_at_3=self._hit_rate_at_k(retrieved_ids, gold_ids, 3),
                    issue_family_correct=float(
                        response.predicted_issue_family == eval_query.gold_issue_family
                    ),
                    escalation_team_correct=float(
                        response.predicted_escalation_team == eval_query.gold_escalation_team
                    ),
                    latency_ms=latency_ms,
                )
            )

        summary = ModeEvaluationSummary(
            mode=mode.value,
            query_count=len(rows),
            recall_at_5=self._average(row.recall_at_5 for row in rows),
            mrr_at_10=self._average(row.mrr_at_10 for row in rows),
            hit_rate_at_3=self._average(row.hit_rate_at_3 for row in rows),
            issue_family_accuracy=self._average(row.issue_family_correct for row in rows),
            escalation_team_accuracy=self._average(row.escalation_team_correct for row in rows),
            avg_latency_ms=self._average(row.latency_ms for row in rows),
        )
        return summary, rows

    def write_summary_json(
        self,
        *,
        summaries: list[ModeEvaluationSummary],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [asdict(summary) for summary in summaries]
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_summary_csv(
        self,
        *,
        summaries: list[ModeEvaluationSummary],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [asdict(summary) for summary in summaries]
        if not rows:
            return

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    def write_query_rows_csv(
        self,
        *,
        rows: list[QueryEvaluationRow],
        output_path: Path,
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serialized_rows = [asdict(row) for row in rows]
        if not serialized_rows:
            return

        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(serialized_rows[0].keys()))
            writer.writeheader()
            writer.writerows(serialized_rows)

    def _build_request(
        self,
        *,
        eval_query: EvalQuery,
        mode: RetrievalMode,
    ) -> RetrieveRequest:
        use_filters = mode in {
            RetrievalMode.HYBRID_FILTERED,
            RetrievalMode.HYBRID_FILTERED_RERANK,
        }

        filters = QueryFilters(**eval_query.filters) if use_filters else QueryFilters()

        return RetrieveRequest(
            query_text=eval_query.query_text,
            mode=mode,
            limit=10,
            alpha=0.35,
            candidate_pool_size=max(self.rerank_candidate_pool_size, 10),
            filters=filters,
        )

    def _recall_at_k(
        self,
        retrieved_ids: list[str],
        gold_ids: list[str],
        k: int,
    ) -> float:
        gold = set(gold_ids)
        if not gold:
            return 0.0
        hits = sum(1 for dispute_id in retrieved_ids[:k] if dispute_id in gold)
        return hits / len(gold)

    def _mrr_at_k(
        self,
        retrieved_ids: list[str],
        gold_ids: list[str],
        k: int,
    ) -> float:
        gold = set(gold_ids)
        for index, dispute_id in enumerate(retrieved_ids[:k], start=1):
            if dispute_id in gold:
                return 1.0 / index
        return 0.0

    def _hit_rate_at_k(
        self,
        retrieved_ids: list[str],
        gold_ids: list[str],
        k: int,
    ) -> float:
        gold = set(gold_ids)
        return float(any(dispute_id in gold for dispute_id in retrieved_ids[:k]))

    def _average(self, values) -> float:
        values_list = list(values)
        if not values_list:
            return 0.0
        return round(sum(values_list) / len(values_list), 4)