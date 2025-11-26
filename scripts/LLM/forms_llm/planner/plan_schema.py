# scripts/planner/plan_schema.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PlannedTask:
    """
    Generic task as understood by the planner.

    kind:
      - "doc_qa"
      - "doc_explain"
      - "doc_summary"
      - "doc_translate"
      - "finance_emi"   (more later if needed)

    scope:
      - "local"  (clause-level, uses chunks)
      - "global" (whole document, uses blocks)   # only for doc_* kinds
    """
    id: str
    kind: str
    query: str | None = None
    domain: str | None = None
    scope: str | None = None
    language: str = "en"
    tone: str | None = None
    depends_on: List[str] = field(default_factory=list)

    # Extra numeric or structured info for tools (e.g. EMI inputs)
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize task with a strict key policy:
        - payload appears ONLY for finance_* tasks AND only if non-empty
        """
        base = {
            "id": self.id,
            "kind": self.kind,
            "query": self.query,
            "domain": self.domain,
            "scope": self.scope,
            "language": self.language,
            "tone": self.tone,
            "depends_on": self.depends_on or [],
        }

        if self.kind.startswith("finance_") and self.payload:
            base["payload"] = self.payload

        return base


@dataclass
class PlannerPlan:
    """
    Full plan produced by the planner LLM (or rule engine).

    - tasks: ordered list; we execute them in list order,
      depends_on expresses relationships.
    - final_answer_instructions: how to combine task results
      in the final LLM answer.
    """
    tasks: List[PlannedTask]
    final_answer_instructions: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tasks": [t.to_dict() for t in self.tasks],
            "final_answer_instructions": self.final_answer_instructions,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PlannerPlan":
        tasks_data = data.get("tasks", [])
        tasks = [
            PlannedTask(
                id=t["id"],
                kind=t["kind"],
                query=t.get("query"),
                domain=t.get("domain"),
                scope=t.get("scope"),
                language=t.get("language", "en"),
                tone=t.get("tone"),
                depends_on=t.get("depends_on", []),
                payload=t.get("payload", {}) or {},
            )
            for t in tasks_data
        ]
        return PlannerPlan(
            tasks=tasks,
            final_answer_instructions=data.get("final_answer_instructions", ""),
        )
