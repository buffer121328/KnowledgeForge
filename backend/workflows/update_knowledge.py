"""LangGraph incremental knowledge-update workflow."""

from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.knowledge_updater import KnowledgeUpdateAgent
from domain.tasks import DocumentChange, UpdateResult


class UpdateState(TypedDict, total=False):
    """Track update state."""
    changes: list[DocumentChange]
    tenant_id: str
    results: list[UpdateResult]


def build_update_knowledge_workflow(
    update_agent: KnowledgeUpdateAgent,
    revision_store: Any = None,
) -> StateGraph:
    """Build the update knowledge workflow."""
    async def process_updates(state: UpdateState) -> dict:
        """Process the updates."""
        return {"results": await update_agent.process_batch(state.get("changes", []), tenant_id=state.get("tenant_id", ""))}

    def should_continue(state: UpdateState) -> str:
        """Handle should continue for the module."""
        return "retry" if any(not result.success for result in state.get("results", [])) else "done"

    async def retry_failed(state: UpdateState) -> dict:
        """Retry the failed."""
        failed_changes = [result.change for result in state.get("results", []) if not result.success]
        retried = await update_agent.process_batch(failed_changes, tenant_id=state.get("tenant_id", ""))
        return {"results": [result for result in state.get("results", []) if result.success] + retried}

    async def advance_revision(state: UpdateState) -> dict:
        """Advance once when at least one mutation completed successfully."""
        if (
            revision_store is not None
            and state.get("tenant_id")
            and any(result.success for result in state.get("results", []))
        ):
            await revision_store.advance(state["tenant_id"])
        return {}

    graph = StateGraph(UpdateState)
    graph.add_node("process", process_updates)
    graph.add_node("retry", retry_failed)
    graph.add_node("advance_revision", advance_revision)
    graph.set_entry_point("process")
    graph.add_conditional_edges("process", should_continue, {"retry": "retry", "done": "advance_revision"})
    graph.add_edge("retry", "advance_revision")
    graph.add_edge("advance_revision", END)
    return graph.compile()
