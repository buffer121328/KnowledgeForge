"""Inject application services and build the stable workflow registry."""

from __future__ import annotations

from typing import Any

from agents.document_parser import DocParserAgent
from agents.knowledge_extractor import KnowledgeExtractAgent
from agents.knowledge_updater import KnowledgeUpdateAgent
from agents.qa_agent import QAAgent
from workflows.ask_question import build_ask_question_workflow
from workflows.qa_dependencies import build_qa_agent_dependencies
from workflows.ingest_document import build_ingest_document_workflow
from workflows.update_knowledge import build_update_knowledge_workflow


def build_workflow_registry(
    vector_store: Any = None,
    knowledge_graph: Any = None,
    sparse_index: Any = None,
    *,
    qa_cache: Any = None,
    revision_store: Any = None,
    semantic_cache: Any = None,
    confirmation_service: Any = None,
) -> dict[str, Any]:
    """Build the lifecycle-owned ingest, QA, and update workflow registry."""
    document_parser = DocParserAgent()
    knowledge_extractor = KnowledgeExtractAgent()
    qa_dependencies = build_qa_agent_dependencies()
    qa_agent = QAAgent(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        sparse_index=sparse_index,
        qa_cache=qa_cache,
        revision_store=revision_store,
        semantic_cache=semantic_cache,
        confirmation_service=confirmation_service,
        dependencies=qa_dependencies,
    )
    knowledge_updater = KnowledgeUpdateAgent(
        doc_parser=document_parser,
        knowledge_extractor=knowledge_extractor,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        sparse_index=sparse_index,
    )
    ingest_kwargs: dict[str, Any] = {}
    if revision_store is not None:
        ingest_kwargs["revision_store"] = revision_store
    if sparse_index is not None:
        ingest_kwargs["sparse_index"] = sparse_index
    ingest_workflow = build_ingest_document_workflow(
        document_parser,
        knowledge_extractor,
        vector_store,
        knowledge_graph,
        **ingest_kwargs,
    )
    update_workflow = (
        build_update_knowledge_workflow(knowledge_updater, revision_store=revision_store)
        if revision_store is not None
        else build_update_knowledge_workflow(knowledge_updater)
    )
    return {
        "ingest": ingest_workflow,
        "qa": build_ask_question_workflow(qa_agent),
        "update": update_workflow,
    }
