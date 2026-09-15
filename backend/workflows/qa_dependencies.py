"""Composition-root construction for runtime QA collaborators."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from services.evidence.qualification import (
    EvidenceQualificationPolicy,
    EvidenceQualifier,
)
from agents.qa_agent import QAAgentDependencies
from services.qa.grounding import GroundingVerifier
from services.safety.pipeline import build_runtime_safety_pipeline
from infrastructure.retrieval.cross_encoder import SafeCrossEncoderReranker
from infrastructure.retrieval.dashscope_reranker import DashScopeRerankAdapter
from shared.config import settings
from shared.utils.model_provider import build_model_provider_timeout


def _generation_model_kwargs() -> dict:
    """Build provider-specific thinking arguments for the generation model.

    mimo（OpenAI 兼容端点）默认开启隐性思考，检索问答不需要推理链，
    关闭后回答延迟约降为三分之一；default 模式不传该参数，跟随 provider。
    """
    if settings.llm_thinking_mode == "disabled":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def build_qa_agent_dependencies() -> QAAgentDependencies:
    """Build the production collaborators used by the QA orchestration facade."""

    cross_encoder_adapter = None
    if settings.qa_cross_encoder_mode == "remote":
        cross_encoder_adapter = DashScopeRerankAdapter(
            endpoint=settings.qa_cross_encoder_remote_endpoint,
            api_key=settings.dashscope_api_key,
            model=settings.qa_cross_encoder_model,
            timeout_seconds=settings.qa_cross_encoder_timeout_seconds,
        )

    return QAAgentDependencies(
        llm=ChatOpenAI(
            model=settings.deepseek_model,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0,
            timeout=build_model_provider_timeout(settings),
            max_retries=settings.llm_max_retries,
            **_generation_model_kwargs(),
        ),
        safety_pipeline=build_runtime_safety_pipeline(
            mode=settings.qa_safety_mode,
            timeout_seconds=settings.qa_guardrail_timeout_seconds,
            rule_version=settings.qa_guardrail_rule_version,
        ),
        cross_encoder_reranker=SafeCrossEncoderReranker(
            mode=settings.qa_cross_encoder_mode,
            adapter=cross_encoder_adapter,
            candidate_limit=settings.qa_cross_encoder_candidate_limit,
            context_limit=settings.qa_cross_encoder_context_limit,
            timeout_seconds=settings.qa_cross_encoder_timeout_seconds,
            max_per_document=settings.qa_rerank_max_per_document,
        ),
        evidence_qualifier=EvidenceQualifier(
            EvidenceQualificationPolicy(
                background_threshold=settings.qa_evidence_background_threshold,
                gray_zone_lower=settings.qa_evidence_gray_zone_lower,
                gray_zone_upper=settings.qa_evidence_gray_zone_upper,
                direct_threshold=settings.qa_evidence_direct_threshold,
                candidate_limit=settings.qa_evidence_candidate_limit,
                policy_version=settings.qa_evidence_policy_version,
                calibration_version=settings.qa_evidence_calibration_version,
            )
        ),
        grounding_verifier=GroundingVerifier(
            policy_version=settings.qa_grounding_policy_version
        ),
    )


__all__ = ["build_qa_agent_dependencies"]
