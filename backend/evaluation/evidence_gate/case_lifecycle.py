"""案例校验、CRUD、maker-checker 流转与批量操作（mixin）。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from evaluation.evidence_gate.benchmark import (
    REQUIRED_EVIDENCE_GATE_CATEGORIES,
    SUPPORTED_EVIDENCE_STATES,
    SUPPORTED_RESPONSE_STATUSES,
    reviewed_case_contract_errors,
)
from evaluation.evidence_gate.workspace_io import (  # noqa: F401
    _CASE_ID,
    _CASE_STATUSES,
    _CURRENT_CORPUS_TEMPLATE_IMPORT_LIMIT,
    _CURRENT_CORPUS_TEMPLATE_STANDARD_CATEGORIES,
    _DECISIONS,
    _ID,
    _ORG_NAMESPACE,
    _REFERENCE_ANSWER_CONTRACT_FIELDS,
    _REFERENCE_ANSWER_CANDIDATE_SCHEMA,
    _REFERENCE_ANSWER_CANDIDATE_EDITOR,
    _REFERENCE_ANSWER_CATEGORIES,
    _REQUIRED_FIXTURES,
    _SOURCE_EVALUATION_DATASETS,
    _SOURCE_TITLE_PREFIX,
    _AUTHORIZATION_FIXTURES,
    _TENANT_ID,
    _AUTHORING_FIELDS,
    EvidenceReviewConflict,
    EvidenceReviewError,
    EvidenceReviewNotFound,
    EvidenceReviewPermissionError,
    EvidenceReviewValidationError,
    _read_json,
    _read_jsonl,
    _string_list,
    _utc_now,
    _write_json,
    _write_jsonl,
    sha256_file,
)


class CaseLifecycleMixin:

    @staticmethod
    def _filter_cases(
        cases: list[dict[str, Any]],
        *,
        category: str | None = None,
        review_status: str | None = None,
        query: str | None = None,
        reviewer_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """按列表页与批量过滤共享的匹配语义筛选用例。

        Args:
            cases: 待筛选的用例列表。
            category: 按类目精确匹配；None 表示不过滤。
            review_status: 按评审状态精确匹配；None 表示不过滤。
            query: 对用例 ID 与问题文本做小写包含匹配；None 表示不过滤。
            reviewer_id: 按已指派复核人精确匹配；None 表示不过滤。
        """
        selected = cases
        if category:
            selected = [case for case in selected if case.get("category") == category]
        if review_status:
            selected = [
                case for case in selected if case.get("review_status") == review_status
            ]
        if reviewer_id is not None:
            selected = [
                case for case in selected if case.get("reviewer_id") == reviewer_id
            ]
        if query:
            token = query.strip().lower()
            selected = [
                case
                for case in selected
                if token in str(case.get("id") or "").lower()
                or token in str(case.get("question") or "").lower()
            ]
        return selected

    def list_cases(
        self,
        org_id: str,
        dataset_id: str,
        *,
        page: int,
        page_size: int,
        category: str | None = None,
        review_status: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        """分页返回组织隔离的用例列表（每条附受限上下文摘要）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            page: 页码，从 1 开始。
            page_size: 每页数量。
            category: 可选的类目过滤。
            review_status: 可选的评审状态过滤。
            query: 可选的 ID/问题关键字过滤。
        """
        paths, workspace, cases = self._load(org_id, dataset_id)
        contexts = self._context_map(paths)
        selected = self._filter_cases(
            cases,
            category=category,
            review_status=review_status,
            query=query,
        )
        # 按 1 起始页码换算偏移并截取当前页
        start = (page - 1) * page_size
        return {
            "dataset_id": dataset_id,
            "revision": workspace["revision"],
            "page": page,
            "page_size": page_size,
            "total": len(selected),
            "cases": [
                self._decorate_case(case, contexts)
                for case in selected[start : start + page_size]
            ],
        }

    def _validate_case(
        self,
        value: Mapping[str, Any],
        *,
        contexts: dict[str, dict[str, Any]],
        existing_id: str | None = None,
    ) -> dict[str, Any]:
        """校验并归一化用例输入，剥离治理字段后返回干净的用例副本。

        Args:
            value: 客户端提交的用例原始数据。
            contexts: context_id 到上下文记录的索引，用于校验引用存在性。
            existing_id: 更新场景下的既有用例 ID；提供时用例 ID 必须与其一致。
        """
        # ① 用例 ID 必须符合格式；更新场景下还要求 ID 不可变
        case_id = str(value.get("id") or existing_id or "").strip()
        if not _CASE_ID.fullmatch(case_id):
            raise EvidenceReviewValidationError("invalid_case_id")
        if existing_id is not None and case_id != existing_id:
            raise EvidenceReviewValidationError("case_id_immutable")
        # ② 问题必填且不超过 8000 字符
        question = str(value.get("question") or "").strip()
        if not question or len(question) > 8000:
            raise EvidenceReviewValidationError("invalid_question")
        reference_answer = str(value.get("reference_answer") or "").strip()
        if len(reference_answer) > 8000:
            raise EvidenceReviewValidationError("invalid_reference_answer")
        # ③ 类目、期望响应状态与证据状态必须属于受控枚举
        category = str(value.get("category") or "")
        if category not in REQUIRED_EVIDENCE_GATE_CATEGORIES:
            raise EvidenceReviewValidationError("invalid_category")
        response_status = str(value.get("expected_response_status") or "")
        if response_status not in SUPPORTED_RESPONSE_STATUSES:
            raise EvidenceReviewValidationError("invalid_response_status")
        states = _string_list(
            value.get("expected_evidence_states"),
            "expected_evidence_states",
            allow_empty=False,
        )
        if set(states) - SUPPORTED_EVIDENCE_STATES:
            raise EvidenceReviewValidationError("invalid_evidence_state")
        # ④ 证据/引用上下文必须都能在工作区目录中找到
        evidence_ids = _string_list(
            value.get("expected_evidence_context_ids") or [],
            "expected_evidence_context_ids",
        )
        citation_ids = _string_list(
            value.get("expected_citation_context_ids") or [],
            "expected_citation_context_ids",
        )
        for context_id in evidence_ids + citation_ids:
            if context_id not in contexts:
                raise EvidenceReviewValidationError(f"unknown_context_id:{context_id}")
        # ⑤ Fixture 名称只允许 standard 或必需集合成员
        fixture = str(value.get("required_fixture") or "standard")
        if fixture != "standard" and fixture not in _REQUIRED_FIXTURES:
            raise EvidenceReviewValidationError("invalid_fixture")
        # ⑥ 安全关卡：剥离客户端不可写的治理字段，再由服务端重置受控字段
        clean = {
            key: item
            for key, item in dict(value).items()
            if key not in _AUTHORING_FIELDS
        }
        clean.update(
            {
                "id": case_id,
                "question": question,
                "reference_answer": reference_answer,
                "category": category,
                "expected_response_status": response_status,
                "expected_evidence_states": states,
                "expected_reason_codes": _string_list(
                    value.get("expected_reason_codes") or [],
                    "expected_reason_codes",
                ),
                "expected_source_document_ids": _string_list(
                    value.get("expected_source_document_ids") or [],
                    "expected_source_document_ids",
                ),
                "expected_evidence_context_ids": evidence_ids,
                "expected_evidence_sections": _string_list(
                    value.get("expected_evidence_sections") or [],
                    "expected_evidence_sections",
                ),
                "expected_citation_context_ids": citation_ids,
                "expected_missing_information_fields": _string_list(
                    value.get("expected_missing_information_fields") or [],
                    "expected_missing_information_fields",
                ),
                "source_benchmark_ids": _string_list(
                    value.get("source_benchmark_ids") or [],
                    "source_benchmark_ids",
                ),
                "required_fixture": fixture,
                "notes": str(value.get("notes") or "")[:2000],
            }
        )
        # ⑦ 分支可用性只保留 dense/bm25/graph 三个受控键，缺省为未配置
        branches = value.get("expected_branch_availability") or {}
        if not isinstance(branches, dict):
            raise EvidenceReviewValidationError("invalid_branch_availability")
        clean["expected_branch_availability"] = {
            branch: str(branches.get(branch) or "not_configured")
            for branch in ("dense", "bm25", "graph")
        }
        return clean

    @staticmethod
    def _current_corpus_fixture_category(fixture: str) -> str:
        """返回 Fixture 固定对应的评测类目，未知 Fixture 返回空字符串。

        Args:
            fixture: Fixture 名称。
        """
        # Fixture 与类目的一一对应关系，供用例/冻结一致性校验使用
        categories = {
            "frozen_corpus_absence_check": "completely_unanswerable",
            "equal_authority_conflicting_documents": "conflicting",
            "finance_only_user_against_hr_document": "authorization_filtered",
            "finance_only_user_against_administration_document": "authorization_filtered",
            "bm25_unavailable_dense_graph_available": "single_branch_unavailable",
            "all_retrieval_branches_unavailable": "all_branches_unavailable",
            "prompt_injection_safety_fixture": "prompt_injection",
        }
        return categories.get(fixture, "")

    def _restore_derived_current_corpus_fixture_bindings(
        self,
        workspace: Mapping[str, Any],
        profile: Mapping[str, Any],
        cases: list[dict[str, Any]],
        *,
        contexts: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], int]:
        """仅依据已复核 Fixture 用例恢复被省略的旧版源绑定。

        早期当前语料冻结版本只保存不含源绑定的公开受限档案。派生草稿
        不能信任浏览器输入、也不得覆盖非空档案，因此核对当前与冻结
        目录元数据后，只为每个 Fixture 恢复唯一无歧义的已复核源清单。

        Args:
            workspace: 工作区元数据。
            profile: 待恢复的 Fixture 档案。
            cases: 冻结版本中的用例列表。
            contexts: context_id 到上下文记录的索引。
        """

        # 非当前语料来源或档案结构非法时不做恢复，原样返回
        if workspace.get("source_type") != "current_corpus":
            return dict(profile), 0
        raw_fixtures = profile.get("fixtures")
        if not isinstance(raw_fixtures, Mapping):
            return dict(profile), 0
        # 当前/冻结目录中的“源文档 -> 部门”索引
        source_departments = {
            str(item.get("source_document_id") or ""): str(item.get("department") or "")
            for item in contexts.values()
            if str(item.get("source_document_id") or "")
        }
        normalized: dict[str, dict[str, Any]] = {}
        restored_count = 0
        for fixture in sorted(_REQUIRED_FIXTURES):
            item = raw_fixtures.get(fixture)
            raw_ids = item.get("source_document_ids") if isinstance(item, Mapping) else None
            if raw_ids is None:
                # 旧版受限冻结档案：刻意省略了仅服务端可见的源绑定，可以修复
                bound_ids = []
                valid_bound_ids = True
            elif isinstance(raw_ids, list):
                bound_ids = [
                    str(value).strip()
                    for value in raw_ids
                    if isinstance(value, str) and value.strip()
                ]
                valid_bound_ids = (
                    len(bound_ids) == len(raw_ids)
                    and len(bound_ids) == len(set(bound_ids))
                )
            else:
                bound_ids = []
                valid_bound_ids = False
            fixture_cases = [
                case
                for case in cases
                if str(case.get("required_fixture") or "") == fixture
            ]
            candidate_lists = {
                tuple(str(value) for value in case.get("expected_source_document_ids") or [])
                for case in fixture_cases
            }
            # 仅当该 Fixture 的用例源清单唯一、类目匹配且部门一致时才恢复绑定
            if not bound_ids and valid_bound_ids and len(candidate_lists) == 1:
                candidate = list(next(iter(candidate_lists)))
                category_matches = all(
                    str(case.get("category") or "")
                    == self._current_corpus_fixture_category(fixture)
                    for case in fixture_cases
                )
                source_matches = bool(candidate) and all(
                    source_departments.get(source_id)
                    == str(fixture_cases[0].get("department_id") or "")
                    for source_id in candidate
                )
                if category_matches and source_matches:
                    bound_ids = candidate
                    restored_count += 1
            normalized[fixture] = {
                "verified": False,
                "tenant_id": "",
                "source_document_ids": bound_ids,
                "source_document_count": len(bound_ids),
            }
        return {
            "schema_version": "evidence-gate-fixture-profile-v2",
            "status": "server_controlled_validation_required",
            "fixtures": normalized,
        }, restored_count

    @staticmethod
    def _frozen_current_corpus_fixture_profile(
        profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        """生成冻结档案：只保留派生未来当前语料所必需的源绑定。

        Args:
            profile: 待冻结的原始 Fixture 档案。
        """

        raw_fixtures = profile.get("fixtures")
        if not isinstance(raw_fixtures, Mapping):
            raise EvidenceReviewValidationError("invalid_fixture_profile")
        fixtures: dict[str, dict[str, Any]] = {}
        for fixture in sorted(_REQUIRED_FIXTURES):
            item = raw_fixtures.get(fixture)
            raw_ids = item.get("source_document_ids") if isinstance(item, Mapping) else None
            if not isinstance(raw_ids, list):
                raise EvidenceReviewValidationError("invalid_fixture")
            # 源 ID 必须是非空、无重复的字符串，且与原始列表一一对应
            source_ids = [
                str(value).strip()
                for value in raw_ids
                if isinstance(value, str) and value.strip()
            ]
            if len(source_ids) != len(raw_ids) or len(source_ids) != len(set(source_ids)):
                raise EvidenceReviewValidationError("invalid_fixture")
            fixtures[fixture] = {
                "source_document_ids": source_ids,
                "source_document_count": len(source_ids),
            }
        return {
            "schema_version": "evidence-gate-fixture-profile-v2",
            "status": "server_controlled_validation_required",
            "fixtures": fixtures,
        }

    def _validate_current_corpus_case(
        self,
        workspace: Mapping[str, Any],
        paths: dict[str, Path],
        case: Mapping[str, Any],
        *,
        contexts: Mapping[str, Mapping[str, Any]],
        fixture_profile: Mapping[str, Any] | None = None,
    ) -> None:
        """限制编写的用例只能引用已复核当前语料源快照中的文档。

        Args:
            workspace: 工作区元数据。
            paths: 工作区文件路径表。
            case: 待校验的用例。
            contexts: context_id 到上下文记录的索引。
            fixture_profile: 可选的 Fixture 档案；缺省时从磁盘读取。
        """

        # 仅当前语料来源的工作区才执行本约束
        if workspace.get("source_type") != "current_corpus":
            return
        if errors := reviewed_case_contract_errors(case):
            raise EvidenceReviewValidationError(errors[0])
        source_departments = {
            str(item.get("source_document_id") or ""): str(item.get("department") or "")
            for item in contexts.values()
            if str(item.get("source_document_id") or "")
        }
        # ① 引用的源文档必须都在已审阅快照中，且部门与用例部门一致
        source_ids = [str(value) for value in case.get("expected_source_document_ids") or []]
        department_id = str(case.get("department_id") or "")
        if any(value not in source_departments for value in source_ids):
            raise EvidenceReviewValidationError("current_corpus_source_document_unknown")
        if any(source_departments[value] != department_id for value in source_ids):
            raise EvidenceReviewValidationError("current_corpus_source_department_mismatch")
        fixture = str(case.get("required_fixture") or "standard")
        # ② 标准用例必须显式绑定至少一个源文档
        if fixture == "standard":
            if not source_ids:
                raise EvidenceReviewValidationError("current_corpus_standard_source_required")
            return
        # ③ Fixture 用例：源清单必须与档案绑定完全一致，类目也须匹配
        profile = fixture_profile or _read_json(paths["fixtures"])
        fixture_values = profile.get("fixtures")
        bound = fixture_values.get(fixture) if isinstance(fixture_values, Mapping) else None
        if not isinstance(bound, Mapping):
            raise EvidenceReviewValidationError("current_corpus_fixture_unknown")
        expected_sources = [
            str(value) for value in bound.get("source_document_ids", []) if isinstance(value, str)
        ]
        if source_ids != expected_sources:
            raise EvidenceReviewValidationError("current_corpus_fixture_source_mismatch")
        if str(case.get("category") or "") != self._current_corpus_fixture_category(fixture):
            raise EvidenceReviewValidationError("current_corpus_fixture_category_mismatch")

    def _validate_current_corpus_fixture_coverage(
        self,
        workspace: Mapping[str, Any],
        paths: dict[str, Path],
        cases: list[dict[str, Any]],
        *,
        contexts: Mapping[str, Mapping[str, Any]],
        fixture_profile: Mapping[str, Any] | None = None,
    ) -> None:
        """冻结前校验当前语料工作区恰好覆盖全部必需 Fixture 且绑定一致。

        Args:
            workspace: 工作区元数据。
            paths: 工作区文件路径表。
            cases: 待冻结的用例列表。
            contexts: context_id 到上下文记录的索引。
            fixture_profile: 可选的 Fixture 档案；缺省时从磁盘读取。
        """
        if workspace.get("source_type") != "current_corpus":
            return
        # 用例中出现的 Fixture 必须恰好等于必需集合，随后逐条复核绑定与类目
        fixtures = {
            str(case.get("required_fixture") or "")
            for case in cases
            if str(case.get("required_fixture") or "") in _REQUIRED_FIXTURES
        }
        if fixtures != _REQUIRED_FIXTURES:
            raise EvidenceReviewConflict("current_corpus_fixture_cases_incomplete")
        for case in cases:
            self._validate_current_corpus_case(
                workspace,
                paths,
                case,
                contexts=contexts,
                fixture_profile=fixture_profile,
            )

    def _load_reference_answer_candidates(self) -> dict[str, dict[str, Any]]:
        """Load document-grounded answer candidates and verify their source files.

        Candidate answers deliberately live outside frozen evaluation bundles.  A
        change to one normalized source document therefore invalidates its
        candidate before it can be copied into an editable workspace, instead of
        silently reusing an answer written against older text.
        """

        if (
            not self.reference_answer_candidates_path.is_file()
            or not self.company_demo_manifest_path.is_file()
        ):
            raise EvidenceReviewValidationError("reference_answer_candidate_catalog_unavailable")
        try:
            catalog = _read_json(self.reference_answer_candidates_path)
            manifest = _read_json(self.company_demo_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise EvidenceReviewValidationError("reference_answer_candidate_catalog_invalid") from error
        if catalog.get("schema_version") != _REFERENCE_ANSWER_CANDIDATE_SCHEMA:
            raise EvidenceReviewValidationError("reference_answer_candidate_catalog_invalid")
        raw_candidates = catalog.get("candidates")
        raw_documents = manifest.get("documents")
        if not isinstance(raw_candidates, list) or not isinstance(raw_documents, list):
            raise EvidenceReviewValidationError("reference_answer_candidate_catalog_invalid")

        documents: dict[str, tuple[str, str]] = {}
        for item in raw_documents:
            if not isinstance(item, Mapping):
                continue
            document_id = str(item.get("source_document_id") or "").strip()
            normalized_path = str(item.get("normalized_path") or "").strip()
            normalized_sha256 = str(item.get("normalized_sha256") or "").strip()
            if (
                document_id
                and normalized_path
                and re.fullmatch(r"[0-9a-f]{64}", normalized_sha256)
            ):
                documents[document_id] = (normalized_path, normalized_sha256)

        candidates: dict[str, dict[str, Any]] = {}
        candidate_root = self.company_demo_manifest_path.parent.resolve()
        for raw in raw_candidates:
            if not isinstance(raw, Mapping):
                raise EvidenceReviewValidationError("reference_answer_candidate_catalog_invalid")
            candidate = dict(raw)
            case_id = str(candidate.get("case_id") or "").strip()
            answer = str(candidate.get("reference_answer") or "").strip()
            source_hashes = candidate.get("source_document_sha256")
            if (
                not _CASE_ID.fullmatch(case_id)
                or not answer
                or len(answer) > 8_000
                or not isinstance(source_hashes, Mapping)
                or case_id in candidates
            ):
                raise EvidenceReviewValidationError("reference_answer_candidate_catalog_invalid")
            for document_id, expected_hash in source_hashes.items():
                if (
                    not isinstance(document_id, str)
                    or not isinstance(expected_hash, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
                    or documents.get(document_id) is None
                ):
                    raise EvidenceReviewValidationError("reference_answer_candidate_catalog_invalid")
                normalized_path, manifest_hash = documents[document_id]
                if expected_hash != manifest_hash:
                    raise EvidenceReviewValidationError("reference_answer_candidate_source_changed")
                source_path = (candidate_root / normalized_path).resolve()
                try:
                    source_path.relative_to(candidate_root)
                except ValueError as error:
                    raise EvidenceReviewValidationError("reference_answer_candidate_catalog_invalid") from error
                if not source_path.is_file() or sha256_file(source_path) != expected_hash:
                    raise EvidenceReviewValidationError("reference_answer_candidate_source_changed")
            candidates[case_id] = candidate
        return candidates

    @staticmethod
    def _reference_answer_candidate_matches(
        case: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> bool:
        """Require a stable question, route and source-document contract match."""

        candidate_case_id = str(candidate.get("case_id") or "")
        case_id = str(case.get("id") or "")
        source_benchmark_ids = {
            str(value)
            for value in case.get("source_benchmark_ids") or []
            if isinstance(value, str) and value
        }
        # Current-corpus templates have their own deterministic workspace ID,
        # while retaining the immutable source benchmark ID from which their
        # question and route contract were derived.  Either identity is valid;
        # all of the contract fields below must still match exactly.
        if candidate_case_id != case_id and candidate_case_id not in source_benchmark_ids:
            return False
        for field in _REFERENCE_ANSWER_CONTRACT_FIELDS:
            if candidate.get(field) != case.get(field):
                return False
        source_hashes = candidate.get("source_document_sha256")
        source_ids = case.get("expected_source_document_ids") or []
        return (
            isinstance(source_hashes, Mapping)
            and set(source_hashes) == set(source_ids)
            and bool(source_ids)
        )

    @classmethod
    def _reference_answer_candidate_for_case(
        cls,
        case: Mapping[str, Any],
        candidates: Mapping[str, Mapping[str, Any]],
    ) -> Mapping[str, Any] | None:
        """Return the verified candidate for an exact or source-benchmark case ID."""

        case_id = str(case.get("id") or "")
        candidate_ids = [case_id]
        candidate_ids.extend(
            str(value)
            for value in case.get("source_benchmark_ids") or []
            if isinstance(value, str) and value
        )
        return next(
            (
                candidates[candidate_id]
                for candidate_id in dict.fromkeys(candidate_ids)
                if candidate_id in candidates
                and cls._reference_answer_candidate_matches(case, candidates[candidate_id])
            ),
            None,
        )

    def populate_reference_answer_candidates(
        self,
        org_id: str,
        dataset_id: str,
        *,
        expected_revision: int,
        actor_id: str,
    ) -> dict[str, Any]:
        """Copy verified document-grounded candidates into an editable workspace.

        This is intentionally a draft-only operation: it never writes a frozen
        version and resets every populated case to ``draft`` for a human
        reviewer.  Existing answers are left untouched so a reviewer can refine
        a candidate without a later refresh overwriting that decision.
        """

        if not isinstance(actor_id, str) or not actor_id.strip():
            raise EvidenceReviewValidationError("invalid_actor")
        candidates = self._load_reference_answer_candidates()
        with self._lock(org_id, dataset_id):
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            contexts = self._context_map(paths)
            now = _utc_now()
            populated_case_ids: list[str] = []
            skipped: list[dict[str, str]] = []
            for index, existing in enumerate(cases):
                case = dict(existing)
                if str(case.get("category") or "") not in _REFERENCE_ANSWER_CATEGORIES:
                    continue
                if str(case.get("reference_answer") or "").strip():
                    continue
                case_id = str(case.get("id") or "")
                candidate = self._reference_answer_candidate_for_case(case, candidates)
                if candidate is None:
                    candidate_ids = [case_id]
                    candidate_ids.extend(
                        str(value)
                        for value in case.get("source_benchmark_ids") or []
                        if isinstance(value, str) and value
                    )
                    has_candidate_identity = any(candidate_id in candidates for candidate_id in candidate_ids)
                    skipped.append({"case_id": str(case.get("id") or ""), "code": "reference_answer_candidate_missing"})
                    if has_candidate_identity:
                        skipped[-1]["code"] = "reference_answer_candidate_contract_mismatch"
                    continue
                updated = dict(case)
                updated["reference_answer"] = str(candidate["reference_answer"]).strip()
                updated = self._validate_case(
                    updated,
                    contexts=contexts,
                    existing_id=str(updated["id"]),
                )
                # Candidate generation must not change the department that
                # governs the subsequent human review.
                updated["department_id"] = str(case.get("department_id") or "")
                self._validate_current_corpus_case(
                    workspace,
                    paths,
                    updated,
                    contexts=contexts,
                )
                updated.update(
                    {
                        "review_status": "draft",
                        "case_revision": int(case.get("case_revision") or 0) + 1,
                        "last_editor_id": _REFERENCE_ANSWER_CANDIDATE_EDITOR,
                        "last_edited_at": now,
                        "submitted_at": None,
                        "reviewer_id": None,
                        "reviewer_department_id": None,
                        "reviewed_at": None,
                        "review_reason": "",
                        "rejection_reason": "",
                    }
                )
                cases[index] = updated
                populated_case_ids.append(str(updated["id"]))

            revision = int(workspace["revision"])
            if populated_case_ids:
                workspace["status"] = "authoring"
                revision = self._bump(workspace)
                self._persist(paths, workspace, cases)
                events = _read_jsonl(paths["events"])
                events.extend(
                    {
                        "event": "reference_answer_candidate_populated",
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "actor_id": actor_id,
                        "candidate_editor": _REFERENCE_ANSWER_CANDIDATE_EDITOR,
                        "revision": revision,
                        "recorded_at": now,
                    }
                    for case_id in populated_case_ids
                )
                _write_jsonl(paths["events"], events[-10_000:])
            return {
                "dataset_id": dataset_id,
                "revision": revision,
                "populated_case_ids": populated_case_ids,
                "skipped": skipped,
            }

    def _persist(
        self,
        paths: dict[str, Path],
        workspace: dict[str, Any],
        cases: list[dict[str, Any]],
    ) -> None:
        """把全量用例列表与工作区元数据原子写回磁盘。

        Args:
            paths: 工作区文件路径表。
            workspace: 工作区元数据。
            cases: 全量用例列表。
        """
        _write_jsonl(paths["cases"], cases)
        _write_json(paths["workspace"], workspace)

    def update_case(
        self,
        org_id: str,
        dataset_id: str,
        case_id: str,
        value: Mapping[str, Any],
        *,
        expected_revision: int,
        actor_id: str,
        actor_department_id: str,
        actor_is_organization_admin: bool = False,
    ) -> dict[str, Any]:
        """更新既有用例并重置为草稿态（部门越权会被拒绝）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            case_id: 待更新的用例 ID。
            value: 客户端提交的用例新内容。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 执行编辑的账号 ID。
            actor_department_id: 操作者部门 ID，用于部门范围校验。
            actor_is_organization_admin: 组织管理员可跨部门编辑。
        """
        with self._lock(org_id, dataset_id):
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            contexts = self._context_map(paths)
            index = next(
                (i for i, case in enumerate(cases) if case.get("id") == case_id), None
            )
            if index is None:
                raise EvidenceReviewNotFound("case_not_found")
            # 安全关卡：用例必须有部门归属；仅本部门成员或组织管理员可编辑
            existing_department = str(cases[index].get("department_id") or "")
            if not existing_department:
                raise EvidenceReviewValidationError("case_department_required")
            if (
                existing_department != actor_department_id
                and not actor_is_organization_admin
            ):
                raise EvidenceReviewPermissionError("department_scope_mismatch")
            clean = self._validate_case(value, contexts=contexts, existing_id=case_id)
            # 部门归属保持不变，随后执行当前语料源绑定校验
            clean["department_id"] = existing_department
            self._validate_current_corpus_case(workspace, paths, clean, contexts=contexts)
            now = _utc_now()
            # 更新即重置为草稿：递增用例版本并清除提交/复核状态
            clean.update(
                {
                    "review_status": "draft",
                    "case_revision": int(cases[index].get("case_revision") or 0) + 1,
                    "last_editor_id": actor_id,
                    "department_id": existing_department,
                    "last_edited_at": now,
                    "submitted_at": None,
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "review_reason": "",
                    "rejection_reason": "",
                }
            )
            cases[index] = clean
            workspace["status"] = "authoring"
            revision = self._bump(workspace)
            self._persist(paths, workspace, cases)
            return {
                "revision": revision,
                "case": self._decorate_case(clean, contexts),
            }

    def create_case(
        self,
        org_id: str,
        dataset_id: str,
        value: Mapping[str, Any],
        *,
        expected_revision: int,
        actor_id: str,
        actor_department_id: str,
    ) -> dict[str, Any]:
        """新建用例并归属到操作者部门，初始为草稿态。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            value: 客户端提交的用例内容。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 创建用例的账号 ID。
            actor_department_id: 操作者部门 ID，作为新用例的归属部门。
        """
        with self._lock(org_id, dataset_id):
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            contexts = self._context_map(paths)
            clean = self._validate_case(value, contexts=contexts)
            clean["department_id"] = actor_department_id
            self._validate_current_corpus_case(workspace, paths, clean, contexts=contexts)
            # 安全关卡：同一数据集内用例 ID 不允许重复
            if any(case.get("id") == clean["id"] for case in cases):
                raise EvidenceReviewConflict("duplicate_case_id")
            now = _utc_now()
            # 初始治理字段：草稿、用例版本 1、无任何复核记录
            clean.update(
                {
                    "review_status": "draft",
                    "case_revision": 1,
                    "last_editor_id": actor_id,
                    "department_id": actor_department_id,
                    "last_edited_at": now,
                    "submitted_at": None,
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "review_reason": "",
                    "rejection_reason": "",
                }
            )
            cases.append(clean)
            workspace["status"] = "authoring"
            revision = self._bump(workspace)
            self._persist(paths, workspace, cases)
            return {
                "revision": revision,
                "case": self._decorate_case(clean, contexts),
            }

    def delete_case(
        self,
        org_id: str,
        dataset_id: str,
        case_id: str,
        *,
        expected_revision: int,
        actor_id: str,
        actor_is_organization_admin: bool = False,
    ) -> dict[str, Any]:
        """删除一个编写态用例，历史冻结版本不受影响。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            case_id: 待删除的用例 ID。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 执行删除的账号 ID。
            actor_is_organization_admin: 删除仅限组织管理员执行。
        """

        with self._lock(org_id, dataset_id):
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            # 安全关卡：删除属于高权限操作，仅组织管理员可执行
            if not actor_is_organization_admin:
                raise EvidenceReviewPermissionError("organization_admin_required")
            index = next(
                (i for i, case in enumerate(cases) if case.get("id") == case_id), None
            )
            if index is None:
                raise EvidenceReviewNotFound("case_not_found")
            deleted = cases.pop(index)
            workspace["status"] = "authoring"
            revision = self._bump(workspace)
            self._persist(paths, workspace, cases)
            events = _read_jsonl(paths["events"])
            events.append(
                {
                    "event": "delete",
                    "dataset_id": dataset_id,
                    "case_id": case_id,
                    "actor_id": actor_id,
                    "department_id": str(deleted.get("department_id") or ""),
                    "revision": revision,
                    "recorded_at": _utc_now(),
                }
            )
            _write_jsonl(paths["events"], events[-10_000:])
            return {"revision": revision, "deleted_case_id": case_id}

    def _case_transition(
        self,
        org_id: str,
        dataset_id: str,
        case_id: str,
        *,
        expected_revision: int,
        actor_id: str,
        actor_department_id: str,
        transition: str,
        reviewer_id: str = "",
        actor_is_department_manager: bool = False,
        actor_is_organization_admin: bool = False,
        decision: str | None = None,
        reason: str = "",
    ) -> dict[str, Any]:
        """执行单个用例的提交或复核状态机流转。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            case_id: 目标用例 ID。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 执行流转的账号 ID。
            actor_department_id: 操作者部门 ID，用于部门范围校验。
            transition: 流转类型："submit" 提交复核或 "review" 复核决定。
            reviewer_id: 提交时指定的复核人 ID。
            actor_is_department_manager: 复核操作要求部门经理权限。
            actor_is_organization_admin: 组织管理员可跨部门操作。
            decision: 复核决定："approve" 或 "reject"。
            reason: 复核理由，复核操作必填。
        """
        with self._lock(org_id, dataset_id):
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            index = next(
                (i for i, case in enumerate(cases) if case.get("id") == case_id), None
            )
            if index is None:
                raise EvidenceReviewNotFound("case_not_found")
            case = dict(cases[index])
            now = _utc_now()
            if transition == "submit":
                # ① 提交：仅草稿可提交；部门范围受限；复核人必填且不得是
                # 最后编辑人（maker-checker 双人复核）
                if case.get("review_status") != "draft":
                    raise EvidenceReviewConflict("case_not_draft")
                existing_department = str(case.get("department_id") or "")
                if not existing_department:
                    raise EvidenceReviewValidationError("case_department_required")
                if (
                    existing_department != actor_department_id
                    and not actor_is_organization_admin
                ):
                    raise EvidenceReviewPermissionError("department_scope_mismatch")
                assigned_reviewer = reviewer_id.strip()
                if not assigned_reviewer:
                    raise EvidenceReviewValidationError("reviewer_required")
                if assigned_reviewer == case.get("last_editor_id"):
                    raise EvidenceReviewConflict("maker_checker_violation")
                case["review_status"] = "pending_review"
                case["reviewer_id"] = assigned_reviewer
                case["submitted_at"] = now
            else:
                # ② 复核：决定必须合法；仅待复核用例可复核；操作者必须是
                # 部门经理、被指派复核人本人且非最后编辑人；理由必填
                if decision not in _DECISIONS:
                    raise EvidenceReviewValidationError("invalid_review_decision")
                if case.get("review_status") != "pending_review":
                    raise EvidenceReviewConflict("case_not_pending_review")
                if not actor_is_department_manager:
                    raise EvidenceReviewPermissionError("department_manager_required")
                if (
                    actor_department_id != case.get("department_id")
                    and not actor_is_organization_admin
                ):
                    raise EvidenceReviewPermissionError("department_scope_mismatch")
                if actor_id == case.get("last_editor_id"):
                    raise EvidenceReviewConflict("maker_checker_violation")
                if actor_id != case.get("reviewer_id"):
                    raise EvidenceReviewPermissionError("assigned_reviewer_required")
                if not reason.strip():
                    raise EvidenceReviewValidationError("review_reason_required")
                # 通过则置 approved；驳回则退回草稿并记录驳回理由
                case["reviewed_at"] = now
                case["review_reason"] = reason.strip()[:1000]
                if decision == "approve":
                    case["review_status"] = "approved"
                    case["rejection_reason"] = ""
                else:
                    case["review_status"] = "draft"
                    case["rejection_reason"] = reason.strip()[:1000]
            cases[index] = case
            revision = self._bump(workspace)
            self._persist(paths, workspace, cases)
            events = _read_jsonl(paths["events"])
            events.append(
                {
                    "event": transition if transition == "submit" else decision,
                    "dataset_id": dataset_id,
                    "case_id": case_id,
                    "actor_id": actor_id,
                    "department_id": actor_department_id,
                    "revision": revision,
                    "recorded_at": now,
                }
            )
            _write_jsonl(paths["events"], events[-10_000:])
            return {
                "revision": revision,
                "case": self._decorate_case(case, self._context_map(paths)),
            }

    def submit_case(
        self,
        org_id: str,
        dataset_id: str,
        case_id: str,
        *,
        expected_revision: int,
        actor_id: str,
        actor_department_id: str,
        reviewer_id: str,
        actor_is_organization_admin: bool = False,
    ) -> dict[str, Any]:
        """把草稿用例提交给指定复核人（maker-checker 第一步）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            case_id: 目标用例 ID。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 提交人账号 ID。
            actor_department_id: 提交人部门 ID。
            reviewer_id: 指定的复核人 ID，不得与提交人相同。
            actor_is_organization_admin: 组织管理员可跨部门提交。
        """
        return self._case_transition(
            org_id,
            dataset_id,
            case_id,
            expected_revision=expected_revision,
            actor_id=actor_id,
            actor_department_id=actor_department_id,
            transition="submit",
            reviewer_id=reviewer_id,
            actor_is_organization_admin=actor_is_organization_admin,
        )

    def review_case(
        self,
        org_id: str,
        dataset_id: str,
        case_id: str,
        *,
        decision: str,
        reason: str,
        expected_revision: int,
        actor_id: str,
        actor_department_id: str,
        actor_is_department_manager: bool,
        actor_is_organization_admin: bool = False,
    ) -> dict[str, Any]:
        """对待复核用例作出通过或驳回决定（maker-checker 第二步）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            case_id: 目标用例 ID。
            decision: 复核决定："approve" 或 "reject"。
            reason: 复核理由，必填。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 复核人账号 ID。
            actor_department_id: 复核人部门 ID。
            actor_is_department_manager: 必须为部门经理。
            actor_is_organization_admin: 组织管理员可跨部门复核。
        """
        return self._case_transition(
            org_id,
            dataset_id,
            case_id,
            expected_revision=expected_revision,
            actor_id=actor_id,
            actor_department_id=actor_department_id,
            transition="review",
            actor_is_department_manager=actor_is_department_manager,
            actor_is_organization_admin=actor_is_organization_admin,
            decision=decision,
            reason=reason,
        )

    @staticmethod
    def _validate_bulk_selection(
        selection_mode: str,
        case_ids: list[str],
        *,
        review_status: str | None,
        query: str | None,
    ) -> None:
        """校验批量选择输入，使直连服务调用方也受同一受限 API 契约限制。

        Args:
            selection_mode: 选择模式："selected" 按 ID 列表或 "filtered" 按过滤条件。
            case_ids: 显式选择的用例 ID 列表（filtered 模式必须为空）。
            review_status: 过滤模式下的评审状态过滤。
            query: 过滤模式下的关键字过滤，最长 200 字符。
        """
        if selection_mode not in {"selected", "filtered"}:
            raise EvidenceReviewValidationError("invalid_selection_mode")
        if len(case_ids) > 10_000:
            raise EvidenceReviewValidationError("too_many_case_ids")
        if len(case_ids) != len(set(case_ids)):
            raise EvidenceReviewValidationError("duplicate_case_id")
        if any(not _CASE_ID.fullmatch(case_id) for case_id in case_ids):
            raise EvidenceReviewValidationError("invalid_case_id")
        if selection_mode == "selected" and not case_ids:
            raise EvidenceReviewValidationError("selected_cases_required")
        if selection_mode == "filtered" and case_ids:
            raise EvidenceReviewValidationError("ambiguous_batch_selection")
        if review_status and review_status not in _CASE_STATUSES:
            raise EvidenceReviewValidationError("invalid_review_status")
        if query is not None and len(query) > 200:
            raise EvidenceReviewValidationError("query_too_long")

    @staticmethod
    def _bulk_skip_code(
        case: dict[str, Any],
        *,
        transition: str,
        actor_id: str,
        actor_department_id: str,
        actor_is_department_manager: bool,
        actor_is_organization_admin: bool,
        reviewer_id: str | None = None,
    ) -> str | None:
        """返回单个用例无法执行批量流转时的受限原因码；可流转则返回 None。

        Args:
            case: 待检查的用例。
            transition: 流转类型："submit" 或 "approve"。
            actor_id: 执行操作的账号 ID。
            actor_department_id: 操作者部门 ID。
            actor_is_department_manager: 操作者是否部门经理。
            actor_is_organization_admin: 操作者是否组织管理员。
            reviewer_id: 批量提交时指定的复核人；缺省回退为操作者本人。
        """
        # ① 状态前置：提交要求草稿，复核要求待复核
        expected_status = "draft" if transition == "submit" else "pending_review"
        if case.get("review_status") != expected_status:
            return "case_not_draft" if transition == "submit" else "case_not_pending_review"
        # ② 部门与权限：必须有部门归属；需部门经理或组织管理员且部门范围一致
        department_id = str(case.get("department_id") or "")
        if not department_id:
            return "case_department_required"
        if not actor_is_organization_admin and not actor_is_department_manager:
            return "department_manager_required"
        if not actor_is_organization_admin and department_id != actor_department_id:
            return "department_scope_mismatch"
        # ③ maker-checker：复核人不得是最后编辑人；复核人必须是被指派人本人
        if transition == "submit":
            assigned_reviewer = reviewer_id or actor_id
            if assigned_reviewer == case.get("last_editor_id"):
                return "maker_checker_violation"
            return None
        if actor_id == case.get("last_editor_id"):
            return "maker_checker_violation"
        if actor_id != case.get("reviewer_id"):
            return "assigned_reviewer_required"
        return None

    def _bulk_transition(
        self,
        org_id: str,
        dataset_id: str,
        *,
        transition: str,
        expected_revision: int,
        selection_mode: str,
        case_ids: list[str],
        actor_id: str,
        actor_department_id: str,
        actor_is_department_manager: bool,
        actor_is_organization_admin: bool = False,
        category: str | None = None,
        review_status: str | None = None,
        query: str | None = None,
        reviewer_id: str | None = None,
        reviewer_id_filter: str | None = None,
        require_current_corpus: bool = False,
    ) -> dict[str, Any]:
        """执行一次安全的部分成功批量流转，并一次性持久化成功的用例。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            transition: 批量流转类型："submit" 或 "approve"。
            expected_revision: 调用方持有的期望 revision。
            selection_mode: 选择模式："selected" 按 ID 列表或 "filtered" 按过滤条件。
            case_ids: 显式选择的用例 ID 列表。
            actor_id: 执行操作的账号 ID。
            actor_department_id: 操作者部门 ID。
            actor_is_department_manager: 操作者是否部门经理。
            actor_is_organization_admin: 组织管理员可跨部门操作。
            category: 过滤模式下的类目过滤。
            review_status: 过滤模式下的评审状态过滤。
            query: 过滤模式下的关键字过滤。
            reviewer_id: 批量提交时指定的复核人。
            reviewer_id_filter: 过滤模式下的已指派复核人过滤。
            require_current_corpus: 为 True 时仅允许当前语料数据集执行。
        """
        # ① 仅允许提交/通过两种批量流转，复核人 ID 需通过格式校验
        if transition not in {"submit", "approve"}:
            raise EvidenceReviewValidationError("invalid_batch_transition")
        if reviewer_id is not None and (
            not isinstance(reviewer_id, str)
            or not reviewer_id.strip()
            or len(reviewer_id.strip()) > 128
        ):
            raise EvidenceReviewValidationError("invalid_reviewer_id")
        self._validate_bulk_selection(
            selection_mode,
            case_ids,
            review_status=review_status,
            query=query,
        )
        with self._lock(org_id, dataset_id):
            # ② revision、编写态与（可选的）当前语料来源校验
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            if require_current_corpus and workspace.get("source_type") != "current_corpus":
                raise EvidenceReviewValidationError("current_corpus_dataset_required")
            indexed = {
                str(case.get("id")): (index, case)
                for index, case in enumerate(cases)
            }
            # ③ selected 模式保留未命中 ID 便于逐条报告；filtered 模式复用列表过滤语义
            if selection_mode == "selected":
                targets: list[tuple[int | None, dict[str, Any]]] = [
                    indexed.get(case_id, (None, {"id": case_id}))
                    for case_id in case_ids
                ]
            else:
                filtered = self._filter_cases(
                    cases,
                    category=category,
                    review_status=review_status,
                    query=query,
                    reviewer_id=reviewer_id_filter,
                )
                targets = [indexed[str(case["id"])] for case in filtered]

            now = _utc_now()
            processed_case_ids: list[str] = []
            skipped_items: list[dict[str, str]] = []
            # ④ 逐条预检：不可流转的记为 skipped，可流转的就地更新状态
            for index, source_case in targets:
                case_id = str(source_case.get("id") or "")
                if index is None:
                    skipped_items.append({"case_id": case_id, "code": "case_not_found"})
                    continue
                case = dict(source_case)
                skip_code = self._bulk_skip_code(
                    case,
                    transition=transition,
                    actor_id=actor_id,
                    actor_department_id=actor_department_id,
                    actor_is_department_manager=actor_is_department_manager,
                    actor_is_organization_admin=actor_is_organization_admin,
                    reviewer_id=reviewer_id,
                )
                if skip_code:
                    skipped_items.append({"case_id": case_id, "code": skip_code})
                    continue
                if transition == "submit":
                    case["review_status"] = "pending_review"
                    case["reviewer_id"] = reviewer_id or actor_id
                    case["submitted_at"] = now
                else:
                    case["review_status"] = "approved"
                    case["reviewed_at"] = now
                    case["review_reason"] = "批量复核通过"
                    case["rejection_reason"] = ""
                cases[index] = case
                processed_case_ids.append(case_id)

            # ⑤ 有成功项才递增 revision 并一次性持久化，事件按用例逐条追加
            revision = int(workspace["revision"])
            if processed_case_ids:
                workspace["status"] = "authoring"
                revision = self._bump(workspace)
                self._persist(paths, workspace, cases)
                events = _read_jsonl(paths["events"])
                events.extend(
                    {
                        "event": transition,
                        "dataset_id": dataset_id,
                        "case_id": case_id,
                        "actor_id": actor_id,
                        "department_id": actor_department_id,
                        "reviewer_id": reviewer_id or actor_id if transition == "submit" else "",
                        "revision": revision,
                        "recorded_at": now,
                    }
                    for case_id in processed_case_ids
                )
                _write_jsonl(paths["events"], events[-10_000:])
            return {
                "revision": revision,
                "matched_count": len(targets),
                "processed_count": len(processed_case_ids),
                "skipped_count": len(skipped_items),
                "processed_case_ids": processed_case_ids,
                "skipped_items": skipped_items,
            }

    def bulk_submit_cases(
        self,
        org_id: str,
        dataset_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """把每个被接受的草稿批量提交并指派复核人（缺省为当前认证账号）。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            **kwargs: 透传给 _bulk_transition 的其余批量参数。
        """
        return self._bulk_transition(
            org_id,
            dataset_id,
            transition="submit",
            **kwargs,
        )

    def bulk_approve_cases(
        self,
        org_id: str,
        dataset_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """批量通过指派给当前认证复核人的全部待复核用例。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            **kwargs: 透传给 _bulk_transition 的其余批量参数。
        """
        return self._bulk_transition(
            org_id,
            dataset_id,
            transition="approve",
            **kwargs,
        )

    def bulk_submit_current_corpus_cases(
        self,
        org_id: str,
        dataset_id: str,
        *,
        reviewer_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """把当前语料数据集的全部草稿提交给一位显式选择的同级管理员复核。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            reviewer_id: 显式指定的同级管理员复核人 ID。
            **kwargs: 透传给 _bulk_transition 的其余批量参数。
        """
        return self._bulk_transition(
            org_id,
            dataset_id,
            transition="submit",
            selection_mode="filtered",
            case_ids=[],
            review_status="draft",
            reviewer_id=reviewer_id,
            require_current_corpus=True,
            **kwargs,
        )

    def bulk_approve_current_corpus_cases(
        self,
        org_id: str,
        dataset_id: str,
        *,
        reviewer_id: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """仅通过指派给该同级管理员的当前语料待复核用例。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            reviewer_id: 同级管理员复核人 ID，作为待复核用例的过滤条件。
            **kwargs: 透传给 _bulk_transition 的其余批量参数。
        """
        return self._bulk_transition(
            org_id,
            dataset_id,
            transition="approve",
            selection_mode="filtered",
            case_ids=[],
            review_status="pending_review",
            reviewer_id_filter=reviewer_id,
            require_current_corpus=True,
            **kwargs,
        )
