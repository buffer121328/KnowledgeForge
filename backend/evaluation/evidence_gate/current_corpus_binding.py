"""现行语料工作区装订、上下文映射与代表性模板导入（mixin）。"""

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
)


class CurrentCorpusBindingMixin:
    def _current_corpus_contexts(source_documents: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """为新工作区生成仅含元数据、不可透视内容的源绑定。

        Args:
            source_documents: 已审阅快照中的源文档元数据列表。
        """

        contexts: list[dict[str, Any]] = []
        seen: set[str] = set()
        # ① 校验元数据完整性与长度上限，并拒绝重复文档
        for item in source_documents:
            document_id = str(item.get("document_id") or "").strip()
            source = str(item.get("source") or "").strip()
            department = str(item.get("department") or "").strip()
            if (
                not document_id
                or not source
                or not department
                or len(document_id) > 256
                or len(source) > 256
                or len(department) > 128
                or document_id in seen
            ):
                raise EvidenceReviewValidationError("current_corpus_source_metadata_invalid")
            # ② context_id 由 document_id 哈希派生，不含任何正文内容或哈希之外的原文
            contexts.append(
                {
                    "context_id": f"catalog-{hashlib.sha256(document_id.encode('utf-8')).hexdigest()[:24]}",
                    "source_document_id": document_id,
                    "title": source,
                    "department": department,
                    "document_version": int(item.get("version") or 1),
                    "chunk_index": 0,
                    "content_sha256": None,
                    "content_excerpt": "",
                }
            )
            seen.add(document_id)
        # ③ 空快照拒绝创建工作区
        if not contexts:
            raise EvidenceReviewValidationError("current_corpus_snapshot_empty")
        return contexts

    @staticmethod
    def _current_corpus_runtime_contexts(
        source_documents: list[Mapping[str, Any]],
        current_chunks: list[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Build the frozen Context catalog from authenticated runtime chunks."""

        documents = {
            str(item.get("document_id") or "").strip(): {
                "title": str(item.get("source") or "").strip(),
                "department": str(item.get("department") or "").strip(),
                "document_version": int(item.get("version") or 1),
            }
            for item in source_documents
            if str(item.get("document_id") or "").strip()
        }
        contexts: list[dict[str, Any]] = []
        seen_context_ids: set[str] = set()
        covered_documents: set[str] = set()
        for item in current_chunks:
            context_id = str(item.get("chunk_id") or "").strip()
            source_document_id = str(item.get("source_document_id") or "").strip()
            content = str(item.get("content") or "")
            document = documents.get(source_document_id)
            if (
                document is None
                or not context_id
                or context_id in seen_context_ids
                or not context_id.startswith(f"{source_document_id}#chunk-")
                or not content.strip()
            ):
                continue
            contexts.append(
                {
                    "context_id": context_id,
                    "source_document_id": source_document_id,
                    "title": document["title"],
                    "department": document["department"],
                    "document_version": document["document_version"],
                    "chunk_index": int(item.get("chunk_index") or 0),
                    "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    "content_excerpt": content[:1_000],
                }
            )
            seen_context_ids.add(context_id)
            covered_documents.add(source_document_id)
        if covered_documents != set(documents):
            raise EvidenceReviewValidationError("current_corpus_chunks_incomplete")
        return sorted(
            contexts,
            key=lambda item: (
                str(item["source_document_id"]),
                int(item["chunk_index"]),
                str(item["context_id"]),
            ),
        )

    def _refresh_current_corpus_case_bindings(
        self,
        cases: list[dict[str, Any]],
        contexts: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Restore reviewed smoke gold bindings onto the current real chunks."""

        source_bundle = self._source_bundle("evidence-gates-v1")
        source_cases = _read_jsonl(source_bundle / "cases.jsonl")
        source_contexts = {
            str(item.get("context_id") or ""): item
            for item in _read_jsonl(source_bundle / "context-catalog.jsonl")
        }
        templates: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for item in source_cases:
            key = (
                str(item.get("question") or "").strip(),
                str(item.get("category") or ""),
                str(item.get("required_fixture") or "standard"),
            )
            templates.setdefault(key, []).append(item)

        runtime_by_id = {str(item["context_id"]): item for item in contexts}
        runtime_by_hash: dict[str, list[dict[str, Any]]] = {}
        runtime_by_document: dict[str, list[dict[str, Any]]] = {}
        current_documents_by_key: dict[tuple[str, str], set[str]] = {}
        for item in contexts:
            digest = str(item.get("content_sha256") or "")
            source_document_id = str(item["source_document_id"])
            runtime_by_hash.setdefault(digest, []).append(item)
            runtime_by_document.setdefault(source_document_id, []).append(item)
            current_documents_by_key.setdefault(
                self._current_corpus_source_key(
                    str(item.get("title") or ""),
                    str(item.get("department") or ""),
                ),
                set(),
            ).add(source_document_id)

        source_document_metadata: dict[str, tuple[str, str]] = {}
        for item in source_contexts.values():
            source_document_id = str(item.get("source_document_id") or "")
            if source_document_id:
                source_document_metadata.setdefault(
                    source_document_id,
                    (str(item.get("title") or ""), str(item.get("department") or "")),
                )

        def map_document_id(source_document_id: str) -> str:
            if source_document_id in runtime_by_document:
                return source_document_id
            title, department = source_document_metadata.get(source_document_id, ("", ""))
            matches = current_documents_by_key.get(
                self._current_corpus_source_key(title, department),
                set(),
            )
            if len(matches) != 1:
                raise EvidenceReviewValidationError("current_corpus_source_unresolved")
            return next(iter(matches))

        def resolve_context(context_id: str) -> str:
            legacy = source_contexts.get(context_id)
            if not legacy:
                raise EvidenceReviewValidationError("current_corpus_context_unresolved")
            current_document_id = map_document_id(
                str(legacy.get("source_document_id") or "")
            )
            direct = runtime_by_id.get(context_id)
            candidates = (
                [direct]
                if direct and str(direct.get("source_document_id") or "") == current_document_id
                else []
            )
            if not candidates:
                candidates = [
                    item
                    for item in runtime_by_hash.get(str(legacy.get("content_sha256") or ""), [])
                    if str(item.get("source_document_id") or "") == current_document_id
                ]
            if not candidates:
                legacy_excerpt = "".join(
                    char
                    for char in str(legacy.get("content_excerpt") or "")
                    if char.isalnum()
                )
                if len(legacy_excerpt) >= 160:
                    candidates = [
                        item
                        for item in runtime_by_document.get(current_document_id, [])
                        if (
                            min(
                                len(legacy_excerpt),
                                len(
                                    "".join(
                                        char
                                        for char in str(item.get("content_excerpt") or "")
                                        if char.isalnum()
                                    )
                                ),
                            )
                            >= 160
                            and (
                                legacy_excerpt
                                in "".join(
                                    char
                                    for char in str(item.get("content_excerpt") or "")
                                    if char.isalnum()
                                )
                                or "".join(
                                    char
                                    for char in str(item.get("content_excerpt") or "")
                                    if char.isalnum()
                                )
                                in legacy_excerpt
                            )
                        )
                    ]
            if len(candidates) != 1:
                raise EvidenceReviewValidationError("current_corpus_context_unresolved")
            return str(candidates[0]["context_id"])

        rebound: list[dict[str, Any]] = []
        changed = 0
        for case in cases:
            next_case = dict(case)
            key = (
                str(case.get("question") or "").strip(),
                str(case.get("category") or ""),
                str(case.get("required_fixture") or "standard"),
            )
            matches = templates.get(key, [])
            if len(matches) != 1:
                raise EvidenceReviewValidationError("current_corpus_template_unresolved")
            template = matches[0]
            for field in (
                "expected_response_status",
                "expected_evidence_states",
                "expected_reason_codes",
                "expected_evidence_sections",
                "expected_missing_information_fields",
                "expected_branch_availability",
            ):
                value = template.get(field)
                next_case[field] = dict(value) if isinstance(value, dict) else list(value or []) if isinstance(value, list) else value
            next_case["source_benchmark_ids"] = [str(template["id"])]
            category = str(next_case.get("category") or "")
            if category in {
                "all_branches_unavailable",
                "completely_unanswerable",
                "prompt_injection",
            }:
                next_case["expected_evidence_context_ids"] = []
                next_case["expected_citation_context_ids"] = []
            else:
                next_case["expected_source_document_ids"] = [
                    map_document_id(str(value))
                    for value in template.get("expected_source_document_ids") or []
                ]
                if category == "authorization_filtered":
                    next_case["expected_evidence_context_ids"] = []
                    next_case["expected_citation_context_ids"] = []
                else:
                    for field in (
                        "expected_evidence_context_ids",
                        "expected_citation_context_ids",
                    ):
                        next_case[field] = [
                            resolve_context(str(value)) for value in template.get(field) or []
                        ]
            if next_case != case:
                changed += 1
            rebound.append(next_case)
        return rebound, changed

    @staticmethod
    def _sync_current_corpus_fixture_bindings(
        fixtures: dict[str, Any],
        cases: list[dict[str, Any]],
    ) -> int:
        """Make the editable Fixture profile match the refreshed reviewed cases."""

        values = fixtures.get("fixtures")
        if not isinstance(values, dict):
            raise EvidenceReviewValidationError("invalid_fixture_profile")
        changed = 0
        for case in cases:
            fixture = str(case.get("required_fixture") or "standard")
            if fixture == "standard":
                continue
            binding = values.get(fixture)
            if not isinstance(binding, dict):
                raise EvidenceReviewValidationError("invalid_fixture")
            source_ids = [
                str(value)
                for value in case.get("expected_source_document_ids") or []
            ]
            if (
                binding.get("source_document_ids") != source_ids
                or int(binding.get("source_document_count") or 0) != len(source_ids)
            ):
                binding["source_document_ids"] = source_ids
                binding["source_document_count"] = len(source_ids)
                changed += 1
        return changed

    @staticmethod
    def _current_corpus_fixture_profile(
        fixture_candidates: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """保存服务端复核过的 Fixture 源绑定，不含任何身份或正文文本。

        Args:
            fixture_candidates: 服务端复核产生的 Fixture 候选绑定列表。
        """

        expected = set(_REQUIRED_FIXTURES)
        values: dict[str, dict[str, Any]] = {}
        # ① 每个 Fixture 只能出现一次，且必须属于必需集合
        for item in fixture_candidates:
            fixture = str(item.get("fixture") or "")
            source_ids = item.get("source_document_ids")
            if fixture not in expected or not isinstance(source_ids, list) or fixture in values:
                raise EvidenceReviewValidationError("current_corpus_fixture_metadata_invalid")
            # ② 源 ID 必须是非空、无重复的字符串，且与原始列表一一对应
            clean_ids = [str(value).strip() for value in source_ids if isinstance(value, str) and value.strip()]
            if len(clean_ids) != len(source_ids) or len(clean_ids) != len(set(clean_ids)):
                raise EvidenceReviewValidationError("current_corpus_fixture_metadata_invalid")
            # 绑定初始为未核验、无租户，仅记录服务端控制的源文档清单
            values[fixture] = {
                "verified": False,
                "tenant_id": "",
                "source_document_ids": clean_ids,
                "source_document_count": len(clean_ids),
            }
        # ③ 必须恰好覆盖全部必需 Fixture
        if set(values) != expected:
            raise EvidenceReviewValidationError("current_corpus_fixture_metadata_incomplete")
        return {
            "schema_version": "evidence-gate-fixture-profile-v2",
            "status": "server_controlled_validation_required",
            "fixtures": values,
        }

    def create_current_corpus_workspace(
        self,
        org_id: str,
        dataset_id: str,
        *,
        draft_id: str,
        snapshot_sha256: str,
        source_documents: list[Mapping[str, Any]],
        fixture_candidates: list[Mapping[str, Any]],
        actor_id: str,
    ) -> dict[str, Any]:
        """从已审批的当前语料草稿创建一个来源不可变的工作区。

        新工作区刻意保持为空：管理员必须编写真实的问题/期望用例，
        并由第二位管理员复核；禁止导入演示数据包或生成虚构答案。

        Args:
            org_id: 组织 ID。
            dataset_id: 新数据集 ID（不得占用旧版 evidence-gates-v1）。
            draft_id: 已审批当前语料草稿的 ID（ccd_ 前缀 32 位十六进制）。
            snapshot_sha256: 草稿对应语料快照的 SHA-256（64 位十六进制）。
            source_documents: 已审阅快照中的源文档元数据列表。
            fixture_candidates: 服务端复核产生的 Fixture 候选绑定列表。
            actor_id: 执行创建操作的管理员 ID。
        """

        # ① 安全关卡：数据集 ID、草稿 ID、快照哈希与操作者逐项校验
        self._validate_dataset_id(dataset_id)
        if dataset_id == "evidence-gates-v1":
            raise EvidenceReviewValidationError("current_corpus_dataset_id_required")
        if not isinstance(draft_id, str) or not re.fullmatch(r"ccd_[a-f0-9]{32}", draft_id):
            raise EvidenceReviewValidationError("current_corpus_draft_id_invalid")
        if not isinstance(snapshot_sha256, str) or not re.fullmatch(r"[a-f0-9]{64}", snapshot_sha256):
            raise EvidenceReviewValidationError("current_corpus_snapshot_invalid")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise EvidenceReviewValidationError("invalid_actor")
        paths = self._paths(org_id, dataset_id)
        # ② 生成仅元数据的上下文绑定与服务端控制的 Fixture 档案
        contexts = self._current_corpus_contexts(source_documents)
        fixtures = self._current_corpus_fixture_profile(fixture_candidates)
        with self._lock(org_id, dataset_id):
            # ③ 幂等：同组织、同草稿且同快照的重复创建直接返回既有数据集
            if paths["workspace"].is_file():
                existing = _read_json(paths["workspace"])
                if (
                    existing.get("org_id_hash") == self._org_namespace(org_id)
                    and existing.get("source_type") == "current_corpus"
                    and existing.get("current_corpus_draft_id") == draft_id
                    and existing.get("current_corpus_snapshot_sha256") == snapshot_sha256
                ):
                    return self.get_dataset(org_id, dataset_id)
                raise EvidenceReviewConflict("authoring_dataset_already_exists")
            # ④ 首次创建：写入空用例列表与初始工作区元数据，来源锚定草稿与快照哈希
            now = _utc_now()
            workspace = {
                "schema_version": "evidence-review-workspace-v1",
                "dataset_id": dataset_id,
                "title": "当前语料评测集",
                "org_id_hash": self._org_namespace(org_id),
                "revision": 1,
                "status": "authoring",
                "created_at": now,
                "updated_at": now,
                "last_frozen_version": None,
                "last_frozen_manifest_sha256": None,
                "last_frozen_at": None,
                "last_frozen_by": None,
                "source_frozen_version": None,
                "source_type": "current_corpus",
                "current_corpus_draft_id": draft_id,
                "current_corpus_snapshot_sha256": snapshot_sha256,
                "source_document_count": len(contexts),
                "required_fixture_names": sorted(_REQUIRED_FIXTURES),
                "created_from_actor": actor_id,
            }
            _write_jsonl(paths["cases"], [])
            _write_jsonl(paths["contexts"], contexts)
            _write_json(paths["fixtures"], fixtures)
            _write_json(paths["workspace"], workspace)
            _write_jsonl(paths["events"], [])
        return self.get_dataset(org_id, dataset_id)

    def rebind_suite_to_current_chunks(
        self,
        org_id: str,
        dataset_id: str,
        *,
        expected_revision: int,
        actor_id: str,
        current_chunks: list[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Replace checked-in suite evidence with verified local chunks atomically.

        A checked-in suite may retain stale chunk identifiers after a local corpus
        rebuild. This operation accepts only tenant-scoped runtime chunks supplied
        by the API dependency, matches every retained positive binding by stable
        identity, immutable content hash, or one unique same-document containment,
        and writes nothing if even one mapping is absent or ambiguous.
        Fixture-only refusal routes deliberately lose their inherited evidence IDs.
        """

        if dataset_id not in _SOURCE_EVALUATION_DATASETS:
            raise EvidenceReviewValidationError("source_evaluation_dataset_required")
        if not isinstance(actor_id, str) or not actor_id.strip():
            raise EvidenceReviewValidationError("invalid_actor")

        with self._lock(org_id, dataset_id):
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)

            runtime_by_hash: dict[str, list[dict[str, Any]]] = {}
            runtime_by_document: dict[str, list[dict[str, Any]]] = {}
            runtime_by_context_id: dict[str, list[dict[str, Any]]] = {}
            for item in current_chunks:
                chunk_id = str(item.get("chunk_id") or "").strip()
                document_id = str(item.get("source_document_id") or "").strip()
                content = str(item.get("content") or "")
                title = str(item.get("title") or "").strip()
                department = str(item.get("department") or "").strip()
                if not all((chunk_id, document_id, content, title, department)):
                    continue
                if not chunk_id.startswith(f"{document_id}#chunk-"):
                    continue
                digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                candidate = {
                    "context_id": chunk_id,
                    "source_document_id": document_id,
                    "title": title[:256],
                    "department": department[:128],
                    "chunk_index": int(item.get("chunk_index") or 0),
                    "content_sha256": digest,
                    "content_excerpt": content[:1_000],
                    "_normalized_content": "".join(char for char in content if char.isalnum()),
                }
                runtime_by_hash.setdefault(digest, []).append(candidate)
                runtime_by_document.setdefault(document_id, []).append(candidate)
                runtime_by_context_id.setdefault(chunk_id, []).append(candidate)
            if not runtime_by_hash:
                raise EvidenceReviewValidationError("current_corpus_chunks_unavailable")

            legacy_contexts = self._context_map(paths)
            clear_categories = {
                "all_branches_unavailable",
                "authorization_filtered",
                "completely_unanswerable",
                "prompt_injection",
            }
            retained_context_ids = {
                str(context_id)
                for case in cases
                if str(case.get("category") or "") not in clear_categories
                for field in (
                    "expected_evidence_context_ids",
                    "expected_citation_context_ids",
                )
                for context_id in case.get(field) or []
                if str(context_id)
            }
            replacements: dict[str, dict[str, Any]] = {}
            for legacy_id in retained_context_ids:
                legacy = legacy_contexts.get(legacy_id)
                digest = str((legacy or {}).get("content_sha256") or "")
                source_document_id = str((legacy or {}).get("source_document_id") or "")
                # Stable ``document_id#chunk-index`` is the primary local identity.
                # Re-ingestion may alter a chunk's content hash while retaining that
                # identity, so its runtime metadata is authoritative in that case.
                candidates = [
                    item
                    for item in runtime_by_context_id.get(legacy_id, [])
                    if item["source_document_id"] == source_document_id
                ]
                if not candidates:
                    candidates = [
                        item for item in runtime_by_hash.get(digest, [])
                        if item["source_document_id"] == source_document_id
                    ]
                if not candidates:
                    # Chunk boundaries legitimately change after a parser rebuild.
                    # A fallback is safe only when a substantial, normalized legacy
                    # excerpt is wholly contained in exactly one current chunk (or
                    # vice versa) of the same immutable source document.
                    legacy_excerpt = "".join(
                        char for char in str((legacy or {}).get("content_excerpt") or "") if char.isalnum()
                    )
                    if len(legacy_excerpt) >= 160:
                        candidates = [
                            item
                            for item in runtime_by_document.get(source_document_id, [])
                            if min(len(legacy_excerpt), len(item["_normalized_content"])) >= 160
                            and (
                                legacy_excerpt in item["_normalized_content"]
                                or item["_normalized_content"] in legacy_excerpt
                            )
                        ]
                if len(candidates) != 1:
                    raise EvidenceReviewValidationError("current_corpus_context_unresolved")
                replacements[legacy_id] = candidates[0]

            rebound_cases: list[dict[str, Any]] = []
            for case in cases:
                next_case = dict(case)
                category = str(next_case.get("category") or "")
                if category in clear_categories:
                    next_case["expected_source_document_ids"] = []
                    next_case["expected_evidence_context_ids"] = []
                    next_case["expected_citation_context_ids"] = []
                else:
                    rebound_context_ids: list[str] = []
                    for field in (
                        "expected_evidence_context_ids",
                        "expected_citation_context_ids",
                    ):
                        ids = [str(value) for value in next_case.get(field) or []]
                        next_case[field] = [replacements[value]["context_id"] for value in ids]
                        rebound_context_ids.extend(next_case[field])
                    referenced_documents = {
                        item["source_document_id"]
                        for item in replacements.values()
                        if item["context_id"] in rebound_context_ids
                    }
                    expected_documents = [
                        str(value) for value in next_case.get("expected_source_document_ids") or []
                    ]
                    if expected_documents and not set(expected_documents).issubset(referenced_documents):
                        raise EvidenceReviewValidationError("current_corpus_source_unresolved")
                next_case.update(
                    {
                        "review_status": "draft",
                        "reviewer_id": None,
                        "reviewed_at": None,
                        "review_reason": "",
                        "rejection_reason": "",
                        "last_editor_id": actor_id,
                        "last_edited_at": _utc_now(),
                        "case_revision": int(next_case.get("case_revision") or 0) + 1,
                    }
                )
                rebound_cases.append(next_case)

            contexts = {
                item["context_id"]: {
                    key: value for key, value in item.items() if key != "_normalized_content"
                }
                for item in replacements.values()
            }
            self._bump(workspace)
            workspace["current_corpus_rebound_at"] = _utc_now()
            workspace["current_corpus_rebound_by"] = actor_id
            _write_jsonl(paths["contexts"], list(contexts.values()))
            _write_jsonl(paths["cases"], rebound_cases)
            _write_json(paths["workspace"], workspace)
        return self.get_dataset(org_id, dataset_id)

    def import_current_corpus_representative_templates(
        self,
        org_id: str,
        dataset_id: str,
        *,
        expected_revision: int,
        actor_id: str,
    ) -> dict[str, Any]:
        """仅从旧数据包创建安全重绑定、确定性的代表性草稿样本。

        这刻意是模板导入而非文档或冻结版本导入：旧文档与 Context 标识
        绝不进入当前语料工作区，下面所有源 ID 都来自已复核的元数据快照。

        Args:
            org_id: 组织 ID。
            dataset_id: 数据集 ID。
            expected_revision: 调用方持有的期望 revision。
            actor_id: 执行导入的管理员 ID。
        """

        with self._lock(org_id, dataset_id):
            # ① 安全关卡：revision、编写态、当前语料来源与操作者逐项校验
            paths, workspace, cases = self._load(org_id, dataset_id)
            self._check_revision(workspace, expected_revision)
            self._require_authoring(workspace)
            if workspace.get("source_type") != "current_corpus":
                raise EvidenceReviewConflict("current_corpus_template_import_not_supported")
            if not isinstance(actor_id, str) or not actor_id.strip():
                raise EvidenceReviewValidationError("invalid_actor")

            # ② 读取旧数据包；不可用时视为模板不可得，而非半途导入
            try:
                source_cases = _read_jsonl(self.source_bundle / "cases.jsonl")
                source_contexts = _read_jsonl(self.source_bundle / "context-catalog.jsonl")
            except (OSError, ValueError, json.JSONDecodeError) as error:
                raise EvidenceReviewValidationError("representative_templates_unavailable") from error

            # ③ 建立当前语料的“源文档 -> 部门”索引，作为重绑定的唯一依据
            contexts = self._context_map(paths)
            source_departments = {
                str(item.get("source_document_id") or ""): str(item.get("department") or "")
                for item in contexts.values()
                if str(item.get("source_document_id") or "")
                and str(item.get("department") or "")
            }
            if not source_departments:
                raise EvidenceReviewValidationError("current_corpus_snapshot_empty")

            # 旧包源索引：文档 ID -> (标题, 部门)；元数据冲突或缺失的文档标记为不可用
            legacy_sources: dict[str, tuple[str, str]] = {}
            invalid_legacy_sources: set[str] = set()
            for context in source_contexts:
                document_id = str(context.get("source_document_id") or "").strip()
                title = str(context.get("title") or "").strip()
                department = str(context.get("department") or "").strip()
                if not document_id:
                    continue
                if not title or not department:
                    invalid_legacy_sources.add(document_id)
                    continue
                existing = legacy_sources.get(document_id)
                if existing is not None and existing != (title, department):
                    invalid_legacy_sources.add(document_id)
                    continue
                legacy_sources[document_id] = (title, department)

            # 当前语料源索引：保守元数据键 -> 文档 ID 列表（要求唯一匹配）
            current_sources: dict[tuple[str, str], list[str]] = {}
            for context in contexts.values():
                document_id = str(context.get("source_document_id") or "").strip()
                title = str(context.get("title") or "").strip()
                department = str(context.get("department") or "").strip()
                if document_id and title and department:
                    current_sources.setdefault(
                        self._current_corpus_source_key(title, department), []
                    ).append(document_id)

            fixture_profile = _read_json(paths["fixtures"])
            profile_values = fixture_profile.get("fixtures")
            if not isinstance(profile_values, Mapping):
                raise EvidenceReviewValidationError("current_corpus_fixture_profile_invalid")

            existing_ids = {str(case.get("id") or "") for case in cases}
            imported: list[str] = []
            skipped: list[dict[str, str]] = []
            now = _utc_now()

            def source_template(*, fixture: str = "", category: str = "") -> dict[str, Any] | None:
                """按 Fixture 名或标准类目从旧用例中找到首个匹配模板。

                Args:
                    fixture: Fixture 名称，非空时匹配该 Fixture 的首个旧用例。
                    category: 标准类目名，fixture 为空时匹配 standard 用例的该类目。
                """
                for item in source_cases:
                    if fixture and str(item.get("required_fixture") or "standard") == fixture:
                        return item
                    if (
                        category
                        and str(item.get("required_fixture") or "standard") == "standard"
                        and str(item.get("category") or "") == category
                    ):
                        return item
                return None

            def rebound_fixture_template(
                fixture: str,
            ) -> tuple[dict[str, Any] | None, list[str], str]:
                """仅当每个源都能唯一映射时才选定旧 Fixture 模板。

                已复核的当前语料候选只证明该 Fixture 允许编写，并不代表该
                部门中任意一份文档都能支撑一个无关的旧问题；因此这里从
                工作区使用的同一份已审批元数据快照推导最终 Fixture 绑定，
                拒绝歧义或缺失的映射。

                Args:
                    fixture: Fixture 名称。
                """

                fixture_templates = [
                    item
                    for item in source_cases
                    if str(item.get("required_fixture") or "standard") == fixture
                ]
                if fixture == "prompt_injection_safety_fixture":
                    # 优先选择无源文档的直接攻击模板：当前语料无需包含恶意文档
                    # 即可验证输入安全边界，也避免把“假设可疑上下文”的普通问题
                    # 误报为一次成功的安全拒绝。
                    source_free = next(
                        (
                            item
                            for item in fixture_templates
                            if not item.get("expected_source_document_ids")
                        ),
                        None,
                    )
                    return (
                        source_free,
                        [],
                        "" if source_free is not None else "representative_template_missing",
                    )
                if fixture in {
                    "all_retrieval_branches_unavailable",
                    "frozen_corpus_absence_check",
                }:
                    template = fixture_templates[0] if fixture_templates else None
                    return (
                        template,
                        [],
                        "" if template is not None else "representative_template_missing",
                    )

                saw_ambiguous = False
                for template in fixture_templates:
                    legacy_ids = [
                        str(value).strip()
                        for value in template.get("expected_source_document_ids", [])
                        if isinstance(value, str) and value.strip()
                    ]
                    if not legacy_ids or any(
                        value in invalid_legacy_sources or value not in legacy_sources
                        for value in legacy_ids
                    ):
                        continue
                    rebound_ids: list[str] = []
                    for legacy_id in legacy_ids:
                        title, department = legacy_sources[legacy_id]
                        matches = current_sources.get(
                            self._current_corpus_source_key(title, department), []
                        )
                        if len(matches) != 1:
                            saw_ambiguous = saw_ambiguous or len(matches) > 1
                            rebound_ids = []
                            break
                        rebound_ids.append(matches[0])
                    if rebound_ids:
                        return template, rebound_ids, ""
                return (
                    None,
                    [],
                    "fixture_template_source_ambiguous"
                    if saw_ambiguous
                    else "fixture_template_source_unaligned",
                )

            def append_case(
                *,
                case_id: str,
                template: Mapping[str, Any],
                fixture: str,
                category: str,
                source_ids: list[str],
                department_id: str,
            ) -> None:
                """把旧模板重绑定后转换为新草稿用例并追加进当前列表。

                Args:
                    case_id: 新用例的确定性种子 ID。
                    template: 选中的旧数据包模板用例。
                    fixture: Fixture 名称，空字符串表示标准用例。
                    category: 新用例所属的评测类目。
                    source_ids: 已重绑定到当前语料的源文档 ID 列表。
                    department_id: 由源部门推导出的用例部门归属。
                """
                raw_case = {
                    "id": case_id,
                    "question": str(template.get("question") or "").strip(),
                    "category": category,
                    # 刻意受控的稀疏故障保留了稠密/图证据，但本质是降级响应
                    # 路径而非完整回答，因此期望状态归一为部分回答。
                    "expected_response_status": (
                        "partially_answered"
                        if fixture == "bm25_unavailable_dense_graph_available"
                        else str(template.get("expected_response_status") or "").strip()
                    ),
                    "expected_evidence_states": list(template.get("expected_evidence_states") or []),
                    "expected_reason_codes": list(template.get("expected_reason_codes") or []),
                    "expected_source_document_ids": source_ids,
                    # 旧版 Context/chunk 与引用标识无法证明与新语料对齐，
                    # 因此清空而不是按文件名映射或导入旧文档引用。
                    "expected_evidence_context_ids": [],
                    "expected_evidence_sections": [],
                    "expected_citation_context_ids": [],
                    "expected_missing_information_fields": list(template.get("expected_missing_information_fields") or []),
                    "expected_branch_availability": dict(template.get("expected_branch_availability") or {}),
                    "source_benchmark_ids": [],
                    "required_fixture": fixture or "standard",
                    "notes": "",
                }
                clean = self._validate_case(raw_case, contexts=contexts)
                clean["department_id"] = department_id
                self._validate_current_corpus_case(
                    workspace,
                    paths,
                    clean,
                    contexts=contexts,
                    fixture_profile=fixture_profile,
                )
                clean.update(
                    {
                        "review_status": "draft",
                        "case_revision": 1,
                        "last_editor_id": actor_id,
                        "last_edited_at": now,
                        "submitted_at": None,
                        "reviewer_id": None,
                        "reviewer_department_id": None,
                        "reviewed_at": None,
                        "review_reason": "",
                        "rejection_reason": "",
                    }
                )
                cases.append(clean)
                imported.append(case_id)

            for fixture in sorted(_REQUIRED_FIXTURES):
                seed_id = self._current_corpus_template_id(fixture=fixture)
                if seed_id in existing_ids or any(
                    str(case.get("required_fixture") or "") == fixture for case in cases
                ):
                    skipped.append({"sample": fixture, "reason_code": "representative_sample_exists"})
                    continue
                binding = profile_values.get(fixture)
                if not isinstance(binding, Mapping):
                    skipped.append({"sample": fixture, "reason_code": "fixture_binding_missing"})
                    continue
                template, source_ids, resolution_reason = rebound_fixture_template(fixture)
                if template is None:
                    skipped.append({"sample": fixture, "reason_code": resolution_reason})
                    continue
                if any(value not in source_departments for value in source_ids):
                    skipped.append({"sample": fixture, "reason_code": "fixture_binding_unaligned"})
                    continue
                # 源 ID 只能经由上面的唯一元数据重绑定选出。先持久化这份已审阅
                # 快照绑定再创建用例，使常规的 Fixture/源一致性约束对后续编辑
                # 与冻结依然生效。
                previous_source_ids: list[str] | None = None
                previous_source_count: int | None = None
                if isinstance(binding, dict):
                    previous_source_ids = list(binding.get("source_document_ids", []))
                    previous_source_count = int(binding.get("source_document_count", 0))
                    binding["source_document_ids"] = list(source_ids)
                    binding["source_document_count"] = len(source_ids)
                department_id = self._current_corpus_template_department(source_ids, source_departments)
                try:
                    append_case(
                        case_id=seed_id,
                        template=template,
                        fixture=fixture,
                        category=self._current_corpus_fixture_category(fixture),
                        source_ids=source_ids,
                        department_id=department_id,
                    )
                except EvidenceReviewValidationError as error:
                    # 校验失败时回滚 Fixture 绑定，保持档案与用例列表一致
                    if isinstance(binding, dict):
                        binding["source_document_ids"] = previous_source_ids or []
                        binding["source_document_count"] = previous_source_count or 0
                    skipped.append({"sample": fixture, "reason_code": str(error).partition(":")[0] or "representative_template_invalid"})

            for category in _CURRENT_CORPUS_TEMPLATE_STANDARD_CATEGORIES:
                seed_id = self._current_corpus_template_id(category=category)
                if seed_id in existing_ids:
                    skipped.append({"sample": category, "reason_code": "representative_sample_exists"})
                    continue
                templates = [
                    item
                    for item in source_cases
                    if str(item.get("required_fixture") or "standard") == "standard"
                    and str(item.get("category") or "") == category
                ]
                if not templates:
                    skipped.append({"sample": category, "reason_code": "representative_template_missing"})
                    continue

                imported_standard = False
                last_reason = "template_source_unaligned"
                for template in templates:
                    legacy_ids = [
                        str(value).strip()
                        for value in template.get("expected_source_document_ids", [])
                        if isinstance(value, str) and value.strip()
                    ]
                    if not legacy_ids or any(
                        value in invalid_legacy_sources or value not in legacy_sources
                        for value in legacy_ids
                    ):
                        last_reason = "template_source_unaligned"
                        continue
                    rebound_ids: list[str] = []
                    for legacy_id in legacy_ids:
                        matches = current_sources.get(
                            self._current_corpus_source_key(*legacy_sources[legacy_id]),
                            [],
                        )
                        if len(matches) != 1:
                            last_reason = (
                                "template_source_ambiguous"
                                if len(matches) > 1
                                else "template_source_unaligned"
                            )
                            rebound_ids = []
                            break
                        rebound_ids.append(matches[0])
                    if not rebound_ids:
                        continue
                    departments = {source_departments[value] for value in rebound_ids}
                    if len(departments) != 1:
                        last_reason = "template_department_mismatch"
                        continue
                    try:
                        append_case(
                            case_id=seed_id,
                            template=template,
                            fixture="",
                            category=category,
                            source_ids=rebound_ids,
                            department_id=next(iter(departments)),
                        )
                    except EvidenceReviewValidationError as error:
                        last_reason = str(error).partition(":")[0] or "representative_template_invalid"
                        continue
                    imported_standard = True
                    break
                if not imported_standard:
                    skipped.append({"sample": category, "reason_code": last_reason})

            if imported:
                revision = self._bump(workspace)
                _write_json(paths["fixtures"], fixture_profile)
                self._persist(paths, workspace, cases)
                events = _read_jsonl(paths["events"])
                events.append(
                    {
                        "event": "current_corpus_representative_template_import",
                        "dataset_id": dataset_id,
                        "actor_id": actor_id,
                        "imported_count": len(imported),
                        "skipped_count": len(skipped),
                        "revision": revision,
                        "recorded_at": now,
                    }
                )
                _write_jsonl(paths["events"], events[-10_000:])
            else:
                revision = int(workspace["revision"])

            return {
                "dataset_id": dataset_id,
                "revision": revision,
                "imported_case_ids": imported[:_CURRENT_CORPUS_TEMPLATE_IMPORT_LIMIT],
                "skipped": skipped[:_CURRENT_CORPUS_TEMPLATE_IMPORT_LIMIT],
            }

