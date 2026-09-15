"""Department metadata consistency across catalog and graph boundaries."""

from __future__ import annotations

import pytest
from domain.departments import department_display_name
from domain.documents import DepartmentRecord
from infrastructure.graph.department_metadata_sync import sync_known_department_metadata


class _Catalog:
    def __init__(self, records: list[DepartmentRecord]) -> None:
        self.records = records
        self.updates: list[DepartmentRecord] = []

    def list_departments(self, *, tenant_id: str, company_id: str | None = None):
        return [
            record
            for record in self.records
            if record.tenant_id == tenant_id
            and (company_id is None or record.company_id == company_id)
        ]

    def upsert_department(self, record: DepartmentRecord) -> DepartmentRecord:
        self.updates.append(record)
        self.records = [
            record if item.department_id == record.department_id else item
            for item in self.records
        ]
        return record


class _Graph:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def sync_department_names(
        self,
        tenant_id: str,
        department_names: dict[str, str],
    ) -> int:
        self.calls.append((tenant_id, department_names))
        return len(department_names)


def _record(department_id: str, name: str) -> DepartmentRecord:
    return DepartmentRecord(
        department_id=department_id,
        tenant_id="tenant-a",
        company_id="tenant-a",
        name=name,
        normalized_key=department_id,
    )


def test_department_display_name_canonicalizes_known_ids_and_preserves_custom_names() -> None:
    assert department_display_name("administration", "administration") == "行政管理部"
    assert department_display_name("finance") == "财务部"
    assert department_display_name("procurement_warehouse") == "采购/仓储部"
    assert department_display_name("research", "研究院") == "研究院"
    assert department_display_name("research") == "research"


@pytest.mark.asyncio
async def test_sync_updates_existing_known_departments_without_creating_custom_records() -> None:
    catalog = _Catalog(
        [
            _record("human_resources", "human_resources"),
            _record("finance", "finance"),
            _record("procurement_warehouse", "procurement_warehouse"),
            _record("administration", "administration"),
            _record("research", "研究院"),
        ]
    )
    graph = _Graph()

    report = await sync_known_department_metadata(
        catalog=catalog,
        graph=graph,
        tenant_id="tenant-a",
    )

    assert report.discovered_departments == 5
    assert report.known_departments == 4
    assert report.catalog_updates == 4
    assert report.graph_updates == 4
    assert {record.department_id for record in catalog.updates} == {
        "administration",
        "finance",
        "human_resources",
        "procurement_warehouse",
    }
    assert next(record for record in catalog.records if record.department_id == "research") == _record(
        "research", "研究院"
    )
    assert graph.calls == [("tenant-a", report.department_names)]

    second = await sync_known_department_metadata(
        catalog=catalog,
        graph=graph,
        tenant_id="tenant-a",
    )
    assert second.catalog_updates == 0
