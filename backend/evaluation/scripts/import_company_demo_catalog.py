"""CLI for idempotently importing company-demo normalized files into the catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from evaluation.company_demo.corpus_import import import_company_demo_catalog
from infrastructure.documents.catalog import SQLiteDocumentCatalogRepository


def main() -> int:
    """Import the controlled corpus and print a path-free operator summary."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--upload-root", type=Path, required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--company", default="")
    args = parser.parse_args()
    report = import_company_demo_catalog(
        args.root,
        catalog=SQLiteDocumentCatalogRepository(args.catalog),
        upload_root=args.upload_root,
        tenant_id=args.tenant,
        company_id=args.company or None,
    )
    print(json.dumps({
        "document_count": report.document_count,
        "imported_count": report.imported_count,
        "existing_count": report.existing_count,
        "department_counts": report.department_counts,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
