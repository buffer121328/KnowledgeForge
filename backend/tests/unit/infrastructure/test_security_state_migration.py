from __future__ import annotations

import json
from pathlib import Path

from infrastructure.security.security_state_migration import inventory_legacy_state


def test_legacy_inventory_is_dry_run_and_secret_free(tmp_path: Path) -> None:
    webhook_file = tmp_path / "webhooks.json"
    webhook_file.write_text(
        json.dumps(
            [
                {
                    "id": "legacy-1",
                    "org_id": "",
                    "url": "https://legacy.example.test/hook",
                    "events": ["qa.completed"],
                    "secret": "do-not-print",
                    "is_active": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    inventory = inventory_legacy_state(webhook_file)

    assert inventory["mode"] == "dry-run"
    assert inventory["legacy_webhooks"] == [{"webhook_id": "legacy-1"}]
    assert inventory["owner_assignment_required"] is True
    rendered = repr(inventory)
    assert "legacy.example.test" not in rendered
    assert "do-not-print" not in rendered
    assert str(webhook_file) not in rendered
