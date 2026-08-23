from __future__ import annotations

import json
from datetime import timedelta

from app.core.time import utc_now_naive
from app.governance.harness import HarnessSupervisor


def test_harness_initializes_v2_state(tmp_path):
    HarnessSupervisor(str(tmp_path))

    data = json.loads((tmp_path / "harness-tasks.json").read_text(encoding="utf-8"))

    assert data["version"] == 2
    assert data["session_config"]["concurrency_mode"] == "exclusive"
    assert data["tasks"] == []
    assert data["session_count"] == 0


def test_harness_upsert_creates_marker_backup_lease_and_summary(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))

    supervisor.upsert_task(
        session_id="session-1",
        public_status="running",
        runtime_status="running",
        budget={"max_tool_calls": 5},
        current_batch=["s1"],
        checkpoint_seq=2,
        pending_approval_id=None,
        worker_id="worker-a",
        state_snapshot={"runtime_status": "running"},
    )

    task = supervisor.get_task("session-1")
    summary = supervisor.get_status_summary()

    assert (tmp_path / ".harness-active").exists()
    assert (tmp_path / "harness-tasks.json.bak").exists()
    assert task is not None
    assert task["status"] == "in_progress"
    assert task["lease"]["claimed_by"] == "worker-a"
    assert task["checkpoint"]["checkpoint_seq"] == 2
    assert summary["running"] == 1
    assert summary["tasks_total"] == 1
    assert "CHECKPOINT [session-1]" in (tmp_path / "harness-progress.txt").read_text(encoding="utf-8")


def test_harness_migrates_v1_state(tmp_path):
    (tmp_path / "harness-tasks.json").write_text(
        json.dumps({
            "version": 1,
            "created": "2026-01-01T00:00:00",
            "tasks": [{"id": "session-1", "runtime_status": "completed"}],
        }),
        encoding="utf-8",
    )

    supervisor = HarnessSupervisor(str(tmp_path))
    summary = supervisor.get_status_summary()
    data = json.loads((tmp_path / "harness-tasks.json").read_text(encoding="utf-8"))

    assert data["version"] == 2
    assert data["created"] == "2026-01-01T00:00:00"
    assert summary["completed"] == 1


def test_harness_recovers_from_backup_when_json_corrupt(tmp_path):
    valid_state = {
        "version": 2,
        "created": "2026-01-01T00:00:00",
        "session_config": {"concurrency_mode": "exclusive", "max_tasks_per_session": 20, "max_sessions": 50},
        "tasks": [{"id": "session-1", "runtime_status": "completed"}],
        "session_count": 1,
        "last_session": "2026-01-01T00:01:00",
    }
    (tmp_path / "harness-tasks.json").write_text("{bad json", encoding="utf-8")
    (tmp_path / "harness-tasks.json.bak").write_text(json.dumps(valid_state), encoding="utf-8")

    supervisor = HarnessSupervisor(str(tmp_path))
    task = supervisor.get_task("session-1")

    assert task is not None
    assert task["runtime_status"] == "completed"
    assert "RECOVERY [ENV_SETUP]" in (tmp_path / "harness-progress.txt").read_text(encoding="utf-8")


def test_harness_clears_active_marker_when_idle(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))
    supervisor.upsert_task(
        session_id="session-1",
        public_status="completed",
        runtime_status="completed",
        budget={},
    )

    supervisor.clear_active_if_idle()

    assert not (tmp_path / ".harness-active").exists()


def test_harness_detects_expired_lease(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))
    task = {
        "runtime_status": "running",
        "lease": {
            "claimed_by": "worker-a",
            "lease_expires_at": (utc_now_naive() - timedelta(seconds=1)).isoformat(),
        },
    }

    assert supervisor.is_lease_expired(task)


def test_harness_summary_counts_blocked_and_expired_leases(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))
    expired = (utc_now_naive() - timedelta(seconds=1)).isoformat()
    data = supervisor._load()
    data["tasks"] = [
        {"id": "failed", "runtime_status": "terminal_failed"},
        {"id": "blocked", "runtime_status": "pending", "depends_on": ["failed"]},
        {"id": "running", "runtime_status": "running", "lease": {"lease_expires_at": expired}},
    ]
    supervisor._save(data)

    summary = supervisor.get_status_summary()

    assert summary["terminal_failed"] == 1
    assert summary["pending"] == 1
    assert summary["blocked"] == 1
    assert summary["lease_expired"] == 1


def test_harness_state_lock_releases_after_state_transaction(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))

    supervisor.upsert_task(
        session_id="session-1",
        public_status="running",
        runtime_status="running",
        budget={},
    )

    assert not (tmp_path / ".harness-state.lock").exists()

    supervisor.renew_lease("session-1", worker_id="worker-b")
    task = supervisor.get_task("session-1")

    assert task["lease"]["claimed_by"] == "worker-b"
    assert not (tmp_path / ".harness-state.lock").exists()


def test_harness_state_lock_removes_stale_lock(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))
    lock_dir = tmp_path / ".harness-state.lock"
    lock_dir.mkdir()
    (lock_dir / "owner.json").write_text(
        json.dumps({
            "pid": 999999,
            "created_at": (utc_now_naive() - timedelta(minutes=10)).isoformat(),
        }),
        encoding="utf-8",
    )

    supervisor.upsert_task(
        session_id="session-1",
        public_status="running",
        runtime_status="running",
        budget={},
    )

    assert not lock_dir.exists()
    assert "removed stale harness state lock" in (tmp_path / "harness-progress.txt").read_text(encoding="utf-8")


def test_harness_validate_dependencies_marks_cycles_and_blocked_tasks(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))
    data = supervisor._load()
    data["tasks"] = [
        {"id": "a", "runtime_status": "pending", "depends_on": ["b"]},
        {"id": "b", "runtime_status": "pending", "depends_on": ["a"]},
        {"id": "failed", "runtime_status": "terminal_failed", "depends_on": []},
        {"id": "blocked", "runtime_status": "pending", "depends_on": ["failed"]},
    ]
    supervisor._save(data)

    result = supervisor.validate_dependencies()
    task_a = supervisor.get_task("a")
    task_b = supervisor.get_task("b")
    blocked = supervisor.get_task("blocked")
    progress = (tmp_path / "harness-progress.txt").read_text(encoding="utf-8")

    assert result["changed"] == 1
    assert result["cycles"] >= 1
    assert result["blocked"] >= 1
    assert task_a["runtime_status"] == "terminal_failed"
    assert task_b["runtime_status"] == "terminal_failed"
    assert blocked["runtime_status"] == "terminal_failed"
    assert "[DEPENDENCY] Circular dependency detected" in task_a["error_log"][0]
    assert "[DEPENDENCY] Blocked by failed" in blocked["error_log"][0]
    assert "ERROR [a] [DEPENDENCY]" in progress
    assert "ERROR [blocked] [DEPENDENCY]" in progress


def test_harness_validate_dependencies_is_idempotent(tmp_path):
    supervisor = HarnessSupervisor(str(tmp_path))
    data = supervisor._load()
    data["tasks"] = [
        {"id": "a", "runtime_status": "pending", "depends_on": ["a"]},
    ]
    supervisor._save(data)

    first = supervisor.validate_dependencies()
    second = supervisor.validate_dependencies()
    task = supervisor.get_task("a")

    assert first["changed"] == 1
    assert second["changed"] == 0
    assert len(task["error_log"]) == 1
