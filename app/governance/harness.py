from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar

from app.config import get_settings
from app.core.time import utc_now_naive


class HarnessSupervisor:
    """File-backed supervisor state for long-running research tasks."""

    STATE_VERSION = 2
    ACTIVE_RUNTIME_STATUSES: ClassVar[set[str]] = {"running", "awaiting_approval", "retryable_failed"}
    TERMINAL_RUNTIME_STATUSES: ClassVar[set[str]] = {"completed", "terminal_failed", "failed"}

    def __init__(self, state_root: str | None = None):
        settings = get_settings()
        root = state_root or getattr(settings, "harness_state_root", None) or "./data/harness"
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.tasks_path = self.root / "harness-tasks.json"
        self.backup_path = self.root / "harness-tasks.json.bak"
        self.progress_path = self.root / "harness-progress.txt"
        self.active_marker = self.root / ".harness-active"
        self.lock_path = self.root / ".harness-state.lock"
        self._lock_depth = 0
        if not self.tasks_path.exists():
            self._save(self._default_state())

    def _now(self) -> str:
        return utc_now_naive().isoformat()

    def _default_state(self) -> dict[str, Any]:
        return {
            "version": self.STATE_VERSION,
            "created": self._now(),
            "session_config": {
                "concurrency_mode": "exclusive",
                "max_tasks_per_session": 20,
                "max_sessions": 50,
            },
            "tasks": [],
            "session_count": 0,
            "last_session": None,
        }

    def _migrate_state(self, data: dict[str, Any]) -> dict[str, Any]:
        if int(data.get("version", 1) or 1) >= self.STATE_VERSION:
            data.setdefault("session_config", self._default_state()["session_config"])
            data.setdefault("tasks", [])
            data.setdefault("session_count", 0)
            data.setdefault("last_session", None)
            return data

        migrated = self._default_state()
        migrated["created"] = data.get("created") or migrated["created"]
        migrated["tasks"] = data.get("tasks") or []
        migrated["last_session"] = data.get("last_session")
        return migrated

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.tasks_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = self._recover_from_backup()
        migrated = self._migrate_state(data)
        if migrated.get("version") != data.get("version"):
            self._save(migrated)
        return migrated

    def _recover_from_backup(self) -> dict[str, Any]:
        if self.backup_path.exists():
            try:
                recovered = json.loads(self.backup_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                recovered = None
            if isinstance(recovered, dict):
                self._append_progress(
                    "SESSION-0",
                    "RECOVERY",
                    category="ENV_SETUP",
                    message="restored harness-tasks.json from backup",
                )
                migrated = self._migrate_state(recovered)
                self._save(migrated)
                return migrated

        self._append_progress(
            "SESSION-0",
            "ERROR",
            category="ENV_SETUP",
            message="harness-tasks.json corrupted and unrecoverable",
        )
        raise RuntimeError("harness-tasks.json corrupted and unrecoverable")

    def _save(self, data: dict[str, Any]) -> None:
        if self.tasks_path.exists():
            backup_tmp = self.backup_path.with_suffix(".bak.tmp")
            backup_tmp.write_text(self.tasks_path.read_text(encoding="utf-8"), encoding="utf-8")
            os.replace(backup_tmp, self.backup_path)
        data["version"] = self.STATE_VERSION
        temp = self.root / "harness-tasks.json.tmp"
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, self.tasks_path)

    @contextmanager
    def _state_lock(self, *, timeout_seconds: float = 5.0, stale_seconds: float = 300.0) -> Iterator[None]:
        if self._lock_depth > 0:
            self._lock_depth += 1
            try:
                yield
            finally:
                self._lock_depth -= 1
            return

        deadline = time.monotonic() + timeout_seconds
        acquired = False
        while not acquired:
            try:
                os.mkdir(self.lock_path)
                owner = {
                    "pid": os.getpid(),
                    "created_at": self._now(),
                }
                (self.lock_path / "owner.json").write_text(json.dumps(owner), encoding="utf-8")
                acquired = True
            except FileExistsError:
                self._clear_stale_lock(stale_seconds=stale_seconds)
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"harness state lock contention: {self.lock_path}")
                time.sleep(0.05)

        self._lock_depth = 1
        try:
            yield
        finally:
            self._lock_depth = 0
            try:
                owner_path = self.lock_path / "owner.json"
                if owner_path.exists():
                    owner_path.unlink()
                self.lock_path.rmdir()
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def _clear_stale_lock(self, *, stale_seconds: float) -> None:
        owner_path = self.lock_path / "owner.json"
        try:
            raw_owner = owner_path.read_text(encoding="utf-8")
            owner = json.loads(raw_owner)
            created_at = datetime.fromisoformat(str(owner.get("created_at")))
        except (json.JSONDecodeError, TypeError, ValueError):
            created_at = utc_now_naive() - timedelta(seconds=stale_seconds + 1)
        if utc_now_naive() - created_at < timedelta(seconds=stale_seconds):
            return
        try:
            if owner_path.exists():
                owner_path.unlink()
            self.lock_path.rmdir()
            self._append_progress(
                "SESSION-0",
                "WARN",
                category="LOCK",
                message="removed stale harness state lock",
            )
        except OSError:
            return

    def _append_progress(
        self,
        session_label: str,
        event_type: str,
        *,
        task_id: str | None = None,
        category: str | None = None,
        message: str = "",
    ) -> None:
        timestamp = self._now()
        parts = [f"[{timestamp}]", f"[{session_label}]", event_type]
        if task_id:
            parts.append(f"[{task_id}]")
        if category:
            parts.append(f"[{category}]")
        if message:
            parts.append(message)
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(" ".join(parts) + "\n")

    def ensure_active(self) -> None:
        self.active_marker.touch(exist_ok=True)

    def clear_active_if_idle(self) -> None:
        with self._state_lock():
            data = self._load()
            if (
                not any(task.get("runtime_status") in self.ACTIVE_RUNTIME_STATUSES for task in data.get("tasks", []))
                and self.active_marker.exists()
            ):
                self.active_marker.unlink()

    def is_lease_expired(self, task: dict[str, Any], *, now: datetime | None = None) -> bool:
        lease = task.get("lease") or {}
        raw_expires_at = lease.get("lease_expires_at")
        if not raw_expires_at:
            return True
        try:
            expires_at = datetime.fromisoformat(str(raw_expires_at))
        except ValueError:
            return True
        return (now or utc_now_naive()) >= expires_at

    def renew_lease(self, session_id: str, *, worker_id: str = "worker-1", ttl_minutes: int = 10) -> dict[str, Any] | None:
        with self._state_lock():
            data = self._load()
            task = next((item for item in data.get("tasks", []) if item.get("id") == session_id), None)
            if task is None:
                return None
            task["lease"] = self._build_lease(worker_id=worker_id, ttl_minutes=ttl_minutes)
            data["last_session"] = self._now()
            self._save(data)
            self._append_progress(
                "SESSION-0",
                "CHECKPOINT",
                task_id=session_id,
                message=f"lease renewed by {worker_id}",
            )
            return task

    def _build_lease(self, *, worker_id: str, ttl_minutes: int = 10) -> dict[str, Any]:
        return {
            "claimed_by": worker_id,
            "lease_expires_at": (utc_now_naive() + timedelta(minutes=ttl_minutes)).isoformat(),
            "last_heartbeat_at": self._now(),
        }

    def upsert_task(
        self,
        *,
        session_id: str,
        public_status: str,
        runtime_status: str,
        budget: dict[str, Any],
        current_batch: list[str] | None = None,
        checkpoint_seq: int = 0,
        pending_approval_id: str | None = None,
        used_total_tokens: int = 0,
        used_cost_usd: float = 0.0,
        last_error: dict[str, Any] | None = None,
        worker_id: str = "worker-1",
        state_snapshot: dict[str, Any] | None = None,
    ) -> None:
        self.ensure_active()
        with self._state_lock():
            data = self._load()
            if not data.get("last_session"):
                data["session_count"] = int(data.get("session_count", 0) or 0) + 1
            tasks = data.setdefault("tasks", [])
            task = next((item for item in tasks if item.get("id") == session_id), None)
            if task is None:
                task = {
                    "id": session_id,
                    "title": f"Research session {session_id}",
                    "status": "pending",
                    "priority": "P1",
                    "depends_on": [],
                    "attempts": 0,
                    "max_attempts": 3,
                    "created_at": self._now(),
                }
                tasks.append(task)
            task["public_status"] = public_status
            task["runtime_status"] = runtime_status
            task["status"] = self._task_status_from_runtime(runtime_status)
            task["budget"] = budget
            task["checkpoint"] = {
                "checkpoint_seq": checkpoint_seq,
                "thread_id": session_id,
                "current_batch": current_batch or [],
                "pending_approval_id": pending_approval_id,
                "used_total_tokens": used_total_tokens,
                "used_cost_usd": used_cost_usd,
                "state_snapshot": state_snapshot,
            }
            task["last_error"] = last_error
            task["lease"] = self._build_lease(worker_id=worker_id)
            task["updated_at"] = self._now()
            if runtime_status in self.TERMINAL_RUNTIME_STATUSES:
                task["completed_at"] = task.get("completed_at") or self._now()
            data["last_session"] = self._now()
            self._save(data)
            self._append_progress(
                self._session_label(data),
                "CHECKPOINT",
                task_id=session_id,
                message=f"runtime={runtime_status} public={public_status} batch={current_batch or []}",
            )

    def get_task(self, session_id: str) -> dict[str, Any] | None:
        data = self._load()
        return next((item for item in data.get("tasks", []) if item.get("id") == session_id), None)

    def validate_dependencies(self) -> dict[str, int]:
        with self._state_lock():
            data = self._load()
            tasks = data.get("tasks", [])
            tasks_by_id = {task.get("id"): task for task in tasks if task.get("id")}
            changed = False
            cycle_count = 0
            blocked_count = 0

            for task in tasks:
                task_id = task.get("id")
                if not task_id or task.get("runtime_status") == "completed":
                    continue
                cycle_path = self._dependency_cycle_path(task_id, tasks_by_id)
                if cycle_path:
                    cycle_count += 1
                    changed = self._mark_dependency_failed(
                        task,
                        f"[DEPENDENCY] Circular dependency detected: {' -> '.join(cycle_path)}",
                    ) or changed

            terminal_failed_ids = self._terminal_failed_ids(tasks)
            while True:
                round_changed = False
                for task in tasks:
                    task_id = task.get("id")
                    if not task_id or task.get("runtime_status") in {"completed", "terminal_failed", "failed"}:
                        continue
                    blocking_deps = sorted(dep for dep in task.get("depends_on", []) if dep in terminal_failed_ids)
                    if not blocking_deps:
                        continue
                    blocked_count += 1
                    round_changed = self._mark_dependency_failed(
                        task,
                        f"[DEPENDENCY] Blocked by failed {blocking_deps[0]}",
                    ) or round_changed
                if not round_changed:
                    break
                changed = True
                terminal_failed_ids = self._terminal_failed_ids(tasks)

            if changed:
                data["last_session"] = self._now()
                self._save(data)
            return {"cycles": cycle_count, "blocked": blocked_count, "changed": int(changed)}

    def _dependency_cycle_path(self, task_id: str, tasks_by_id: dict[str, dict[str, Any]]) -> list[str]:
        stack: list[str] = []
        visited: set[str] = set()

        def visit(current_id: str) -> list[str]:
            if current_id in stack:
                return stack[stack.index(current_id):] + [current_id]
            if current_id in visited:
                return []
            visited.add(current_id)
            stack.append(current_id)
            current = tasks_by_id.get(current_id) or {}
            for dep in current.get("depends_on", []):
                if dep in tasks_by_id:
                    cycle = visit(dep)
                    if cycle:
                        return cycle
                elif dep == task_id:
                    return [task_id, dep]
            stack.pop()
            return []

        return visit(task_id)

    def _mark_dependency_failed(self, task: dict[str, Any], message: str) -> bool:
        error_log = task.setdefault("error_log", [])
        already_marked = message in error_log and task.get("runtime_status") == "terminal_failed"
        if already_marked:
            return False
        if message not in error_log:
            error_log.append(message)
        task["runtime_status"] = "terminal_failed"
        task["public_status"] = "failed"
        task["status"] = "failed"
        task["last_error"] = {"category": "dependency", "message": message}
        task["completed_at"] = task.get("completed_at") or self._now()
        self._append_progress(
            "SESSION-0",
            "ERROR",
            task_id=str(task.get("id")),
            category="DEPENDENCY",
            message=message,
        )
        return True

    def _terminal_failed_ids(self, tasks: list[dict[str, Any]]) -> set[str]:
        return {
            str(task.get("id"))
            for task in tasks
            if task.get("id")
            and (
                task.get("runtime_status") in {"terminal_failed", "failed"}
                or (task.get("attempts", 0) >= task.get("max_attempts", 3) and task.get("runtime_status") == "retryable_failed")
            )
        }

    def get_status_summary(self) -> dict[str, Any]:
        data = self._load()
        tasks = data.get("tasks", [])
        counts = {
            "tasks_total": len(tasks),
            "completed": 0,
            "running": 0,
            "awaiting_approval": 0,
            "retryable_failed": 0,
            "terminal_failed": 0,
            "pending": 0,
            "blocked": 0,
            "lease_expired": 0,
        }
        terminal_failed_ids = {
            task.get("id")
            for task in tasks
            if task.get("runtime_status") in {"terminal_failed", "failed"}
            or (task.get("attempts", 0) >= task.get("max_attempts", 3) and task.get("runtime_status") == "retryable_failed")
        }
        for task in tasks:
            runtime_status = task.get("runtime_status") or task.get("status") or "pending"
            if runtime_status == "completed":
                counts["completed"] += 1
            elif runtime_status == "running":
                counts["running"] += 1
            elif runtime_status == "awaiting_approval":
                counts["awaiting_approval"] += 1
            elif runtime_status == "retryable_failed":
                counts["retryable_failed"] += 1
            elif runtime_status in {"terminal_failed", "failed"}:
                counts["terminal_failed"] += 1
            else:
                counts["pending"] += 1
            if any(dep in terminal_failed_ids for dep in task.get("depends_on", [])):
                counts["blocked"] += 1
            if task.get("runtime_status") in self.ACTIVE_RUNTIME_STATUSES and self.is_lease_expired(task):
                counts["lease_expired"] += 1
        counts["session_count"] = data.get("session_count", 0)
        counts["last_session"] = data.get("last_session")
        return counts

    def log_stats(self) -> None:
        summary = self.get_status_summary()
        message = " ".join(f"{key}={value}" for key, value in summary.items())
        self._append_progress("SESSION-0", "STATS", message=message)

    def _task_status_from_runtime(self, runtime_status: str) -> str:
        if runtime_status == "completed":
            return "completed"
        if runtime_status in {"terminal_failed", "failed"}:
            return "failed"
        if runtime_status in self.ACTIVE_RUNTIME_STATUSES:
            return "in_progress"
        return "pending"

    def _session_label(self, data: dict[str, Any]) -> str:
        session_count = int(data.get("session_count", 0) or 0)
        return f"SESSION-{max(session_count, 1)}"
