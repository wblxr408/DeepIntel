from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.db.connection import get_db_pool

SECRET_KEYS = {"api_key", "token", "secret", "password", "authorization"}
DEFAULT_SUMMARY_CHARS = 500


@dataclass
class McpToolRequest:
    session_id: str
    decision_id: str | None
    tool_name: str
    requested_args: dict[str, Any]
    server_id: str
    approval_id: str | None = None


@dataclass
class McpToolResult:
    status: str
    result_summary: str | None = None
    result_hash: str | None = None
    server_fingerprint: str | None = None
    redaction_applied: bool = False
    error_category: str | None = None
    error_message: str | None = None
    approval_request: dict[str, Any] | None = None
    payload: Any | None = None


class McpPolicyProxy:
    """Governed MCP entrypoint: allowlist, trust, readonly-default, redaction."""

    def __init__(self, executor: Callable[[McpToolRequest], Awaitable[Any]] | None = None):
        self.executor = executor

    async def invoke(self, request: McpToolRequest) -> McpToolResult:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            registry = await conn.fetchrow(
                """
                SELECT server_id, trust_status, allowed_tools_json, read_only_default,
                       secret_policy_json, fingerprint
                FROM mcp_server_registry
                WHERE server_id = $1
                """,
                request.server_id,
            )
        if not registry:
            return McpToolResult(
                status="terminal_error",
                error_category="untrusted_mcp_server",
                error_message=f"unregistered server: {request.server_id}",
            )
        if registry["trust_status"] != "trusted":
            return McpToolResult(
                status="terminal_error",
                error_category="untrusted_mcp_server",
                error_message=f"untrusted server: {request.server_id}",
                server_fingerprint=registry["fingerprint"],
            )

        secret_policy = self._coerce_policy(registry["secret_policy_json"])
        allowed_tools = set(registry["allowed_tools_json"] or [])
        if request.tool_name not in allowed_tools:
            return McpToolResult(
                status="terminal_error",
                error_category="tool_not_allowed",
                error_message=f"tool not allowlisted: {request.tool_name}",
                server_fingerprint=registry["fingerprint"],
            )

        schema = self._tool_schema(secret_policy, request.tool_name)
        schema_error = self._validate_schema(
            request.requested_args,
            schema,
            label="argument",
            root_name="tool arguments",
        )
        if schema_error:
            return McpToolResult(
                status="terminal_error",
                error_category="schema_invalid",
                error_message=schema_error,
                server_fingerprint=registry["fingerprint"],
            )

        read_only_default = bool(registry["read_only_default"])
        write_tools = set(secret_policy.get("write_tools") or [])
        requires_approval = (not read_only_default) or request.tool_name in write_tools
        if requires_approval and not request.approval_id:
            redacted_args, _ = self._redact(request.requested_args, secret_policy=secret_policy)
            args_hash = self._stable_hash(request.requested_args)
            approval_request = {
                "approval_id": self._approval_id(request, args_hash),
                "node_id": None,
                "tool_name": request.tool_name,
                "risk_level": "high",
                "reason": "mcp_write_requires_approval",
                "request_payload": {
                    "server_id": request.server_id,
                    "tool_name": request.tool_name,
                    "args": redacted_args,
                    "args_hash": args_hash,
                },
                "status": "pending",
            }
            return McpToolResult(
                status="awaiting_approval",
                approval_request=approval_request,
                server_fingerprint=registry["fingerprint"],
            )

        if self.executor is None:
            return McpToolResult(
                status="terminal_error",
                error_category="unsupported_input",
                error_message="no MCP executor configured",
                server_fingerprint=registry["fingerprint"],
            )

        payload = await self.executor(request)
        output_schema = self._tool_output_schema(secret_policy, request.tool_name)
        output_schema_error = self._validate_schema(
            self._structured_output(payload),
            output_schema,
            label="output",
            root_name="tool output",
        )
        if output_schema_error:
            return McpToolResult(
                status="terminal_error",
                error_category="output_schema_invalid",
                error_message=output_schema_error,
                server_fingerprint=registry["fingerprint"],
            )

        redacted, redaction_applied = self._redact(payload, secret_policy=secret_policy)
        summary_chars = self._summary_chars(secret_policy)
        summary = str(redacted)[:summary_chars]
        return McpToolResult(
            status="success",
            result_summary=summary,
            result_hash=self._stable_hash(redacted),
            server_fingerprint=registry["fingerprint"],
            redaction_applied=redaction_applied,
            payload=redacted,
        )

    def _coerce_policy(self, raw_policy: Any) -> dict[str, Any]:
        if isinstance(raw_policy, dict):
            return raw_policy
        if isinstance(raw_policy, str):
            try:
                parsed = json.loads(raw_policy)
            except (json.JSONDecodeError, TypeError, ValueError):
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _summary_chars(self, secret_policy: dict[str, Any]) -> int:
        try:
            value = int(secret_policy.get("summary_chars", DEFAULT_SUMMARY_CHARS))
        except (TypeError, ValueError):
            return DEFAULT_SUMMARY_CHARS
        return max(80, min(value, 5000))

    def _tool_schema(self, secret_policy: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
        schemas = secret_policy.get("tool_schemas") or {}
        if not isinstance(schemas, dict):
            return None
        schema = schemas.get(tool_name)
        if not isinstance(schema, dict):
            return None
        input_schema = schema.get("inputSchema") or schema.get("input_schema")
        if isinstance(input_schema, dict):
            return input_schema
        return schema

    def _tool_output_schema(self, secret_policy: dict[str, Any], tool_name: str) -> dict[str, Any] | None:
        output_schemas = secret_policy.get("tool_output_schemas") or secret_policy.get("output_schemas") or {}
        if isinstance(output_schemas, dict):
            schema = output_schemas.get(tool_name)
            if isinstance(schema, dict):
                return schema
        schemas = secret_policy.get("tool_schemas") or {}
        if isinstance(schemas, dict):
            schema = schemas.get(tool_name)
            if isinstance(schema, dict):
                output_schema = schema.get("outputSchema") or schema.get("output_schema")
                if isinstance(output_schema, dict):
                    return output_schema
        return None

    def _structured_output(self, payload: Any) -> Any:
        if isinstance(payload, dict) and "structuredContent" in payload:
            return payload["structuredContent"]
        return payload

    def _validate_schema(
        self,
        value: Any,
        schema: dict[str, Any] | None,
        *,
        label: str,
        root_name: str,
    ) -> str | None:
        if not schema:
            return None
        if schema.get("type") != "object":
            return f"{root_name} schema must use type=object"
        return self._validate_object(label, value, schema, root=True, label=label, root_name=root_name)

    def _validate_object(
        self,
        path: str,
        value: Any,
        rule: dict[str, Any],
        *,
        root: bool = False,
        label: str,
        root_name: str,
    ) -> str | None:
        if not isinstance(value, dict):
            return f"{root_name} must be an object" if root else f"{path} must be object"
        properties = rule.get("properties") or {}
        if not isinstance(properties, dict):
            return f"{root_name} schema properties must be an object"
        required = rule.get("required") or []
        if not isinstance(required, list):
            return f"{root_name} schema required must be a list"
        for key in required:
            if key not in value:
                missing_path = key if root else f"{path}.{key}"
                return f"missing required {label}: {missing_path}"
        if rule.get("additionalProperties") is False:
            allowed = set(properties)
            extra = sorted(set(value) - allowed)
            if extra:
                unexpected_path = extra[0] if root else f"{path}.{extra[0]}"
                return f"unexpected {label}: {unexpected_path}"
        for key, child_value in value.items():
            child_rule = properties.get(key)
            if isinstance(child_rule, dict):
                child_path = key if root else f"{path}.{key}"
                type_error = self._validate_value(child_path, child_value, child_rule, label=label, root_name=root_name)
                if type_error:
                    return type_error
        return None

    def _validate_value(
        self,
        path: str,
        value: Any,
        rule: dict[str, Any],
        *,
        label: str,
        root_name: str,
    ) -> str | None:
        enum_values = rule.get("enum")
        if isinstance(enum_values, list) and value not in enum_values:
            return f"{label} {path} must be one of {enum_values}"

        expected = rule.get("type")
        if expected == "string" and not isinstance(value, str):
            return f"{label} {path} must be string"
        if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
            return f"{label} {path} must be integer"
        if expected == "number" and (not isinstance(value, (int, float)) or isinstance(value, bool)):
            return f"{label} {path} must be number"
        if expected == "boolean" and not isinstance(value, bool):
            return f"{label} {path} must be boolean"
        if expected == "array":
            if not isinstance(value, list):
                return f"{label} {path} must be array"
            items_rule = rule.get("items")
            if isinstance(items_rule, dict):
                for index, item in enumerate(value):
                    item_error = self._validate_value(
                        f"{path}[{index}]",
                        item,
                        items_rule,
                        label=label,
                        root_name=root_name,
                    )
                    if item_error:
                        return item_error
        if expected == "object":
            return self._validate_object(path, value, rule, label=label, root_name=root_name)
        return None

    def _redact(self, value: Any, *, secret_policy: dict[str, Any] | None = None) -> tuple[Any, bool]:
        secret_policy = secret_policy or {}
        configured_keys = {
            str(item).lower()
            for item in secret_policy.get("redact_keys", [])
            if str(item).strip()
        }
        secret_keys = SECRET_KEYS | configured_keys
        if isinstance(value, dict):
            changed = False
            output: dict[str, Any] = {}
            for key, item in value.items():
                if str(key).lower() in secret_keys:
                    output[key] = "[REDACTED]"
                    changed = True
                else:
                    redacted, child_changed = self._redact(item, secret_policy=secret_policy)
                    output[key] = redacted
                    changed = changed or child_changed
            return output, changed
        if isinstance(value, list):
            changed = False
            output = []
            for item in value:
                redacted, child_changed = self._redact(item, secret_policy=secret_policy)
                output.append(redacted)
                changed = changed or child_changed
            return output, changed
        return value, False

    def _stable_hash(self, value: Any) -> str:
        serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _approval_id(self, request: McpToolRequest, args_hash: str) -> str:
        payload = {
            "session_id": request.session_id,
            "decision_id": request.decision_id,
            "server_id": request.server_id,
            "tool_name": request.tool_name,
            "args_hash": args_hash,
        }
        return f"ap-{self._stable_hash(payload)[:12]}"
