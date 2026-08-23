"""Scoped, merge-only backups for the self-hosted administrator's content."""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.documents import _ingest_source
from app.db.connection import get_db_pool
from app.security.auth import Principal, require_principal

router = APIRouter(prefix="/api/v1/backup", tags=["backup"], dependencies=[Depends(require_principal)])
principal_dependency = Depends(require_principal)
BACKUP_VERSION = 1


class BackupImportRequest(BaseModel):
    version: int = Field(ge=1, le=1)
    sources: list[dict[str, Any]] = Field(default_factory=list, max_length=500)


@router.get("/export")
async def export_backup(principal: Principal = principal_dependency):
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT name, group_name, source_type, file_name, file_ext, original_text,
                      chunk_size, chunk_overlap, metadata
                 FROM document_sources WHERE owner_id = $1::uuid ORDER BY created_at ASC""",
            principal.id,
        )
    # Secrets, session cookies, API tokens, database config, and audit records
    # deliberately never leave this export.
    return {"version": BACKUP_VERSION, "sources": [dict(row) for row in rows]}


@router.post("/import")
async def import_backup(payload: BackupImportRequest, principal: Principal = principal_dependency):
    if payload.version != BACKUP_VERSION:
        raise HTTPException(status_code=400, detail="unsupported_backup_version")
    imported: list[str] = []
    skipped: list[str] = []
    pool = await get_db_pool()
    async with pool.acquire() as conn, conn.transaction():
        for source in payload.sources:
            allowed = {"name", "group_name", "source_type", "file_name", "file_ext", "original_text", "chunk_size", "chunk_overlap", "metadata"}
            if not isinstance(source, dict) or set(source) - allowed:
                raise HTTPException(status_code=400, detail="invalid_backup_source")
            content = str(source.get("original_text") or "").strip()
            name = str(source.get("name") or "").strip()
            group_name = str(source.get("group_name") or "").strip()
            if not content or not name or not group_name:
                raise HTTPException(status_code=400, detail="invalid_backup_source")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM document_sources WHERE owner_id = $1::uuid AND metadata->>'content_hash' = $2)",
                principal.id, digest,
            )
            if exists:
                skipped.append(name)
                continue
            result = await _ingest_source(
                name=name,
                group_name=group_name,
                source_type=str(source.get("source_type") or "import"),
                content=content,
                metadata=dict(source.get("metadata") or {}),
                file_name=source.get("file_name"),
                file_ext=source.get("file_ext"),
                chunk_size=int(source.get("chunk_size") or 400),
                chunk_overlap=int(source.get("chunk_overlap") or 80),
                owner_id=principal.id,
                content_hash=digest,
                connection=conn,
            )
            imported.append(result["source_id"])
    return {"version": BACKUP_VERSION, "imported_source_ids": imported, "skipped": skipped}
