from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.security.auth import Principal, require_principal
from app.skills import SkillCreateRequest, SkillUpdateRequest, get_skill_registry
from app.skills.service import create_skill, delete_skill, update_skill

router = APIRouter(prefix="/api/v1/skills", tags=["skills"], dependencies=[Depends(require_principal)])
PrincipalDep = Annotated[Principal, Depends(require_principal)]
SkillUpload = Annotated[UploadFile, File(...)]


def _runtime_payload(skill: Any) -> dict[str, Any]:
    return skill.to_runtime_view().model_dump()


@router.get("")
async def get_skills(principal: PrincipalDep):
    registry = get_skill_registry(principal.id)
    await registry.ensure_loaded()
    skills = registry.list_runtime_skills()
    return {"items": [skill.to_runtime_view().model_dump() for skill in skills]}


@router.get("/{skill_id}")
async def get_skill(skill_id: str, principal: PrincipalDep):
    skill = await get_skill_registry(principal.id).get_skill_record(skill_id)
    if skill.is_deleted:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _runtime_payload(skill)


@router.post("")
async def create_skill_endpoint(request: SkillCreateRequest, principal: PrincipalDep):
    skill = await create_skill(request.markdown_content, principal.id)
    await get_skill_registry(principal.id).upsert(skill)
    return _runtime_payload(skill)


@router.post("/upload")
async def upload_skill(file: SkillUpload, principal: PrincipalDep):
    if not file.filename or not file.filename.lower().endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md skill files are supported")
    content = await file.read()
    markdown = content.decode("utf-8")
    skill = await create_skill(markdown, principal.id)
    await get_skill_registry(principal.id).upsert(skill)
    return _runtime_payload(skill)


@router.put("/{skill_id}")
async def update_skill_endpoint(skill_id: str, request: SkillUpdateRequest, principal: PrincipalDep):
    skill = await update_skill(skill_id, request.markdown_content, principal.id)
    await get_skill_registry(principal.id).upsert(skill)
    return _runtime_payload(skill)


@router.delete("/{skill_id}")
async def delete_skill_endpoint(skill_id: str, principal: PrincipalDep):
    await delete_skill(skill_id, principal.id)
    await get_skill_registry(principal.id).remove(skill_id)
    return {"message": "Skill deleted"}
