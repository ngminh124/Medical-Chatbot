"""Admin router for monitoring, analytics, users and conversations."""

from datetime import datetime, timedelta, timezone
import math
from typing import Any, Optional
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import desc, func, or_
from sqlalchemy.orm import Session

from backend.models import Message, Thread, User

from ..core.runtime_settings import get_runtime_settings, update_runtime_settings
from ..core.security import get_current_admin
from ..database import get_db_session

router = APIRouter(prefix="/v1/admin", tags=["admin"])

PROMETHEUS_URL = "http://localhost:9090"
LOKI_URL = "http://localhost:3100"
TEMPO_URL = "http://localhost:3200"


async def _prom_query(query: str):
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query",
            params={"query": query},
        )
        resp.raise_for_status()
        return resp.json()


def _extract_scalar(prom_resp: dict) -> float:
    try:
        result = prom_resp.get("data", {}).get("result", [])
        if not result:
            return 0.0
        value = float(result[0].get("value", [0, "0"])[1])
        return value if math.isfinite(value) else 0.0
    except Exception:
        return 0.0


@router.get("/overview")
async def overview(
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db_session),
):
    total_users = db.query(func.count(User.id)).scalar() or 0
    total_conversations = db.query(func.count(Thread.id)).scalar() or 0

    since = datetime.now(timezone.utc) - timedelta(days=1)
    active_users_today = (
        db.query(func.count(func.distinct(Thread.user_id)))
        .filter(Thread.updated_at >= since)
        .scalar()
        or 0
    )

    requests_per_minute = 0.0
    avg_response_time = 0.0
    cache_hit_rate = 0.0
    tokens_per_second = 0.0
    active_requests = 0.0

    try:
        rpm = await _prom_query("sum(increase(rag_requests_total[5m])) / 5")
        requests_per_minute = _extract_scalar(rpm)

        avg = await _prom_query(
            "sum(rate(rag_request_duration_seconds_sum[5m])) / clamp_min(sum(rate(rag_request_duration_seconds_count[5m])), 1)"
        )
        avg_response_time = _extract_scalar(avg)

        cache = await _prom_query(
            "sum(rate(rag_cache_hits_total[5m])) / clamp_min(sum(rate(rag_cache_requests_total[5m])), 1)"
        )
        cache_hit_rate = _extract_scalar(cache)

        tps = await _prom_query("sum(increase(rag_tokens_generated_total[5m])) / 300")
        tokens_per_second = _extract_scalar(tps)

        active = await _prom_query("sum(rag_active_requests)")
        active_requests = _extract_scalar(active)

        fastapi_up = await _prom_query('sum(up{job="fastapi"})')
        prometheus_up = await _prom_query('sum(up{job="prometheus"})')
        monitoring_up = {
            "fastapi": _extract_scalar(fastapi_up) > 0,
            "prometheus": _extract_scalar(prometheus_up) > 0,
        }
    except Exception:
        monitoring_up = {
            "fastapi": False,
            "prometheus": False,
        }

    return {
        "total_users": total_users,
        "active_users_today": active_users_today,
        "total_conversations": total_conversations,
        "requests_per_minute": requests_per_minute,
        "tokens_per_second": tokens_per_second,
        "active_requests": active_requests,
        "cache_hit_rate": cache_hit_rate,
        "average_response_time": avg_response_time,
        "monitoring": monitoring_up,
    }


@router.get("/users")
def list_users(
    search: str = "",
    role: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db_session),
):
    query = db.query(User)

    if search:
        pattern = f"%{search}%"
        query = query.filter(or_(User.email.ilike(pattern), User.name.ilike(pattern)))
    if role:
        query = query.filter(User.type == role)

    total = query.count()
    items = (
        query.order_by(desc(User.created_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    user_ids = [u.id for u in items]
    last_active_map: dict[str, Optional[str]] = {}
    if user_ids:
        rows = (
            db.query(Thread.user_id, func.max(Thread.updated_at))
            .filter(Thread.user_id.in_(user_ids))
            .group_by(Thread.user_id)
            .all()
        )
        last_active_map = {str(uid): ts.isoformat() if ts else None for uid, ts in rows}

    return {
        "items": [
            {
                "id": str(u.id),
                "email": u.email,
                "name": u.name,
                "role": u.type or "user",
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "last_active": last_active_map.get(str(u.id)),
            }
            for u in items
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@router.get("/conversations")
def list_conversations(
    search: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db_session),
):
    query = db.query(Thread, User).join(User, Thread.user_id == User.id)

    if search:
        pattern = f"%{search}%"
        query = query.filter(
            or_(Thread.title.ilike(pattern), User.email.ilike(pattern), User.name.ilike(pattern))
        )

    total = query.count()
    rows = (
        query.order_by(desc(Thread.updated_at))
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    items = []
    for thread, user in rows:
        msg_count = db.query(func.count(Message.id)).filter(Message.thread_id == thread.id).scalar() or 0
        items.append(
            {
                "id": str(thread.id),
                "title": thread.title,
                "user_email": user.email,
                "updated_at": thread.updated_at.isoformat() if thread.updated_at else None,
                "created_at": thread.created_at.isoformat() if thread.created_at else None,
                "message_count": msg_count,
            }
        )

    return {"items": items, "total": total, "page": page, "page_size": page_size}


@router.get("/conversations/{thread_id}/messages")
def get_conversation_messages(
    thread_id: UUID,
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db_session),
):
    messages = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .all()
    )

    rows = []
    pending_user: Optional[Message] = None
    for m in messages:
        if m.role == "user":
            pending_user = m
            continue
        if m.role == "assistant" and pending_user:
            latency = None
            if pending_user.created_at and m.created_at:
                latency = max((m.created_at - pending_user.created_at).total_seconds(), 0.0)
            rows.append(
                {
                    "question": pending_user.content,
                    "answer": m.content,
                    "latency": latency,
                    "timestamp": m.created_at.isoformat() if m.created_at else None,
                }
            )
            pending_user = None

    return {"items": rows}


@router.get("/metrics/range")
async def metrics_range(
    query: str,
    start: int,
    end: int,
    response: Response,
    step: str = "5s",
    _: User = Depends(get_current_admin),
):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": step,
            },
        )
        resp.raise_for_status()
        payload = resp.json() or {}

    result = payload.get("data", {}).get("result", [])
    values = result[0].get("values", []) if result else []

    points: list[dict[str, float | int]] = []
    for ts, val in values:
        try:
            ts_value = int(float(ts))
            value = float(val)
            if not math.isfinite(value):
                continue

            points.append({"timestamp": ts_value, "value": value})
        except Exception:
            continue

    return {
        "points": points,
        "step": step,
    }


@router.get("/logs")
async def logs(
    level: str = "",
    keyword: str = "",
    limit: int = Query(200, ge=10, le=1000),
    _: User = Depends(get_current_admin),
):
    filters = ["{job=\"fastapi\",service=\"rag\"}"]
    if level:
        filters.append(f" |= \"{level.upper()}\"")
    if keyword:
        filters.append(f" |= \"{keyword}\"")

    q = "".join(filters)

    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            f"{LOKI_URL}/loki/api/v1/query_range",
            params={
                "query": q,
                "limit": limit,
                "direction": "backward",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    lines = []
    for stream in data.get("data", {}).get("result", []):
        labels = stream.get("stream", {})
        for ts, line in stream.get("values", []):
            lines.append({"ts": ts, "line": line, "labels": labels})

    lines.sort(key=lambda item: item["ts"], reverse=True)
    return {"items": lines[:limit]}


@router.get("/traces")
async def traces(
    limit: int = Query(20, ge=1, le=100),
    service: str = "rag_pipeline",
    _: User = Depends(get_current_admin),
):
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            resp = await client.get(
                f"{TEMPO_URL}/api/search",
                params={"limit": limit, "tags": f"service.name={service}"},
            )
            resp.raise_for_status()
            search_data = resp.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Tempo unavailable: {exc}") from exc

    traces_data = search_data.get("traces") or search_data.get("data") or []
    return {"items": traces_data[:limit]}


@router.get("/settings")
def get_settings(_: User = Depends(get_current_admin)):
    return get_runtime_settings()


@router.put("/settings")
def put_settings(
    payload: dict[str, Any],
    _: User = Depends(get_current_admin),
):
    clean_payload = {}
    if "rewrite_enabled" in payload:
        clean_payload["rewrite_enabled"] = bool(payload["rewrite_enabled"])
    if "rerank_enabled" in payload:
        clean_payload["rerank_enabled"] = bool(payload["rerank_enabled"])
    if "max_tokens" in payload:
        clean_payload["max_tokens"] = max(128, min(int(payload["max_tokens"]), 4096))
    if "top_k" in payload:
        clean_payload["top_k"] = max(1, min(int(payload["top_k"]), 20))

    return update_runtime_settings(clean_payload)
