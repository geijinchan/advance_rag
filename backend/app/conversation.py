"""Conversation memory — multi-turn sessions with follow-up resolution.

Design (kept deliberately small and transparent):
* ``SessionStore`` keeps threads of turns (question, answer, route,
  citations) with a ``MAX_TURNS`` cap per session and a total-session cap.
  Sessions persist to a JSON file (atomic write) so history survives service
  restarts — the demo keeps working after a redeploy; production would move
  to Redis/Postgres with the same interface.
* ``resolve_followup`` carries conversational context into short/elliptical
  follow-ups: "and the X300?" or "what about its accuracy?" inherit the
  entity (model code / error code) from the previous grounded turn. The
  mechanism is heuristic and fully observable — the resolved query and the
  inherited entities are returned so the UI can show exactly what was
  understood.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from app.persist import load_json, save_json

MAX_TURNS = 20

# Entity patterns worth carrying across turns (model codes, error codes, ports).
_ENTITY_RES = [
    re.compile(r"\b([xsc]\d{2,3})\b", re.I),          # x100 x200 x300 s200 s350
    re.compile(r"\b(e1[0-9]{2})\b", re.I),            # E101..E120
    re.compile(r"\b(v\d\.\d(?:\.\d)?)\b", re.I),      # v3.2.1
]

# Elliptical follow-up cues: pronouns without their referent.
_FOLLOWUP_CUE_RE = re.compile(
    r"^(and|also|what about|how about|tell me more|more|why|and the|its|it)\b|"
    r"\b(its|it)\b|\?$",
    re.I,
)


@dataclass
class Turn:
    question: str
    effective_question: str
    answer: str
    route: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    agent_path: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    follow_up_applied: bool = False


@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=time.time)
    turns: list[Turn] = field(default_factory=list)

    @property
    def last_activity(self) -> float:
        return self.turns[-1].ts if self.turns else self.created_at

    def preview(self) -> str:
        if not self.turns:
            return "(empty)"
        last = self.turns[-1]
        text = last.question if len(last.question) <= 48 else last.question[:45] + "…"
        return f"{len(self.turns)} turn{'s' if len(self.turns) != 1 else ''} · {text}"

    def entities(self) -> list[str]:
        """Distinct entity codes mentioned across the session's questions."""
        found: list[str] = []
        for t in self.turns:
            for rx in _ENTITY_RES:
                found.extend(m.group(1).lower() for m in rx.finditer(t.effective_question))
        out: list[str] = []
        for e in found:
            if e not in out:
                out.append(e)
        return out

    def last_grounded_entities(self) -> list[str]:
        for t in reversed(self.turns):
            if t.route == "document":
                ents: list[str] = []
                for rx in _ENTITY_RES:
                    ents.extend(m.group(1).lower() for m in rx.finditer(t.effective_question))
                if ents:
                    return ents
        return []


class SessionStore:
    """Session registry with JSON persistence (single-process demo; Redis in
    prod). Restores prior sessions on boot; mutations persist atomically."""

    def __init__(self, max_sessions: int = 200, persist_path: str | None = None) -> None:
        self._sessions: dict[str, Session] = {}
        self._max_sessions = max_sessions
        self._persist_path = persist_path
        self._restore()

    def _restore(self) -> None:
        if not self._persist_path:
            return
        data = load_json(self._persist_path)
        if not isinstance(data, dict):
            return
        for sid, sdata in data.get("sessions", {}).items():
            if not isinstance(sdata, dict):
                continue
            turns: list[Turn] = []
            for t in sdata.get("turns", []):
                if not isinstance(t, dict):
                    continue
                try:
                    turns.append(
                        Turn(
                            question=str(t.get("question", "")),
                            effective_question=str(t.get("effective_question", "")),
                            answer=str(t.get("answer", "")),
                            route=str(t.get("route", "")),
                            citations=list(t.get("citations", [])),
                            agent_path=list(t.get("agent_path", [])),
                            ts=float(t.get("ts", time.time())),
                            follow_up_applied=bool(t.get("follow_up_applied", False)),
                        )
                    )
                except (TypeError, ValueError):
                    continue
            self._sessions[str(sid)] = Session(
                id=str(sid),
                created_at=float(sdata.get("created_at", time.time())),
                turns=turns,
            )
        if self._sessions:
            import logging

            logging.getLogger("rag-agent").info(
                "restored %d sessions (%d turns) from %s",
                len(self._sessions),
                sum(len(s.turns) for s in self._sessions.values()),
                self._persist_path,
            )

    def _persist(self) -> None:
        if not self._persist_path:
            return
        save_json(
            self._persist_path,
            {
                "sessions": {
                    s.id: {"created_at": s.created_at, "turns": [self._turn_dict(t) for t in s.turns]}
                    for s in self._sessions.values()
                }
            },
        )

    @staticmethod
    def _turn_dict(t: Turn) -> dict[str, Any]:
        return {
            "question": t.question,
            "effective_question": t.effective_question,
            "answer": t.answer,
            "route": t.route,
            "citations": t.citations,
            "agent_path": t.agent_path,
            "ts": t.ts,
            "follow_up_applied": t.follow_up_applied,
        }

    # ------------------------------------------------------------- lifecycle
    def get_or_create(self, session_id: str | None) -> Session:
        sid = (session_id or "").strip() or uuid.uuid4().hex[:10]
        s = self._sessions.get(sid)
        if s is None:
            if len(self._sessions) >= self._max_sessions:
                self._evict_oldest()
            s = Session(id=sid)
            self._sessions[sid] = s
        return s

    def delete(self, session_id: str) -> bool:
        existed = self._sessions.pop(session_id, None) is not None
        if existed:
            self._persist()
        return existed

    def clear(self) -> None:
        self._sessions.clear()
        self._persist()

    # ---------------------------------------------------------------- access
    def history(self, session_id: str) -> list[dict[str, Any]] | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        return [self._turn_payload(t) for t in s.turns]

    def list_sessions(self, limit: int = 25) -> list[dict[str, Any]]:
        sessions = sorted(self._sessions.values(), key=lambda s: s.last_activity, reverse=True)
        return [
            {
                "id": s.id,
                "turns": len(s.turns),
                "preview": s.preview(),
                "created_at": round(s.created_at, 1),
                "last_activity": round(s.last_activity, 1),
                "entities": s.entities()[:6],
            }
            for s in sessions[:limit]
        ]

    def stats(self) -> dict[str, Any]:
        total_turns = sum(len(s.turns) for s in self._sessions.values())
        followups = sum(1 for s in self._sessions.values() for t in s.turns if t.follow_up_applied)
        return {"sessions": len(self._sessions), "total_turns": total_turns, "follow_ups_resolved": followups}

    # ------------------------------------------------------------------ write
    def append_turn(self, session_id: str, turn: Turn) -> Session:
        s = self.get_or_create(session_id)
        s.turns.append(turn)
        if len(s.turns) > MAX_TURNS:
            s.turns = s.turns[-MAX_TURNS:]
        self._persist()
        return s

    @staticmethod
    def _turn_payload(t: Turn) -> dict[str, Any]:
        return {
            "question": t.question,
            "effective_question": t.effective_question,
            "answer": t.answer,
            "route": t.route,
            "citations": t.citations,
            "agent_path": t.agent_path,
            "ts": round(t.ts, 1),
            "follow_up_applied": t.follow_up_applied,
        }

    def _evict_oldest(self) -> None:
        if not self._sessions:
            return
        oldest = min(self._sessions.values(), key=lambda s: s.last_activity)
        self._sessions.pop(oldest.id, None)


def resolve_followup(session: Session, question: str) -> tuple[str, bool, list[str]]:
    """Resolve an elliptical follow-up into a self-contained question.

    Returns ``(effective_question, follow_up_applied, inherited_entities)``.
    Only fires when the question itself carries no entity AND no strong
    corpus term — i.e. it genuinely needs the previous turn's subject.
    """
    q = question.strip()
    has_entity = any(rx.search(q) for rx in _ENTITY_RES)
    if has_entity or not session.turns:
        return q, False, []

    inherited = session.last_grounded_entities()
    if not inherited:
        return q, False, []

    # Heuristic: question is short or starts with a follow-up cue.
    is_short = len(q.split()) <= 6
    has_cue = bool(_FOLLOWUP_CUE_RE.search(q))
    if not (is_short or has_cue):
        return q, False, []

    effective = f"{q} {' '.join(inherited[:2])}".strip()
    return effective, True, inherited[:2]
