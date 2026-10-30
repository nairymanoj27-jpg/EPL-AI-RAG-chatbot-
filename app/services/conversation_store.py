"""SQLite-backed conversation memory.

This is intentionally separate from retrieval.  Disabling it
(``CONVERSATION_STORAGE=none``) leaves the chatbot fully functional, just
without multi-turn context.  Each conversation is isolated by its ID.

SQLite is a reasonable single-process choice for a student project.  It is not
a multi-tenant production database.  Conversation IDs behave like bearer
tokens; use APP_API_KEY and/or upstream authentication for sensitive data.
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

_MAX_MESSAGE_CHARS = 20_000


@dataclass(frozen=True)
class ConversationTurn:
    role: Literal["user", "assistant"]
    content: str
    created_at: str | None = None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _truncate(text: str) -> str:
    text = text.strip()
    if len(text) <= _MAX_MESSAGE_CHARS:
        return text
    return text[: _MAX_MESSAGE_CHARS - 1].rstrip() + "…"


class SQLiteConversationStore:
    """Small, connection-per-operation SQLite store with WAL enabled."""

    def __init__(self, database_path: Path) -> None:
        self._path = database_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    @property
    def path(self) -> Path:
        return self._path

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL
                        REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON messages(conversation_id, id);
                """
            )

    def get_history(
        self,
        conversation_id: str,
        *,
        max_turns: int = 6,
        max_characters: int = 6_000,
    ) -> list[ConversationTurn]:
        """Return recent history in chronological order.

        ``max_turns`` counts user/assistant messages, so ``max_turns=6`` keeps
        roughly three exchanges.  The oldest messages are dropped first.
        """
        if max_turns <= 0 or max_characters <= 0:
            return []
        limit = max_turns
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        selected: list[ConversationTurn] = []
        used = 0
        for row in rows:
            content = str(row["content"])
            if used + len(content) > max_characters:
                remaining = max_characters - used
                if remaining < 100:
                    break
                content = content[: remaining - 1].rstrip() + "…"
            role: Literal["user", "assistant"] = row["role"]
            selected.append(
                ConversationTurn(
                    role=role,
                    content=content,
                    created_at=str(row["created_at"]),
                )
            )
            used += len(content)
        selected.reverse()
        return selected

    def append_exchange(
        self,
        conversation_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        user_message = _truncate(user_message)
        assistant_message = _truncate(assistant_message)
        now = _utc_now()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO conversations (id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (conversation_id, now, now),
            )
            connection.executemany(
                """
                INSERT INTO messages (conversation_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (conversation_id, "user", user_message, now),
                    (conversation_id, "assistant", assistant_message, now),
                ],
            )

    def clear(self, conversation_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
            return cursor.rowcount > 0

    def exists(self, conversation_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            return row is not None

    def close(self) -> None:
        """No persistent connection is held; included for a uniform interface."""
        return None

    def __repr__(self) -> str:
        return f"SQLiteConversationStore(path={str(self._path)!r})"


def history_to_messages(history: Sequence[ConversationTurn]) -> list[dict[str, str]]:
    """Optional helper for debugging/tests; keeps roles explicit."""
    return [{"role": turn.role, "content": turn.content} for turn in history]
