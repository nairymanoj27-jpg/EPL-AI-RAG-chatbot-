"""SQLite conversation memory tests."""

from __future__ import annotations

from pathlib import Path

from app.services.conversation_store import SQLiteConversationStore


def make_store(tmp_path: Path) -> SQLiteConversationStore:
    return SQLiteConversationStore(tmp_path / "memory.sqlite3")


def test_history_is_isolated_by_conversation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.append_exchange("conversation-a", "A question", "A answer")
    store.append_exchange("conversation-b", "B question", "B answer")

    assert [turn.content for turn in store.get_history("conversation-a")] == [
        "A question",
        "A answer",
    ]
    assert [turn.content for turn in store.get_history("conversation-b")] == [
        "B question",
        "B answer",
    ]
    assert store.get_history("conversation-c") == []


def test_history_is_chronological_and_limited(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    for index in range(4):
        store.append_exchange("conv", f"q{index}", f"a{index}")

    history = store.get_history("conv", max_turns=4, max_characters=10_000)

    assert [turn.content for turn in history] == ["q2", "a2", "q3", "a3"]


def test_history_character_budget_keeps_newest_turns(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.append_exchange("conv", "old question " * 20, "old answer " * 20)
    store.append_exchange("conv", "new question", "new answer")

    history = store.get_history("conv", max_turns=10, max_characters=120)

    assert history[-1].content == "new answer"
    assert sum(len(turn.content) for turn in history) <= 120


def test_clear_removes_only_one_conversation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.append_exchange("one", "q", "a")
    store.append_exchange("two", "q", "a")

    assert store.clear("one") is True
    assert store.clear("one") is False
    assert store.exists("one") is False
    assert store.exists("two") is True


def test_empty_history_when_limits_are_zero(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.append_exchange("conv", "q", "a")

    assert store.get_history("conv", max_turns=0) == []
    assert store.get_history("conv", max_characters=0) == []


def test_database_is_created_and_reusable(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "memory.sqlite3"
    store = SQLiteConversationStore(path)
    store.append_exchange("conv", "q", "a")

    reopened = SQLiteConversationStore(path)
    assert reopened.exists("conv") is True
    assert len(reopened.get_history("conv")) == 2
