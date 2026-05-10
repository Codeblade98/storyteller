"""SQLite-based story run persistence."""

import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from story_engine.core.executor import StoryRun


class SQLiteStoryLog:
    """Persists story runs to a SQLite database.

    Provides save and load operations for serializing story generation results.
    """
    def __init__(self, path: str | Path = "story_logs.sqlite3") -> None:
        """Initialize the SQLite story log.

        Args:
            path: Path to the SQLite database file.
        """
        self.path = Path(path)
        self._init_schema()

    def save(self, run: StoryRun) -> str:
        """Save a story run to the database.

        Args:
            run: StoryRun to persist.

        Returns:
            The unique ID of the saved story run.
        """
        story_id = str(uuid4())
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "insert into story_runs (id, payload_json) values (?, ?)",
                (story_id, run.model_dump_json()),
            )
        return story_id

    def load(self, story_id: str) -> dict[str, Any] | None:
        """Load a story run from the database.

        Args:
            story_id: Unique ID of the story run to load.

        Returns:
            Dictionary representation of the story run, or None if not found.
        """
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "select payload_json from story_runs where id = ?",
                (story_id,),
            ).fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def _init_schema(self) -> None:
        """Initialize the database schema if needed."""
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                create table if not exists story_runs (
                    id text primary key,
                    payload_json text not null,
                    created_at timestamp default current_timestamp
                )
                """
            )
