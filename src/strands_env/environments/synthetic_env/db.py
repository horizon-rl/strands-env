"""SQLite database creation and management for synthetic environments.

Creates databases from AWM-format schema definitions and sample data.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


def create_database(db_path: Path | str, db_schema: dict, sample_data: dict) -> None:
    """Create and populate a SQLite database from schema and sample data.

    Args:
        db_path: File path for the SQLite database.
        db_schema: Schema dict with ``tables`` list, each containing ``ddl`` and ``indexes``.
        sample_data: Sample data dict with ``tables`` list, each containing ``insert_statements``.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing DB file to start fresh
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")

    try:
        # Create tables
        for table in db_schema.get("tables", []):
            ddl = table.get("ddl", "")
            if ddl:
                cursor.execute(ddl)
                logger.debug("Created table: %s", table.get("name", "unknown"))

            # Create indexes
            for index_ddl in table.get("indexes", []):
                try:
                    cursor.execute(index_ddl)
                except sqlite3.OperationalError as e:
                    # Index may already exist from DDL or conflict — skip
                    logger.debug("Skipping index: %s", e)

        # Insert sample data
        for table_data in sample_data.get("tables", []):
            table_name = table_data.get("table_name", "unknown")
            for stmt in table_data.get("insert_statements", []):
                try:
                    cursor.execute(stmt)
                except sqlite3.Error as e:
                    logger.warning("INSERT failed for table '%s': %s — statement: %.100s...", table_name, e, stmt)

        conn.commit()
        logger.info("Database created at %s", db_path)
    finally:
        conn.close()


def copy_database(src_path: Path | str, dst_path: Path | str) -> None:
    """Copy a SQLite database file for snapshot comparison.

    Args:
        src_path: Source database file path.
        dst_path: Destination path for the copy.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src_path), str(dst_path))
    logger.debug("Database copied: %s -> %s", src_path, dst_path)
