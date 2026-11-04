"""SQLite is the authority for state, operations, and the durable event outbox."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from vibeforge.contracts.event import make_event
from vibeforge.storage.atomic import canonical_json


class Database:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=10, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS campaigns (id TEXT PRIMARY KEY, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS iterations (
                campaign_id TEXT NOT NULL, number INTEGER NOT NULL, data TEXT NOT NULL,
                PRIMARY KEY(campaign_id, number));
            CREATE TABLE IF NOT EXISTS operations (
                id TEXT PRIMARY KEY, campaign_id TEXT NOT NULL, kind TEXT NOT NULL,
                status TEXT NOT NULL, data TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL, data TEXT NOT NULL);
        """)

    def close(self):
        self.connection.close()

    @contextmanager
    def transaction(self):
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def get(self, campaign_id):
        row = self.connection.execute("SELECT data FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown campaign: {campaign_id}")
        return json.loads(row["data"])

    def list_campaigns(self):
        return [json.loads(row[0]) for row in self.connection.execute("SELECT data FROM campaigns ORDER BY id")]

    def put(self, campaign_id, data):
        self.connection.execute("INSERT INTO campaigns VALUES (?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data",
                                (campaign_id, canonical_json(data)))

    def emit(self, campaign_id, event_type, **fields):
        event = make_event(campaign_id, event_type, **fields)
        cursor = self.connection.execute("INSERT INTO events(campaign_id,data) VALUES (?,?)",
                                         (campaign_id, canonical_json(event)))
        event["sequence"] = cursor.lastrowid
        return event

    def update(self, campaign_id, changes, event_type="state_changed", **fields):
        with self.transaction():
            data = self.get(campaign_id)
            previous = data.get("state")
            data.update(changes)
            self.put(campaign_id, data)
            self.emit(campaign_id, event_type, state_before=previous, state_after=data.get("state"), **fields)
        return data

    def iteration(self, campaign_id, number, changes=None):
        row = self.connection.execute("SELECT data FROM iterations WHERE campaign_id=? AND number=?",
                                      (campaign_id, number)).fetchone()
        data = json.loads(row[0]) if row else {"iteration": number}
        if changes is not None:
            data.update(changes)
            self.connection.execute("INSERT INTO iterations VALUES (?,?,?) ON CONFLICT(campaign_id,number) "
                                    "DO UPDATE SET data=excluded.data", (campaign_id, number, canonical_json(data)))
        return data

    def iterations(self, campaign_id):
        return [json.loads(row[0]) for row in self.connection.execute(
            "SELECT data FROM iterations WHERE campaign_id=? ORDER BY number", (campaign_id,))]

    def events(self, campaign_id, after=0):
        for row in self.connection.execute("SELECT sequence,data FROM events WHERE campaign_id=? AND sequence>? "
                                           "ORDER BY sequence", (campaign_id, after)):
            yield {**json.loads(row["data"]), "sequence": row["sequence"]}
