"""Export the SQLite outbox to append-only JSONL, repairing torn tails."""

import json
import os
from vibeforge.storage.atomic import canonical_json
from vibeforge.storage.locks import exclusive_lock


class EventLog:
    def __init__(self, database, campaign_id, directory):
        self.database, self.campaign_id = database, campaign_id
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "events.jsonl"

    def sync(self):
        with exclusive_lock(self.path.with_suffix(".lock")):
            last, valid_end = 0, 0
            if self.path.exists():
                with self.path.open("rb") as stream:
                    for line in stream:
                        try:
                            event = json.loads(line)
                            if not line.endswith(b"\n") or event["sequence"] <= last:
                                break
                        except (ValueError, KeyError):
                            break
                        last, valid_end = event["sequence"], stream.tell()
                with self.path.open("r+b") as stream:
                    stream.truncate(valid_end)
            with self.path.open("a", encoding="utf-8") as stream:
                for event in self.database.events(self.campaign_id, last):
                    stream.write(canonical_json(event) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def emit(self, event_type, **fields):
        event = self.database.emit(self.campaign_id, event_type, **fields)
        self.sync()
        return event
