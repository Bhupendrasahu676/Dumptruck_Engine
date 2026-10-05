"""SQLite engine lifecycle storage. Unknown IDs raise KeyError; bad writes ValueError."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

WEAR_KEYS = ("lubrication", "bearing", "cooling", "combustion")
DISPOSITIONS = frozenset({"confirmed_fault", "false_alert", "sensor_issue", "normal_variation", "uncertain"})


def _now():
    return datetime.now(timezone.utc).isoformat()


def _number(value, label, minimum=0, maximum=None):
    try:
        valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
    except (OverflowError, TypeError):
        valid = False
    if not valid or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{label} must be a finite number in its allowed range")
    return value


def _identifier(value, label):
    try:
        if not isinstance(value, str):
            raise ValueError()
        UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ValueError(f"{label} must be a UUID string") from exc
    return value


def _text(value, label, required=False):
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"{label} must be {'nonempty ' if required else ''}text")
    return value


def _json(value):
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings")
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            _number(item, "JSON number", minimum=None)
        elif item is not None and not isinstance(item, (str, bool)):
            raise ValueError("Results must contain only JSON-compatible values")
    try:
        check(value)
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("Results must contain finite, JSON-compatible values") from exc


def _wear(value, partial=False):
    if not isinstance(value, dict) or set(value) - set(WEAR_KEYS) or (not partial and set(value) != set(WEAR_KEYS)):
        raise ValueError(f"wear/repairs must use keys: {', '.join(WEAR_KEYS)}")
    for key, fraction in value.items():
        _number(fraction, key, 0, 1)


class EngineStore:
    """File-backed store, also supporting ':memory:' and context-manager cleanup.

    Each operation closes its connection. In-memory stores retain an anchor until
    close(). Writes serialize stale-state validation and updates in one transaction.
    Read methods never initialize, update, or repair stored records.
    """

    def __init__(self, path):
        self.path = str(path)
        self._closed = False
        self._anchor = None
        self._uri = self.path == ":memory:"
        if self._uri:
            self._database = f"file:engine_lab_{uuid4().hex}?mode=memory&cache=shared"
            self._anchor = sqlite3.connect(self._database, uri=True)
        else:
            target = Path(path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            self._database = str(target)
        try:
            with self._connection(write=True) as connection:
                for statement in (
                    """CREATE TABLE IF NOT EXISTS engines (
                        engine_id TEXT PRIMARY KEY, name TEXT NOT NULL,
                        operating_hours REAL NOT NULL, wear TEXT NOT NULL,
                        current_overhaul_id TEXT NOT NULL, created_at TEXT NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS overhauls (
                        overhaul_id TEXT PRIMARY KEY,
                        engine_id TEXT NOT NULL REFERENCES engines(engine_id),
                        operating_hours REAL NOT NULL, notes TEXT NOT NULL,
                        repairs TEXT NOT NULL, wear_before TEXT NOT NULL,
                        wear_after TEXT NOT NULL, is_baseline INTEGER NOT NULL,
                        created_at TEXT NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS tests (
                        test_id TEXT PRIMARY KEY,
                        engine_id TEXT NOT NULL REFERENCES engines(engine_id),
                        overhaul_id TEXT NOT NULL REFERENCES overhauls(overhaul_id),
                        initial_hours REAL NOT NULL, operating_hours REAL NOT NULL,
                        duration_s REAL NOT NULL, payload TEXT NOT NULL,
                        created_at TEXT NOT NULL)""",
                    """CREATE TABLE IF NOT EXISTS feedback (
                        feedback_id TEXT PRIMARY KEY,
                        test_id TEXT NOT NULL REFERENCES tests(test_id),
                        sample_index INTEGER NOT NULL, disposition TEXT NOT NULL,
                        cause TEXT NOT NULL, evidence TEXT NOT NULL,
                        technician TEXT NOT NULL, created_at TEXT NOT NULL)""",
                    "CREATE INDEX IF NOT EXISTS overhaul_engine_idx ON overhauls(engine_id)",
                    "CREATE INDEX IF NOT EXISTS test_engine_idx ON tests(engine_id)",
                    "CREATE INDEX IF NOT EXISTS feedback_test_idx ON feedback(test_id)",
                ):
                    connection.execute(statement)
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _connection(self, write=False):
        if self._closed:
            raise RuntimeError("EngineStore is closed")
        connection = sqlite3.connect(self._database, uri=self._uri, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.commit()
        except BaseException:
            if write:
                connection.rollback()
            raise
        finally:
            connection.close()

    def close(self):
        if self._anchor is not None:
            self._anchor.close()
            self._anchor = None
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("EngineStore is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def _engine(connection, engine_id):
        _identifier(engine_id, "engine_id")
        row = connection.execute("SELECT * FROM engines WHERE engine_id = ?", (engine_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown engine: {engine_id}")
        return dict(row, wear=json.loads(row["wear"]))

    @staticmethod
    def _test(connection, test_id):
        _identifier(test_id, "test_id")
        row = connection.execute("SELECT * FROM tests WHERE test_id = ?", (test_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown test: {test_id}")
        return row

    @staticmethod
    def _overhaul(row):
        result = dict(row)
        for key in ("repairs", "wear_before", "wear_after"):
            result[key] = json.loads(result[key])
        result["is_baseline"] = bool(result["is_baseline"])
        return result

    def create_engine(self, name, initial_hours=0):
        name = _text(name, "name", required=True).strip()
        _number(initial_hours, "initial_hours")
        engine_id, overhaul_id, created_at = str(uuid4()), str(uuid4()), _now()
        wear = _json(dict.fromkeys(WEAR_KEYS, 0.0))
        with self._connection(write=True) as connection:
            connection.execute("INSERT INTO engines VALUES (?, ?, ?, ?, ?, ?)",
                               (engine_id, name, initial_hours, wear, overhaul_id, created_at))
            connection.execute("INSERT INTO overhauls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (overhaul_id, engine_id, initial_hours, "Initial baseline", "{}", wear, wear, 1, created_at))
            return self._engine(connection, engine_id)

    def list_engines(self):
        with self._connection() as connection:
            rows = connection.execute("SELECT * FROM engines ORDER BY created_at, rowid").fetchall()
            return [dict(row, wear=json.loads(row["wear"])) for row in rows]

    def get_engine(self, engine_id):
        with self._connection() as connection:
            return self._engine(connection, engine_id)

    def create_overhaul(self, engine_id, notes, repairs):
        _text(notes, "notes")
        _wear(repairs, partial=True)
        repairs_json = _json(repairs)
        repairs = json.loads(repairs_json)
        with self._connection(write=True) as connection:
            engine = self._engine(connection, engine_id)
            before = engine["wear"]
            after = {key: value * (1 - repairs.get(key, 0)) for key, value in before.items()}
            overhaul_id, created_at = str(uuid4()), _now()
            connection.execute("INSERT INTO overhauls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (overhaul_id, engine_id, engine["operating_hours"], notes,
                                repairs_json, _json(before), _json(after), 0, created_at))
            connection.execute("UPDATE engines SET wear = ?, current_overhaul_id = ? WHERE engine_id = ?",
                               (_json(after), overhaul_id, engine_id))
            return self._overhaul(connection.execute("SELECT * FROM overhauls WHERE overhaul_id = ?", (overhaul_id,)).fetchone())

    def list_overhauls(self, engine_id):
        with self._connection() as connection:
            self._engine(connection, engine_id)
            rows = connection.execute("SELECT * FROM overhauls WHERE engine_id = ? ORDER BY created_at, rowid", (engine_id,)).fetchall()
            return [self._overhaul(row) for row in rows]

    @staticmethod
    def _validate_result(result):
        if not isinstance(result, dict) or not isinstance(result.get("config"), dict):
            raise ValueError("Test result and config must be objects")
        config = result["config"]
        for key in ("engine_id", "overhaul_id", "test_id"):
            _identifier(config.get(key), key)
        _number(config.get("initial_hours"), "initial_hours")
        for key in ("duration_s", "timestep_s"):
            if _number(config.get(key), key) == 0:
                raise ValueError(f"{key} must be positive")
        if not isinstance(config.get("seed"), int) or isinstance(config["seed"], bool):
            raise ValueError("seed must be an integer")
        for key in ("start_condition", "payload_class"):
            _text(config.get(key), key, required=True)
        _wear(result.get("final_wear"))
        samples, analyses = result.get("samples"), result.get("analyses")
        if not isinstance(samples, list) or not samples:
            raise ValueError("A test requires nonempty samples")
        if not isinstance(analyses, list) or len(analyses) != len(samples):
            raise ValueError("analyses must be a list aligned with samples")
        previous = -1
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict):
                raise ValueError(f"sample {index} must be an object")
            elapsed = _number(sample.get("elapsed_s"), "elapsed_s", 0, config["duration_s"])
            if elapsed <= previous:
                raise ValueError("Sample elapsed_s must be strictly increasing")
            previous = elapsed
            hours = _number(sample.get("operating_hours"), "operating_hours")
            if not math.isclose(hours, config["initial_hours"] + elapsed / 3600, rel_tol=0, abs_tol=1e-8):
                raise ValueError("Sample operating_hours does not match its elapsed time")
            for key in ("context", "observations", "truth"):
                if not isinstance(sample.get(key), dict):
                    raise ValueError(f"sample {index} requires {key} object")
            if not isinstance(analyses[index], dict):
                raise ValueError(f"analysis {index} must be an object")
        if samples[0]["elapsed_s"] != 0 or samples[-1]["elapsed_s"] != config["duration_s"]:
            raise ValueError("Saved test must include both elapsed-time endpoints")
        final_hours = _number(config["initial_hours"] + config["duration_s"] / 3600, "final operating_hours")
        if final_hours <= config["initial_hours"]:
            raise ValueError("Test duration is too small to advance operating_hours")
        return config

    def save_test(self, result):
        """Atomically save a full result; identical retries return the original result."""
        payload = _json(result)
        result = json.loads(payload)
        config = self._validate_result(result)
        with self._connection(write=True) as connection:
            existing = connection.execute("SELECT payload FROM tests WHERE test_id = ?", (config["test_id"],)).fetchone()
            if existing is not None:
                if existing["payload"] != payload:
                    raise ValueError("test_id already exists with different content")
                return json.loads(existing["payload"])
            engine = self._engine(connection, config["engine_id"])
            if config["overhaul_id"] != engine["current_overhaul_id"]:
                raise ValueError("Test overhaul_id is not this engine's current overhaul")
            if config["initial_hours"] != engine["operating_hours"]:
                raise ValueError("Stale test initial_hours; reload the engine before running a test")
            final_hours = config["initial_hours"] + config["duration_s"] / 3600
            connection.execute("INSERT INTO tests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (config["test_id"], config["engine_id"], config["overhaul_id"],
                                config["initial_hours"], final_hours, config["duration_s"], payload, _now()))
            connection.execute("UPDATE engines SET operating_hours = ?, wear = ? WHERE engine_id = ?",
                               (final_hours, _json(result["final_wear"]), config["engine_id"]))
            return result

    def list_tests(self, engine_id):
        with self._connection() as connection:
            self._engine(connection, engine_id)
            rows = connection.execute("""SELECT test_id, engine_id, overhaul_id, initial_hours,
                                      operating_hours, duration_s, created_at FROM tests
                                      WHERE engine_id = ? ORDER BY operating_hours, rowid""", (engine_id,)).fetchall()
            return [dict(row) for row in rows]

    def get_test(self, test_id):
        with self._connection() as connection:
            return json.loads(self._test(connection, test_id)["payload"])

    def record_feedback(self, test_id, sample_index, disposition, cause="", evidence="", technician=""):
        if not isinstance(disposition, str) or disposition not in DISPOSITIONS:
            raise ValueError(f"Unknown feedback disposition: {disposition}")
        if not isinstance(sample_index, int) or isinstance(sample_index, bool) or sample_index < 0:
            raise ValueError("sample_index must be a nonnegative integer")
        for key, value in (("cause", cause), ("evidence", evidence), ("technician", technician)):
            _text(value, key, required=disposition == "confirmed_fault")
        with self._connection(write=True) as connection:
            test = json.loads(self._test(connection, test_id)["payload"])
            if sample_index >= len(test["samples"]):
                raise ValueError("sample_index is outside the saved test")
            record = dict(feedback_id=str(uuid4()), test_id=test_id, sample_index=sample_index,
                          disposition=disposition, cause=cause, evidence=evidence,
                          technician=technician, created_at=_now())
            connection.execute("INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?)", tuple(record.values()))
            return record

    def list_feedback(self, test_id):
        with self._connection() as connection:
            self._test(connection, test_id)
            rows = connection.execute("SELECT * FROM feedback WHERE test_id = ? ORDER BY created_at, rowid", (test_id,)).fetchall()
            return [dict(row) for row in rows]

    def health_history(self, engine_id):
        """Return each test's last valid observed health score, without invented outcomes."""
        history = []
        with self._connection() as connection:
            self._engine(connection, engine_id)
            rows = connection.execute("SELECT * FROM tests WHERE engine_id = ? ORDER BY operating_hours, rowid", (engine_id,)).fetchall()
            for row in rows:
                result = json.loads(row["payload"])
                for index in range(len(result["analyses"]) - 1, -1, -1):
                    analysis = result["analyses"][index]
                    health = analysis.get("health")
                    if not isinstance(health, dict) or health.get("valid") is False:
                        continue
                    score = health.get("score")
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        continue
                    quality = analysis.get("quality", {})
                    if isinstance(quality, dict) and quality.get("valid") is False:
                        continue
                    if analysis.get("state") in ("invalid", "insufficient_data", "unavailable"):
                        continue
                    history.append(dict(test_id=row["test_id"], overhaul_id=row["overhaul_id"],
                                        operating_hours=result["samples"][index]["operating_hours"], health_score=score))
                    break
        return history
