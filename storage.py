from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime

import prompts
from config import ExperimentConfig
from models import GameResult, count_multiword

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    encryptor_backend TEXT NOT NULL,
    decoder_backend TEXT NOT NULL,
    interceptor_backend TEXT NOT NULL,
    encryptor_model TEXT NOT NULL,
    decoder_model TEXT NOT NULL,
    interceptor_model TEXT NOT NULL,
    encryptor_temperature REAL,
    decoder_temperature REAL,
    interceptor_temperature REAL,
    max_tokens INTEGER,
    prompt_version INTEGER NOT NULL,
    num_keywords INTEGER NOT NULL,
    code_length INTEGER NOT NULL,
    rounds_per_game INTEGER NOT NULL,
    num_keyword_sets INTEGER NOT NULL,
    games_per_keyword_set INTEGER NOT NULL,
    seed INTEGER,
    keyword_seed INTEGER,
    keyword_sets_file TEXT,
    prompt_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS games (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    keyword_set_id INTEGER NOT NULL,
    game_number INTEGER NOT NULL,
    keywords TEXT NOT NULL,
    total_seconds REAL
);

CREATE TABLE IF NOT EXISTS rounds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL REFERENCES games(id),
    round_number INTEGER NOT NULL,
    code TEXT NOT NULL,
    clues TEXT NOT NULL,
    interceptor_guess TEXT NOT NULL,
    decoder_guess TEXT NOT NULL,
    decoder_correct INTEGER NOT NULL,
    interceptor_correct INTEGER NOT NULL,
    successful INTEGER NOT NULL,
    encryptor_fallback INTEGER NOT NULL DEFAULT 0,
    interceptor_fallback INTEGER NOT NULL DEFAULT 0,
    decoder_fallback INTEGER NOT NULL DEFAULT 0,
    multiword_clues INTEGER NOT NULL DEFAULT 0,
    elapsed_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_games_experiment ON games(experiment_id);
CREATE INDEX IF NOT EXISTS idx_rounds_game ON rounds(game_id);
"""

# Applied only on the write path (insert_experiment / insert_game), so opening an
# old database for analysis never alters it.
CALLS_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    round_id INTEGER NOT NULL REFERENCES rounds(id),
    agent TEXT NOT NULL,
    model_requested TEXT NOT NULL,
    model_served TEXT,
    provider TEXT,
    response_id TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    cost REAL,
    started_at TEXT,
    elapsed_seconds REAL,
    content TEXT NOT NULL,
    reasoning TEXT,
    attempt INTEGER NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_round ON calls(round_id);

CREATE TABLE IF NOT EXISTS game_status (
    experiment_id INTEGER NOT NULL REFERENCES experiments(id),
    keyword_set_id INTEGER NOT NULL,
    game_number INTEGER NOT NULL,
    status TEXT NOT NULL,          -- queued | running | done | error
    starts INTEGER NOT NULL DEFAULT 0,
    started_at TEXT,
    finished_at TEXT,
    last_error TEXT,
    PRIMARY KEY (experiment_id, keyword_set_id, game_number)
);
"""

QUEUED, RUNNING, DONE, ERROR = "queued", "running", "done", "error"
DB_BUSY_TIMEOUT_S = 60


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_experiment(conn: sqlite3.Connection, experiment_id: int):
    """Reconstruct ExperimentConfig and effective backends from a stored experiment row."""
    from config import AgentConfig, ExperimentConfig
    # Databases created before these columns existed: NULL means "same as seed" / "drawn, not read".
    kw_seed_col = "keyword_seed" if _has_column(conn, "experiments", "keyword_seed") else "NULL"
    kw_file_col = "keyword_sets_file" if _has_column(conn, "experiments", "keyword_sets_file") else "NULL"
    row = conn.execute(
        f"""SELECT encryptor_backend, decoder_backend, interceptor_backend,
                  encryptor_model, decoder_model, interceptor_model,
                  encryptor_temperature, decoder_temperature, interceptor_temperature,
                  max_tokens, prompt_version,
                  num_keywords, code_length, rounds_per_game,
                  num_keyword_sets, games_per_keyword_set, seed, {kw_seed_col}, {kw_file_col}
           FROM experiments WHERE id = ?""",
        (experiment_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Experiment #{experiment_id} not found.")
    (enc_b, dec_b, int_b,
     enc_m, dec_m, int_m,
     enc_t, dec_t, int_t,
     max_tok, prompt_v,
     num_kw, code_len, rounds,
     num_sets, games_per_set, seed, keyword_seed, keyword_sets_file) = row

    agents = AgentConfig(
        encryptor_model=enc_m,   encryptor_backend=enc_b,
        decoder_model=dec_m,     decoder_backend=dec_b,
        interceptor_model=int_m, interceptor_backend=int_b,
        encryptor_temperature=enc_t,
        decoder_temperature=dec_t,
        interceptor_temperature=int_t,
        max_tokens=max_tok,
        prompt_version=prompt_v,
    )
    cfg = ExperimentConfig(
        backend=enc_b,   # default backend = encryptor's (arbitrary, overridden per agent)
        num_keywords=num_kw,
        code_length=code_len,
        rounds_per_game=rounds,
        num_keyword_sets=num_sets,
        games_per_keyword_set=games_per_set,
        seed=seed,
        keyword_seed=keyword_seed,
        keyword_sets_file=keyword_sets_file,
        agents=agents,
    )
    return cfg


def stored_prompt_sha256(conn: sqlite3.Connection, experiment_id: int) -> str | None:
    """Hash of the prompts an experiment was run with; None for older databases."""
    if not _has_column(conn, "experiments", "prompt_sha256"):
        return None
    return conn.execute(
        "SELECT prompt_sha256 FROM experiments WHERE id = ?", (experiment_id,)
    ).fetchone()[0]


def enqueue_games(conn: sqlite3.Connection, experiment_id: int, keys: list[tuple[int, int]]) -> None:
    """Register every (keyword_set_id, game_number) of the experiment as queued."""
    _ensure_write_schema(conn)
    conn.executemany(
        "INSERT OR IGNORE INTO game_status (experiment_id, keyword_set_id, game_number, status) "
        "VALUES (?, ?, ?, ?)",
        [(experiment_id, ks, g, QUEUED) for ks, g in keys],
    )
    conn.commit()


def ensure_status_rows(conn: sqlite3.Connection, experiment_id: int,
                       keys: list[tuple[int, int]]) -> None:
    """Experiments created before game_status existed: rebuild it from the games table."""
    _ensure_write_schema(conn)
    if conn.execute("SELECT 1 FROM game_status WHERE experiment_id = ? LIMIT 1",
                    (experiment_id,)).fetchone():
        return
    enqueue_games(conn, experiment_id, keys)
    conn.execute(
        f"""UPDATE game_status SET status = '{DONE}' WHERE experiment_id = ?
            AND (keyword_set_id, game_number) IN
                (SELECT keyword_set_id, game_number FROM games WHERE experiment_id = ?)""",
        (experiment_id, experiment_id),
    )
    conn.commit()


def requeue_unfinished(conn: sqlite3.Connection, experiment_id: int) -> int:
    """Put back in the queue the games that failed or were interrupted mid-run."""
    n = conn.execute(
        f"UPDATE game_status SET status = '{QUEUED}' WHERE experiment_id = ? "
        f"AND status IN ('{RUNNING}', '{ERROR}')",
        (experiment_id,),
    ).rowcount
    conn.commit()
    return n


def queued_games(conn: sqlite3.Connection, experiment_id: int) -> list[tuple[int, int]]:
    return [tuple(r) for r in conn.execute(
        f"SELECT keyword_set_id, game_number FROM game_status WHERE experiment_id = ? "
        f"AND status = '{QUEUED}' ORDER BY keyword_set_id, game_number",
        (experiment_id,),
    )]


def set_status(conn: sqlite3.Connection, experiment_id: int, keyword_set_id: int,
               game_number: int, status: str, error: str | None = None) -> None:
    key = (experiment_id, keyword_set_id, game_number)
    where = "WHERE experiment_id = ? AND keyword_set_id = ? AND game_number = ?"
    if status == RUNNING:
        conn.execute(f"UPDATE game_status SET status = ?, starts = starts + 1, started_at = ?, "
                     f"finished_at = NULL {where}", (status, _now(), *key))
    elif status == QUEUED:
        conn.execute(f"UPDATE game_status SET status = ? {where}", (status, *key))
    else:
        conn.execute(f"UPDATE game_status SET status = ?, finished_at = ?, last_error = ? {where}",
                     (status, _now(), error, *key))
    conn.commit()


def status_summary(conn: sqlite3.Connection, experiment_id: int) -> dict:
    """Counts per status, plus the running and errored games (read-only)."""
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='game_status'").fetchone():
        return {}
    counts = dict(conn.execute(
        "SELECT status, COUNT(*) FROM game_status WHERE experiment_id = ? GROUP BY status",
        (experiment_id,)))
    running = conn.execute(
        f"SELECT keyword_set_id, game_number, started_at FROM game_status WHERE experiment_id = ? "
        f"AND status = '{RUNNING}' ORDER BY keyword_set_id", (experiment_id,)).fetchall()
    errors = conn.execute(
        f"SELECT keyword_set_id, game_number, starts, last_error FROM game_status WHERE experiment_id = ? "
        f"AND status = '{ERROR}' ORDER BY keyword_set_id", (experiment_id,)).fetchall()
    return {"counts": counts, "running": running, "errors": errors}


def get_conn(db_path: str) -> sqlite3.Connection:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # Several runs may write to the same database; each write takes milliseconds, so
    # waiting up to DB_BUSY_TIMEOUT_S for the other run's lock is plenty.
    conn = sqlite3.connect(db_path, timeout=DB_BUSY_TIMEOUT_S)
    conn.executescript(SCHEMA)
    _add_multiword_column(conn)
    conn.commit()
    return conn


def _add_multiword_column(conn: sqlite3.Connection) -> None:
    """Databases created before the column existed: add it and backfill from stored clues."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(rounds)")}
    if "multiword_clues" in columns:
        return
    conn.execute("ALTER TABLE rounds ADD COLUMN multiword_clues INTEGER NOT NULL DEFAULT 0")
    rows = conn.execute("SELECT id, clues FROM rounds").fetchall()
    conn.executemany(
        "UPDATE rounds SET multiword_clues = ? WHERE id = ?",
        [(count_multiword(json.loads(clues)), round_id) for round_id, clues in rows],
    )


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return column in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_write_schema(conn: sqlite3.Connection) -> None:
    """Migrate only when writing, so opening an old database for analysis never alters it."""
    for column, sql_type in (("keyword_seed", "INTEGER"), ("keyword_sets_file", "TEXT"),
                             ("prompt_sha256", "TEXT")):
        if not _has_column(conn, "experiments", column):
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {column} {sql_type}")
    conn.executescript(CALLS_SCHEMA)
    if not _has_column(conn, "calls", "started_at"):
        conn.execute("ALTER TABLE calls ADD COLUMN started_at TEXT")


def insert_experiment(conn: sqlite3.Connection, cfg: ExperimentConfig) -> int:
    _ensure_write_schema(conn)
    enc_backend = cfg.agents.encryptor_backend if cfg.agents.encryptor_backend is not None else cfg.backend
    dec_backend = cfg.agents.decoder_backend if cfg.agents.decoder_backend is not None else cfg.backend
    int_backend = cfg.agents.interceptor_backend if cfg.agents.interceptor_backend is not None else cfg.backend

    cur = conn.execute(
        """INSERT INTO experiments (
            created_at,
            encryptor_backend, decoder_backend, interceptor_backend,
            encryptor_model, decoder_model, interceptor_model,
            encryptor_temperature, decoder_temperature, interceptor_temperature,
            max_tokens, prompt_version,
            num_keywords, code_length, rounds_per_game,
            num_keyword_sets, games_per_keyword_set, seed, keyword_seed, keyword_sets_file,
            prompt_sha256
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            enc_backend,
            dec_backend,
            int_backend,
            cfg.agents.encryptor_model,
            cfg.agents.decoder_model,
            cfg.agents.interceptor_model,
            cfg.agents.encryptor_temperature,
            cfg.agents.decoder_temperature,
            cfg.agents.interceptor_temperature,
            cfg.agents.max_tokens,
            cfg.agents.prompt_version,
            cfg.num_keywords,
            cfg.code_length,
            cfg.rounds_per_game,
            cfg.num_keyword_sets,
            cfg.games_per_keyword_set,
            cfg.seed,
            cfg.keyword_seed,
            cfg.keyword_sets_file,
            prompts.prompts_sha256(),
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_game(conn: sqlite3.Connection, experiment_id: int, result: GameResult) -> int:
    _ensure_write_schema(conn)
    cur = conn.execute(
        """INSERT INTO games (experiment_id, keyword_set_id, game_number, keywords, total_seconds)
           VALUES (?, ?, ?, ?, ?)""",
        (
            experiment_id,
            result.keyword_set_id,
            result.game_id,
            json.dumps({kp.number: kp.word for kp in result.keywords}),
            result.total_seconds,
        ),
    )
    game_id = cur.lastrowid

    for r in result.rounds:
        round_id = conn.execute(
            """INSERT INTO rounds (
                game_id, round_number, code, clues,
                interceptor_guess, decoder_guess,
                decoder_correct, interceptor_correct, successful,
                multiword_clues, elapsed_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                game_id,
                r.round_number,
                json.dumps(list(r.code)),
                json.dumps(list(r.clues)),
                json.dumps(list(r.interceptor_guess)),
                json.dumps(list(r.decoder_guess)),
                int(r.decoder_correct),
                int(r.interceptor_correct),
                int(r.transmission_successful),
                r.multiword_clues,
                r.elapsed_seconds,
            ),
        ).lastrowid
        conn.executemany(
            """INSERT INTO calls (
                round_id, agent, model_requested, model_served, provider, response_id,
                prompt_tokens, completion_tokens, reasoning_tokens, cost,
                started_at, elapsed_seconds, content, reasoning, attempt, status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (round_id, agent, c.model_requested, c.model_served, c.provider, c.response_id,
                 c.prompt_tokens, c.completion_tokens, c.reasoning_tokens, c.cost,
                 c.started_at, c.elapsed_seconds, c.content, c.reasoning, c.attempt, c.status, c.error)
                for agent, c in r.calls
            ],
        )
    # Same transaction as the game rows, so a crash never leaves a saved game still "running".
    conn.execute(
        f"UPDATE game_status SET status = '{DONE}', finished_at = ?, last_error = NULL "
        "WHERE experiment_id = ? AND keyword_set_id = ? AND game_number = ?",
        (_now(), experiment_id, result.keyword_set_id, result.game_id),
    )
    conn.commit()
    return game_id
