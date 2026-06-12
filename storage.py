from __future__ import annotations
import json
import os
import sqlite3
from datetime import datetime

from config import ExperimentConfig
from models import GameResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    backend TEXT NOT NULL,
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
    seed INTEGER
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
    elapsed_seconds REAL
);

CREATE INDEX IF NOT EXISTS idx_games_experiment ON games(experiment_id);
CREATE INDEX IF NOT EXISTS idx_rounds_game ON rounds(game_id);
"""


def get_conn(db_path: str) -> sqlite3.Connection:
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_experiment(conn: sqlite3.Connection, cfg: ExperimentConfig) -> int:
    cur = conn.execute(
        """INSERT INTO experiments (
            created_at, backend,
            encryptor_model, decoder_model, interceptor_model,
            encryptor_temperature, decoder_temperature, interceptor_temperature,
            max_tokens, prompt_version,
            num_keywords, code_length, rounds_per_game,
            num_keyword_sets, games_per_keyword_set, seed
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now().isoformat(timespec="seconds"),
            cfg.backend,
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
        ),
    )
    conn.commit()
    return cur.lastrowid


def insert_game(conn: sqlite3.Connection, experiment_id: int, result: GameResult) -> int:
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

    conn.executemany(
        """INSERT INTO rounds (
            game_id, round_number, code, clues,
            interceptor_guess, decoder_guess,
            decoder_correct, interceptor_correct, successful,
            encryptor_fallback, interceptor_fallback, decoder_fallback,
            elapsed_seconds
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
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
                int(r.encryptor_fallback),
                int(r.interceptor_fallback),
                int(r.decoder_fallback),
                r.elapsed_seconds,
            )
            for r in result.rounds
        ],
    )
    conn.commit()
    return game_id
