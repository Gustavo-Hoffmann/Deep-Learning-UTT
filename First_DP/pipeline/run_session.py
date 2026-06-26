#!/usr/bin/env python3
"""
Sessão de execução longa com pasta timestamp, log e retomada (--resume).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO


class RunSession:
    """Gerencia pasta timestampada, log incremental e estado para retomada."""

    def __init__(self, base_output_dir: Path, resume: bool = False) -> None:
        self.base_output_dir = base_output_dir
        self.runs_root = base_output_dir / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.run_dir = self._resolve_run_dir()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._log_file: TextIO | None = None
        self.latest_pointer = self.runs_root / "latest_run.txt"

    def _resolve_run_dir(self) -> Path:
        if self.resume:
            latest = self.runs_root / "latest_run.txt"
            if latest.is_file():
                path = Path(latest.read_text(encoding="utf-8").strip())
                if path.is_dir():
                    return path
            existing = sorted(self.runs_root.glob("20*"), key=lambda p: p.name, reverse=True)
            if existing:
                return existing[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.runs_root / stamp

    def open_log(self) -> None:
        self._log_file = (self.run_dir / "log.txt").open("a", encoding="utf-8")
        if self.resume:
            self.log("--- Retomada de execução ---")
        else:
            self.log(f"--- Nova execução: {self.run_dir.name} ---")

    def log(self, message: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line)
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def save_config(self, payload: dict[str, Any]) -> Path:
        path = self.run_dir / "config.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.latest_pointer.write_text(str(self.run_dir), encoding="utf-8")
        return path

    def append_results_csv(self, filename: str, row: dict[str, Any]) -> Path:
        import pandas as pd

        path = self.run_dir / filename
        df_new = pd.DataFrame([row])
        if path.is_file():
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(path, index=False)
        return path

    def load_completed_config_ids(self, filename: str, id_col: str = "config_id") -> set[str]:
        path = self.run_dir / filename
        if not path.is_file():
            return set()
        import pandas as pd

        df = pd.read_csv(path)
        if id_col not in df.columns:
            return set()
        return set(df[id_col].astype(str).tolist())

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def copy_best_to_configs(self, best_path: Path, configs_dir: Path) -> Path:
        import shutil

        configs_dir.mkdir(parents=True, exist_ok=True)
        dest = configs_dir / "best_config.json"
        shutil.copy2(best_path, dest)
        return dest


def tee_stdout(session: RunSession) -> None:
    """Redireciona stdout para log (opcional, leve)."""
    _orig = sys.stdout

    class _Tee:
        def write(self, data: str) -> None:
            _orig.write(data)
            if session._log_file and data.strip():
                session._log_file.write(data)

        def flush(self) -> None:
            _orig.flush()
            if session._log_file:
                session._log_file.flush()

    sys.stdout = _Tee()  # type: ignore[assignment]
