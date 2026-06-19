"""Общая конфигурация pytest: путь к ``src`` и пути к артефактам/таблицам проекта."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REAL_OBJECTS = REPO_ROOT / "data" / "real_objects"
DEMO_RUN = REPO_ROOT / "data" / "demo_run"
TABLES = REPO_ROOT / "results" / "tables"
