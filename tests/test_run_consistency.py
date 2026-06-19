"""
Тест сквозной согласованности артефакта (обёртка над ``run_consistency.py``).

Запускает скрипт проверки digitrock-консистентности (убранные утечные поля, Vs↔G0, plaxis по
грансоставу, Cu=D60/D10, отсутствие NaN в признаках, наличие измеренной CRR) и требует, чтобы
ни одна проверка не упала.
"""
import subprocess
import sys

import pytest

from conftest import REAL_OBJECTS, REPO_ROOT

_skip = pytest.mark.skipif(not REAL_OBJECTS.exists(), reason="нет артефакта data/real_objects")


@_skip
def test_run_consistency_all_ok():
    proc = subprocess.run([sys.executable, "run_consistency.py"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=300)
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"run_consistency завершился с ошибкой:\n{out[-800:]}"
    assert "DONE" in out, "run_consistency не дошёл до конца"
    assert "FAIL" not in out, f"провалена проверка консистентности:\n{out}"
