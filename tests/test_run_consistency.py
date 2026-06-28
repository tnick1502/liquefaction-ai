"""
Тест сквозной согласованности артефакта.

Вызывает библиотечную проверку digitrock-консистентности (убранные утечные поля, Vs↔G0, plaxis по
грансоставу, Cu=D60/D10, отсутствие NaN в признаках, наличие измеренной CRR) и требует, чтобы ни
одна проверка не упала. Логика — в ``liquefaction_ai.data.consistency`` (раньше была в
корневом скрипте run_consistency.py; теперь её же использует ноутбук 3_8).
"""
import pytest

from conftest import REAL_OBJECTS

from liquefaction_ai.data.consistency import check_artifact_consistency

_skip = pytest.mark.skipif(not REAL_OBJECTS.exists(), reason="нет артефакта data/real_objects")


@_skip
def test_artifact_consistency_all_ok():
    ok, report = check_artifact_consistency(str(REAL_OBJECTS))
    assert "DONE" in report[-1], "проверка не дошла до конца"
    assert ok, "провалена проверка консистентности:\n" + "\n".join(report)
