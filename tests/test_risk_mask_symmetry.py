"""
Тест симметрии risk-маски: незавершённые исходы исключаются из BCE у ВСЕХ моделей (не только
структурных), иначе leaderboard нечестен (метрики их исключают, а обучение baseline — нет).
"""
import re
from pathlib import Path

import torch

from liquefaction_ai.training.losses import masked_bce_with_logits

SRC = Path(__file__).resolve().parents[1] / "src" / "liquefaction_ai" / "models"


def test_masked_bce_excludes_unobserved():
    logit = torch.tensor([10.0, -10.0, 10.0])      # 2 уверенно-pos, 1 уверенно-neg
    label = torch.tensor([1.0, 0.0, 0.0])          # последний — «ложный негатив» (unfinished)
    observed = torch.tensor([1.0, 1.0, 0.0])       # последний не наблюдаем
    full = masked_bce_with_logits(logit, label, None)
    masked = masked_bce_with_logits(logit, label, observed)
    assert masked < full                            # исключение ложного негатива снижает лосс
    # маска == ручной BCE по наблюдаемым
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logit[:2], label[:2])
    assert torch.allclose(masked, ref, atol=1e-6)


def test_all_unobserved_returns_zero():
    logit = torch.tensor([1.0, 2.0]); label = torch.tensor([0.0, 0.0])
    assert float(masked_bce_with_logits(logit, label, torch.tensor([0.0, 0.0]))) == 0.0


def test_no_unmasked_bce_on_label_remains_in_models():
    # ни в одной модели не должно остаться BCE по полному label без маски наблюдаемости
    pat = re.compile(r'binary_cross_entropy_with_logits\([^)]*batch\["label"\]\s*\)')
    offenders = []
    for f in SRC.glob("*.py"):
        for m in pat.finditer(f.read_text(encoding="utf-8")):
            offenders.append(f"{f.name}: {m.group(0)[:60]}")
    assert not offenders, "немаскированный risk-BCE остался: " + "; ".join(offenders)
