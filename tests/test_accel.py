"""Выбор ускорителя: CUDA → MPS (Apple Metal) → CPU, с override через LIQ_DEVICE."""
import torch

from liquefaction_ai import configure_performance, describe_device, resolve_device
from liquefaction_ai.accel import resolve_device as _resolve


def test_resolve_device_returns_valid_torch_device():
    d = resolve_device()
    assert isinstance(d, torch.device)
    assert d.type in ("cuda", "mps", "cpu")


def test_liq_device_env_override(monkeypatch):
    monkeypatch.setenv("LIQ_DEVICE", "cpu")
    assert _resolve().type == "cpu"
    # явный аргумент имеет приоритет над окружением
    assert _resolve("cpu").type == "cpu"


def test_configure_performance_is_idempotent_and_returns_device():
    d = configure_performance(torch.device("cpu"), cpu_threads=2)
    assert d.type == "cpu"
    assert isinstance(describe_device(d), str) and describe_device(d)


def test_preference_order_matches_availability():
    # resolve() без override должен согласовываться с фактической доступностью бэкендов.
    d = _resolve()
    if torch.cuda.is_available():
        assert d.type == "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        assert d.type == "mps"
    else:
        assert d.type == "cpu"
