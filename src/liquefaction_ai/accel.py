"""Выбор ускорителя, единый для всех ноутбуков и скриптов.

Порядок предпочтения: **CUDA** (NVIDIA) → **MPS** (Apple Metal, Mac) → **CPU**. Переопределяется
переменной окружения ``LIQ_DEVICE`` (например ``LIQ_DEVICE=cpu`` для строгой воспроизводимости или
``LIQ_DEVICE=mps``). Для MPS автоматически включается прозрачный CPU-fallback (некоторые физические
операции — ``cumprod``/сложная индексация — ещё не реализованы в Metal-бэкенде PyTorch).
"""
from __future__ import annotations

import os

import torch


def resolve_device(requested: str | None = None) -> torch.device:
    """
    Вернуть лучший доступный torch-device: CUDA → MPS (Apple) → CPU.

    :param requested: явное имя устройства ('cuda'/'mps'/'cpu'/'cuda:1'); если None — берётся из
        ``LIQ_DEVICE`` либо автоопределяется.
    :return: ``torch.device``. Для MPS выставляет ``PYTORCH_ENABLE_MPS_FALLBACK=1`` (иначе неподдержанные
        Metal-ом операции упали бы вместо отката на CPU).
    """
    req = requested or os.environ.get("LIQ_DEVICE")
    if req:
        device = torch.device(req)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return device


def describe_device(device: torch.device | None = None) -> str:
    """Человекочитаемое описание устройства (имя GPU для CUDA)."""
    device = device or resolve_device()
    if device.type == "cuda":
        try:
            return f"cuda ({torch.cuda.get_device_name(device)})"
        except Exception:
            return "cuda"
    if device.type == "mps":
        return "mps (Apple Metal)"
    return "cpu"


def configure_performance(device: torch.device | None = None, cpu_threads: int | None = None) -> torch.device:
    """
    Настроить бэкенд под устройство и вернуть его.

    - CUDA: включает TF32 (matmul/cudnn) и ``float32_matmul_precision='high'`` — заметное ускорение на
      Ampere+ без значимой потери точности для наших сетей.
    - MPS: гарантирует CPU-fallback (см. :func:`resolve_device`).
    - CPU: при ``cpu_threads`` фиксирует число потоков (полезно для детерминизма/пропускной способности).

    :param device: устройство (если None — :func:`resolve_device`).
    :param cpu_threads: явное число CPU-потоков (только для CPU-устройства).
    :return: то же устройство.
    """
    device = device or resolve_device()
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    elif device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    elif device.type == "cpu" and cpu_threads:
        torch.set_num_threads(int(cpu_threads))
    return device
