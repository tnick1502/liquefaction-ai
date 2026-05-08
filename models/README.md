# Сохранённые веса

Чекпоинты обучения складывайте сюда. Файлы `*.pt` / `*.pth` в git не коммитятся.

Рекомендуемая структура прогона:

- `models/<run_id>/mlp_risk.pt`, `gru.pt`, `tcn.pt`, `dpi_flow.pt`, `evt_ssm.pt`
- опционально `models/<run_id>/config.json` (копия или дамп `ExperimentConfig`)

Синтетические датасеты в корне репозитория: **`data/<run_id>/`** (например `data/demo_run/` из ноутбука 01). Содержимое `data/*` по умолчанию в git не коммитится, остаётся `data/.gitkeep`.
