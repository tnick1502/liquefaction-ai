# Обученные модели

Ноутбуки папки `notebooks/2_model_training/` сохраняют каждую модель в собственный
каталог `models/<имя_модели>/` тремя файлами:

- `weights.pt` — веса модели (`state_dict`); в git не коммитится;
- `hyperparams.json` — тип модели, аргументы конструктора и гиперпараметры обучения;
- `history.parquet` — история обучения (кривые train/val по эпохам).

Текущие модели:

- `models/mlp_risk/` — статический MLP-классификатор риска (базовая модель);
- `models/gru/` — рекуррентная последовательностная модель (базовая);
- `models/tcn/` — каузальная свёрточная модель (базовая);
- `models/dpi_flow/` — DPI-Flow (вероятностный вывод параметров через ODE-слой);
- `models/evt_ssm/` — EVT-NeuralSSM (событийно-переключаемая модель состояний).

Загрузка модели для оценки выполняется в ноутбуках `notebooks/3_evaluations/` через
`liquefaction_ai.training.load_model_metadata` и `load_weights_into`: архитектура
восстанавливается по `hyperparams.json`, затем загружаются веса.

Синтетический датасет лежит в `data/demo_run/` (готовится ноутбуком
`1_data_preparation/1_1_data_generation.ipynb`). Содержимое `data/*` по умолчанию в git
не коммитится, остаётся `data/.gitkeep`.
