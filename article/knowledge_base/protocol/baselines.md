---
tags: [protocol, baselines, P1]
---
# Baselines (чтобы рецензент не снёс)

Связано: [[../literature/maurer-sanger-2023]] (требует сравнения с practice).

| Группа | Модели | Заметка |
|---|---|---|
| State-of-practice geotech | Seed/Idriss, Boulanger/Idriss, Cetin/Moss/Kayen triggering | [[../literature/boulanger-idriss-2016]] · ответ на критику Maurer&Sanger |
| GBDT/tabular SOTA | **CatBoost** (risk + exact-event N_liq; censored metric N/A), XGBoost/LightGBM, RF | [[../literature/prokhorenkova-2018-catboost]] |
| Deep tabular | MLP/ResNet, **TabNet**, **FT-Transformer** (есть) | [[../literature/gorishniy-2021-fttransformer]] · [[../literature/arik-pfister-2021-tabnet]] |
| Sequence | **GRU/LSTM/TCN/Transformer** (есть) | задача prefix-conditioned continuation |
| Survival/event-time | **Cox/DeepSurv**/discrete-time survival | [[../literature/katzman-2018-deepsurv]] · для censored N_liq |
| Published liquefaction-ML SOTA | **Sanger/Geyin/Maurer 2025** mechanics-informed | [[../literature/sanger-2025-mechanics]] · ОБЯЗАТЕЛЬНО, иначе «все бейзлайны самодельные» |
| Physics baseline | PINN (есть) | [[../literature/raissi-2019-pinn]] |

**Уже реализовано в репо:** CatBoost, FT-Transformer, GRU/LSTM/TCN/Transformer, PINN, NSF/RealNVP, DeepState, EVT-NeuralSSM, DPI-EVT, RiskMLP.
**Gap:** classical triggering curves + survival baseline (Cox/DeepSurv) + явная ссылка на Sanger 2025.
