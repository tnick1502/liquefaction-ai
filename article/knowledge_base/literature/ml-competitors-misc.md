---
tags: [literature, ml-competitor, related-work]
year: 2011-2022
role: related-work-cluster
---
# Кластер ML-конкурентов (Related Work 2.2)
Краткие записи; полные роли — в таблице исходных docx.

- **Samui & Sitharam (2011)** — ANN/SVM susceptibility. DOI 10.5194/nhess-11-1-2011. Ранний ML.
- **Demir & Sahin (2022)** — robust SVM/RF/XGBoost. DOI 10.1007/s12665-022-10578-4. Baseline-блок.
- **Jas & Dodagoudar (2023, XGBoost-SHAP)** — explainable ML. DOI 10.1016/j.soildyn.2022.107662.
- **Njock et al. (2020)** — ENN/t-SNE manifold. DOI 10.1016/j.soildyn.2019.105988. Latent/manifold уже применялись, но не как physical PPR-ODE forecast → подчёркивает новизну.
- **Ozsagir et al. (2022)** — fine-grained soils ML. DOI 10.1016/j.compgeo.2022.105014. Важна, если в данных есть fine-grained/mixed.
- **Zhang & Wang (2021)** — ensemble multi-dataset. DOI 10.1007/s00521-020-05084-2.
- **Zhao et al. (2022)** — XGBoost + Bayesian probabilistic. DOI 10.1016/j.compgeo.2022.104868. Сильный probabilistic ML competitor, но **без** dynamic PPR/N_liq continuation.

**Тезис кластера:** почти всё — static classification/triggering; никто не делает prefix-conditioned PPR continuation + censored N_liq + physics-structured flow.
**Связи:** [[sanger-2025-mechanics]] · [[maurer-sanger-2023]] · [[jas-dodagoudar-2023-review]]
