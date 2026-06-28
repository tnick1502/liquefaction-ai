---
tags: [formulation, abstract, draft]
status: draft-v0
---
# Черновик abstract (v0)

> Soil liquefaction is a dynamic failure process that engineering practice evaluates with CSR/CRR triggering procedures, which return a boundary or probability rather than a forecast of the pore-pressure response and the time to onset. We cast laboratory liquefaction assessment as **prefix-conditioned probabilistic event-time forecasting**: from static soil/loading descriptors and an early excess pore-pressure (PPR) prefix, predict the PPR continuation, liquefaction risk, and the (possibly right-censored) cycle count to onset N_liq. We introduce **DPI-Flow**, a physics-structured architecture that amortizes inference of physically constrained parameters through a **conditional affine flow** and rolls them through a **differentiable analytical CRR/damage/PPR layer**, guaranteeing feasible monotone accumulation. A censoring-aware objective trains jointly on liquefied (exact), stabilized (right-censored), and unfinished tests. On **1093 real cyclic tests across 20 sites**, under **object-held-out** evaluation with multi-seed confidence intervals, DPI-Flow is the **best physically admissible, onset-aware probabilistic forecaster**, with zero physics violations and calibrated uncertainty, while remaining competitive with strong tabular, sequence, and physics-informed baselines.

Заметки: числа AUROC не выносить в abstract до проверки prefix-leakage ([[../protocol/evaluation-protocol]]). «Calibrated» — мягко, как результат.
