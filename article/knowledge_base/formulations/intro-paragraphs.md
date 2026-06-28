---
tags: [formulation, intro, draft]
---
# Готовые абзацы введения

Связано: [[../protocol/claim-to-citation]] · [[thesis-and-contributions]]

> Рамка значимости и ценности архива: [[significance-and-data-value]] (вплести в §1 stakes и §2→§3 scarcity).

**§1 — Problem & impact.** Soil liquefaction is a high-impact dynamic failure process traditionally evaluated through CSR/CRR triggering procedures. These procedures are well established, but they compress a cyclic process into a triggering boundary or probability, rather than forecasting the full pore-pressure trajectory and the event time under partial early observations. → [[../literature/seed-idriss-1971]] [[../literature/youd-2001]] [[../literature/nasem-2021]]

**§2 — Gap in AI liquefaction.** ML models for liquefaction have grown rapidly, but prior work is criticized for weak comparison to state-of-practice, insufficient validation, limited reproducibility, and a tendency toward static classification. This creates an opportunity for AI methods that are physically structured, interpretable, and evaluated under site-level distribution shift. → [[../literature/maurer-sanger-2023]] [[../literature/jas-dodagoudar-2023-review]]

**§3 — AI methodological framing.** We formulate laboratory liquefaction assessment as prefix-conditioned probabilistic event-time forecasting: given static soil/loading descriptors and an early PPR prefix, predict the continuation of PPR(N), the probability of liquefaction, and the censored time-to-event N_liq. This connects scientific ML, latent-variable inference, neural ODEs, normalizing flows, uncertainty calibration, and survival modeling. → [[../literature/wang-2024-aaai]] [[../literature/karniadakis-2021]] [[../literature/chen-2018-neuralode]] [[../literature/rezende-mohamed-2015]]

**§4 — Contribution.** We introduce DPI-Flow, a physics-structured probabilistic architecture that amortizes inference of physically constrained ODE parameters, refines them through prefix calibration, and integrates an analytical differentiable CRR/damage/PPR layer. Evaluated against strong tabular and sequence baselines under grouped site-held-out and OOD stress protocols, with calibration and censoring-aware metrics. → [[thesis-and-contributions]]
