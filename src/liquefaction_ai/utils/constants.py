SOIL_CLASS_SPECS = {
    "loose_sand": {
        "e": (0.78, 1.05),
        "D_r": (0.25, 0.55),
        "I_p": (0.0, 3.0),
        "V_s": (110.0, 180.0),
        "xi": (0.02, 0.05),
        "sigma_eff": (70.0, 180.0),
        "log_perm": (-4.4, -3.2),
    },
    "silty_sand": {
        "e": (0.70, 0.98),
        "D_r": (0.35, 0.65),
        "I_p": (1.0, 6.0),
        "V_s": (125.0, 210.0),
        "xi": (0.025, 0.06),
        "sigma_eff": (80.0, 210.0),
        "log_perm": (-5.2, -3.8),
    },
    "low_plastic_silt": {
        "e": (0.62, 0.92),
        "D_r": (0.30, 0.60),
        "I_p": (4.0, 10.0),
        "V_s": (135.0, 225.0),
        "xi": (0.03, 0.07),
        "sigma_eff": (90.0, 240.0),
        "log_perm": (-6.0, -4.4),
    },
    "clayey_silt": {
        "e": (0.58, 0.88),
        "D_r": (0.28, 0.52),
        "I_p": (8.0, 20.0),
        "V_s": (145.0, 250.0),
        "xi": (0.04, 0.10),
        "sigma_eff": (95.0, 260.0),
        "log_perm": (-6.8, -5.2),
    },
    "dense_sand": {
        "e": (0.42, 0.72),
        "D_r": (0.65, 0.95),
        "I_p": (0.0, 2.0),
        "V_s": (210.0, 360.0),
        "xi": (0.015, 0.04),
        "sigma_eff": (120.0, 320.0),
        "log_perm": (-4.0, -2.8),
    },
}

LOAD_MODE_SPECS = {
    "storm": {
        "CSR": (0.10, 0.24),
        "frequency": (0.05, 0.35),
        "amp_scale": (0.85, 1.25),
        "N_max": (700.0, 1_500.0),
        "nonstationarity": (0.20, 0.60),
    },
    "seismic": {
        "CSR": (0.18, 0.48),
        "frequency": (0.8, 3.5),
        "amp_scale": (0.90, 1.35),
        "N_max": (60.0, 500.0),
        "nonstationarity": (0.45, 0.95),
    },
    "technogenic": {
        "CSR": (0.06, 0.18),
        "frequency": (6.0, 20.0),
        "amp_scale": (0.90, 1.10),
        "N_max": (250.0, 1_200.0),
        "nonstationarity": (0.08, 0.25),
    },
    "stationary_cyclic": {
        "CSR": (0.08, 0.22),
        "frequency": (0.8, 5.0),
        "amp_scale": (0.95, 1.10),
        "N_max": (350.0, 1_500.0),
        "nonstationarity": (0.05, 0.18),
    },
    "variable_amplitude": {
        "CSR": (0.10, 0.30),
        "frequency": (0.2, 10.0),
        "amp_scale": (0.85, 1.35),
        "N_max": (180.0, 1_500.0),
        "nonstationarity": (0.35, 0.85),
    },
}

SOIL_NAMES = list(SOIL_CLASS_SPECS.keys())
LOAD_NAMES = list(LOAD_MODE_SPECS.keys())

SOIL_DISPLAY_NAMES = {
    "loose_sand": "рыхлый песок",
    "silty_sand": "пылеватый песок",
    "low_plastic_silt": "малопластичный ил",
    "clayey_silt": "глинистый ил",
    "dense_sand": "плотный песок",
}

LOAD_DISPLAY_NAMES = {
    "storm": "штормовой",
    "seismic": "сейсмический",
    "technogenic": "техногенный",
    "stationary_cyclic": "стационарно-циклический",
    "variable_amplitude": "переменная амплитуда",
}

GENERATOR_FAMILY_DISPLAY_NAMES = {
    "hyperbolic": "гиперболическое",
    "power": "степенное",
    "exponential": "экспоненциальное",
    "logarithmic": "логарифмическое",
}
