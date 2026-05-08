from __future__ import annotations

from typing import Dict

import pandas as pd

from liquefaction_ai.utils.constants import (
    GENERATOR_FAMILY_DISPLAY_NAMES,
    LOAD_DISPLAY_NAMES,
    SOIL_DISPLAY_NAMES,
)


def localize_series(values: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    return values.map(mapping).fillna(values)


def localize_meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    localized = df.copy()
    if "soil_type" in localized.columns:
        localized["soil_type_ru"] = localize_series(localized["soil_type"], SOIL_DISPLAY_NAMES)
    if "load_mode" in localized.columns:
        localized["load_mode_ru"] = localize_series(localized["load_mode"], LOAD_DISPLAY_NAMES)
    if "generator_family" in localized.columns:
        localized["generator_family_ru"] = localize_series(
            localized["generator_family"], GENERATOR_FAMILY_DISPLAY_NAMES
        )
    return localized
