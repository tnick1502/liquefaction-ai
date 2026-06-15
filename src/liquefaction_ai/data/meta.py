"""
Локализация мета-данных популяции.

Утилиты добавляют к таблице метаданных русскоязычные колонки-подписи
(тип грунта, режим нагружения, семейство генератора), не изменяя исходные
англоязычные идентификаторы, по которым работает остальной код.
"""

from __future__ import annotations

from typing import Dict

import pandas as pd

from liquefaction_ai.constants import (
    GENERATOR_FAMILY_DISPLAY_NAMES,
    LOAD_DISPLAY_NAMES,
    RESPONSE_TYPE_DISPLAY_NAMES,
    SOIL_DISPLAY_NAMES,
)

__all__ = ["localize_series", "localize_meta_frame"]


def localize_series(values: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """
    Заменить значения категориальной серии русскоязычными подписями.

    Значения, отсутствующие в словаре соответствий, остаются без изменений.

    :param values: исходная серия с англоязычными идентификаторами
    :param mapping: словарь соответствий «идентификатор → русская подпись»
    :return: серия с локализованными значениями
    """
    return values.map(mapping).fillna(values)


def localize_meta_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Добавить к таблице метаданных русскоязычные колонки-подписи.

    Для колонок ``soil_type``, ``load_mode`` и ``generator_family`` создаются
    парные колонки с суффиксом ``_ru``. Исходные колонки сохраняются, чтобы
    не нарушить группировки и стратификацию по англоязычным ключам.

    :param df: таблица метаданных популяции
    :return: копия таблицы с дополнительными локализованными колонками
    """
    localized = df.copy()
    if "soil_type" in localized.columns:
        localized["soil_type_ru"] = localize_series(localized["soil_type"], SOIL_DISPLAY_NAMES)
    if "load_mode" in localized.columns:
        localized["load_mode_ru"] = localize_series(localized["load_mode"], LOAD_DISPLAY_NAMES)
    if "generator_family" in localized.columns:
        localized["generator_family_ru"] = localize_series(
            localized["generator_family"], GENERATOR_FAMILY_DISPLAY_NAMES
        )
    if "response_type" in localized.columns:
        localized["response_type_ru"] = localize_series(
            localized["response_type"], RESPONSE_TYPE_DISPLAY_NAMES
        )
    return localized
