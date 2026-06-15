"""
Эмпирические модели циклической сопротивляемости CRR (Cyclic Resistance Ratio).

Здесь собраны функции из литературы по потенциалу разжижения, связывающие число
циклов нагружения N и коэффициент циклических напряжений CSR на границе разжижения.
Представлены гиперболическое, степенное, экспоненциальное и логарифмическое семейства
в обеих формах: CSR(N) и, где это аналитически возможно, обратная N(CSR).

Все функции принимают как скаляр, так и массив, и возвращают результат
соответствующего типа.
"""

from typing import List, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, List[float], Tuple[float, ...], float, int]


def _to_numpy_1d(values: ArrayLike) -> np.ndarray:
    """
    Преобразование входных данных в одномерный numpy-массив типа float64.

    :param values: скаляр, список, кортеж или numpy-массив
    :return: одномерный массив numpy.float64
    """
    arr: np.ndarray = np.asarray(values, dtype=np.float64)

    if arr.ndim == 0:
        arr = arr.reshape(1)

    return arr


def _return_scalar_if_needed(
    original_values: ArrayLike,
    result: np.ndarray,
) -> Union[np.ndarray, float]:
    """
    Возврат скаляра, если исходный аргумент был скаляром.

    :param original_values: исходное значение, переданное пользователем
    :param result: вычисленный numpy-массив результата
    :return: float для скалярного ввода или numpy-массив для векторного
    """
    if np.isscalar(original_values):
        return float(result[0])

    return result


def _validate_positive_cycles(n_cycles: ArrayLike) -> np.ndarray:
    """
    Проверка массива числа циклов N.

    Для всех моделей из статьи аргумент N должен быть строго положительным,
    так как используются ln(N) и/или степенные преобразования.

    :param n_cycles: число циклов нагружения
    :return: одномерный массив числа циклов
    """
    n_arr: np.ndarray = _to_numpy_1d(n_cycles)

    if np.any(n_arr <= 0.0):
        raise ValueError("Все значения n_cycles должны быть > 0.")

    return n_arr


def bilge_exponential_n_from_csr(
    csr: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """
    Экспоненциальная функция H. Bilge et al. в форме N(CSR).

    Формула статьи:
        N = alpha * exp(beta * CSR)

    :param csr: значение CSR
    :param alpha: коэффициент alpha
    :param beta: коэффициент beta
    :return: значение числа циклов N
    """
    if alpha <= 0.0:
        raise ValueError("Параметр alpha должен быть > 0.")

    csr_arr: np.ndarray = _to_numpy_1d(csr)
    result: np.ndarray = alpha * np.exp(beta * csr_arr)

    return _return_scalar_if_needed(csr, result)


def bilge_exponential_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """
    Экспоненциальная функция H. Bilge et al. в форме CSR(N).

    Формула статьи:
        CSR = ln(N / alpha) / beta

    :param n_cycles: число циклов нагружения N
    :param alpha: коэффициент alpha
    :param beta: коэффициент beta
    :return: значение CSR
    """
    if alpha <= 0.0:
        raise ValueError("Параметр alpha должен быть > 0.")
    if beta == 0.0:
        raise ValueError("Параметр beta не должен быть равен 0.")

    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = np.log(n_arr / alpha) / beta

    return _return_scalar_if_needed(n_cycles, result)


def lentini_logarithmic_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """
    Логарифмическая функция V. Lentini et al. в форме CSR(N).

    Формула статьи:
        CSR = alpha * log(N) + beta

    Здесь используется натуральный логарифм numpy.log.

    :param n_cycles: число циклов нагружения N
    :param alpha: коэффициент alpha
    :param beta: коэффициент beta
    :return: значение CSR
    """
    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = alpha * np.log(n_arr) + beta

    return _return_scalar_if_needed(n_cycles, result)


def guoxing_power_n_from_csr(
    csr: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """
    Степенная функция C. Guoxing et al. в форме N(CSR).

    Формула статьи:
        N = alpha * CSR^beta

    :param csr: значение CSR
    :param alpha: коэффициент alpha
    :param beta: коэффициент beta
    :return: значение числа циклов N
    """
    if alpha <= 0.0:
        raise ValueError("Параметр alpha должен быть > 0.")

    csr_arr: np.ndarray = _to_numpy_1d(csr)

    if np.any(csr_arr < 0.0):
        raise ValueError("Для степенной модели значения CSR должны быть >= 0.")

    result: np.ndarray = alpha * np.power(csr_arr, beta)

    return _return_scalar_if_needed(csr, result)


def guoxing_power_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """
    Степенная функция C. Guoxing et al. в форме CSR(N).

    Формула статьи:
        CSR = (N / alpha)^(1 / beta)

    :param n_cycles: число циклов нагружения N
    :param alpha: коэффициент alpha
    :param beta: коэффициент beta
    :return: значение CSR
    """
    if alpha <= 0.0:
        raise ValueError("Параметр alpha должен быть > 0.")
    if beta == 0.0:
        raise ValueError("Параметр beta не должен быть равен 0.")

    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = np.power(n_arr / alpha, 1.0 / beta)

    return _return_scalar_if_needed(n_cycles, result)


def meziane_logarithmic_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """
    Логарифмическая функция E. Meziane et al. в форме CSR(N).

    Формула статьи:
        CSR = (alpha - ln(N)) / beta

    :param n_cycles: число циклов нагружения N
    :param alpha: коэффициент alpha
    :param beta: коэффициент beta
    :return: значение CSR
    """
    if beta == 0.0:
        raise ValueError("Параметр beta не должен быть равен 0.")

    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = (alpha - np.log(n_arr)) / beta

    return _return_scalar_if_needed(n_cycles, result)


def author_hyperbolic_csr(
    n_cycles: ArrayLike,
    alpha: float,
    beta: float,
) -> Union[np.ndarray, float]:
    """
    Гиперболическая функция авторов исследования в форме CSR(N).

    Формула статьи:
        CSR = beta / N^(1 - alpha)

    Эквивалентно:
        CSR = beta * N^(alpha - 1)

    :param n_cycles: число циклов нагружения N
    :param alpha: коэффициент alpha
    :param beta: коэффициент beta
    :return: значение CSR
    """
    n_arr: np.ndarray = _validate_positive_cycles(n_cycles)
    result: np.ndarray = beta / np.power(n_arr, 1.0 - alpha)

    return _return_scalar_if_needed(n_cycles, result)
