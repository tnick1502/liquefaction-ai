from typing import Dict, Optional, Tuple, Union
import numpy as np


ArrayLike = Union[np.ndarray, list, tuple, float, int]


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


def _return_scalar_if_needed(original_values: ArrayLike, result: np.ndarray) -> Union[np.ndarray, float]:
    """
    Возврат скаляра, если исходный аргумент был скаляром.

    :param original_values: исходное значение, переданное пользователем
    :param result: вычисленный numpy-массив результата
    :return: float для скалярного ввода или numpy-массив для векторного
    """
    if np.isscalar(original_values):
        return float(result[0])

    return result


def compute_ppr(
    u: ArrayLike,
    sigma_3: float,
) -> Union[np.ndarray, float]:
    """
    Расчёт приведённого порового давления PPR.

    PPR(N) = u / sigma_3

    :param u: поровое давление
    :param sigma_3: обжимающее напряжение
    :return: значение PPR
    """
    if sigma_3 == 0:
        raise ValueError("Параметр sigma_3 не должен быть равен 0.")

    u_arr: np.ndarray = _to_numpy_1d(u)
    ppr: np.ndarray = u_arr / float(sigma_3)

    return _return_scalar_if_needed(u, ppr)


def cpt_pore_pressure_ma_wang(
    n_cycles: ArrayLike,
    a: float,
    k: float,
    A: float,
) -> Union[np.ndarray, float]:
    """
    Расчёт порового давления по CPT-модели Ма и Ванга.

    Реализуется составная модель:
        u(N) = a * A * (1 - exp(-k * N / A))
               + (1 - a) * N / (1 / k + N / A)

    где:
    - первый член отвечает за экспоненциальный рост;
    - второй член отвечает за гиперболическое насыщение.

    :param n_cycles: число циклов нагружения N
    :param a: весовой коэффициент экспоненциальной части, обычно 0 <= a <= 1
    :param k: коэффициент, характеризующий динамику начального роста
    :param A: параметр максимального уровня функции
    :return: массив значений порового давления u(N)
    """
    if A <= 0:
        raise ValueError("Параметр A должен быть > 0.")
    if k <= 0:
        raise ValueError("Параметр k должен быть > 0.")

    n: np.ndarray = _to_numpy_1d(n_cycles)

    exp_term: np.ndarray = a * A * (1.0 - np.exp(-k * n / A))
    hyp_term: np.ndarray = (1.0 - a) * n / ((1.0 / k) + (n / A))

    u: np.ndarray = exp_term + hyp_term
    return _return_scalar_if_needed(n_cycles, u)


def cpt_ppr_ma_wang(
    n_cycles: ArrayLike,
    a: float,
    k: float,
    A: float,
    sigma_3: Optional[float] = None,
) -> Union[np.ndarray, float]:
    """
    Расчёт PPR по CPT-модели Ма и Ванга.

    Если sigma_3 не задано, считается, что модель уже возвращает нормированную величину.
    Если sigma_3 задано, сначала считается u(N), затем PPR = u / sigma_3.

    :param n_cycles: число циклов нагружения N
    :param a: весовой коэффициент экспоненциальной части
    :param k: коэффициент динамики роста
    :param A: параметр максимального уровня функции
    :param sigma_3: обжимающее напряжение, если нужно перейти от u к PPR
    :return: массив значений PPR(N)
    """
    u: Union[np.ndarray, float] = cpt_pore_pressure_ma_wang(
        n_cycles=n_cycles,
        a=a,
        k=k,
        A=A,
    )

    if sigma_3 is None:
        return u

    return compute_ppr(u=u, sigma_3=sigma_3)


def extended_cpt_ppr(
    n_cycles: ArrayLike,
    a: float,
    b: float,
    k: float,
    A: float,
    n_max: float,
) -> Union[np.ndarray, float]:
    """
    Расчёт PPR по расширенной CPT-модели.

    Формула:
        f1(N) = a * A * (1 - exp(-k * N / A))
                + b * N / (1 / k + N / A)
                + (1 - a - b) * A * ln(k * N + 1) / ln(k * N_max + 1)

    Ограничения модели из статьи:
        a + b <= 1
        a, b ∈ [0, 1]

    :param n_cycles: число циклов нагружения N
    :param a: коэффициент влияния экспоненциального члена
    :param b: коэффициент влияния гиперболического члена
    :param k: коэффициент динамики роста PPR
    :param A: коэффициент, регулирующий максимальный уровень функции
    :param n_max: максимальное число циклов в рассматриваемом диапазоне, используется для нормировки логарифмического члена
    :return: массив значений PPR(N)
    """
    if A <= 0:
        raise ValueError("Параметр A должен быть > 0.")
    if k <= 0:
        raise ValueError("Параметр k должен быть > 0.")
    if n_max <= 0:
        raise ValueError("Параметр n_max должен быть > 0.")
    if not (0.0 <= a <= 1.0):
        raise ValueError("Параметр a должен лежать в диапазоне [0, 1].")
    if not (0.0 <= b <= 1.0):
        raise ValueError("Параметр b должен лежать в диапазоне [0, 1].")
    if a + b > 1.0:
        raise ValueError("Должно выполняться условие a + b <= 1.")

    n: np.ndarray = _to_numpy_1d(n_cycles)

    exp_term: np.ndarray = a * A * (1.0 - np.exp(-k * n / A))
    hyp_term: np.ndarray = b * n / ((1.0 / k) + (n / A))

    log_denominator: float = np.log(k * n_max + 1.0)
    if np.isclose(log_denominator, 0.0):
        raise ValueError("Знаменатель ln(k * n_max + 1) не должен быть равен 0.")

    log_term: np.ndarray = (1.0 - a - b) * A * np.log(k * n + 1.0) / log_denominator

    f1: np.ndarray = exp_term + hyp_term + log_term
    return _return_scalar_if_needed(n_cycles, f1)


def logarithmic_ppr_raw(
    n_cycles: ArrayLike,
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> Union[np.ndarray, float]:
    """
    Расчёт ненормированной логарифмической функции роста/диссипации PPR.

    Формула:
        g(N) = A1 * ln(k1 * N + 1) - A2 * ln(k2 * N + 1)

    Эта функция задаёт форму кривой до масштабирования коэффициентом C.

    :param n_cycles: число циклов нагружения N
    :param A1: коэффициент роста логарифмического члена
    :param A2: коэффициент члена, отвечающего за снижение / диссипацию
    :param k1: коэффициент динамики первого логарифмического члена
    :param k2: коэффициент динамики второго логарифмического члена
    :return: массив значений g(N)
    """
    if k1 <= 0:
        raise ValueError("Параметр k1 должен быть > 0.")
    if k2 <= 0:
        raise ValueError("Параметр k2 должен быть > 0.")

    n: np.ndarray = _to_numpy_1d(n_cycles)
    g: np.ndarray = A1 * np.log(k1 * n + 1.0) - A2 * np.log(k2 * n + 1.0)

    return _return_scalar_if_needed(n_cycles, g)


def logarithmic_model_peak_cycle(
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> float:
    """
    Расчёт числа циклов N_peak, в котором ненормированная логарифмическая функция достигает экстремума.

    Из условия:
        d/dN [A1 * ln(k1 * N + 1) - A2 * ln(k2 * N + 1)] = 0

    Получаем:
        N_peak = (A2 * k2 - A1 * k1) / (k1 * k2 * (A1 - A2))

    :param A1: коэффициент роста первого логарифмического члена
    :param A2: коэффициент второго логарифмического члена
    :param k1: коэффициент динамики первого логарифмического члена
    :param k2: коэффициент динамики второго логарифмического члена
    :return: число циклов N_peak
    """
    denominator: float = k1 * k2 * (A1 - A2)

    if np.isclose(denominator, 0.0):
        raise ValueError("Невозможно вычислить N_peak: знаменатель равен 0.")

    n_peak: float = (A2 * k2 - A1 * k1) / denominator
    return float(n_peak)


def logarithmic_model_scale_factor(
    A: float,
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> float:
    """
    Расчёт коэффициента масштабирования C для нормированной логарифмической модели.

    Логика:
    1. Строится базовая функция:
           g(N) = A1 * ln(k1 * N + 1) - A2 * ln(k2 * N + 1)
    2. Находится N_peak — точка экстремума g(N)
    3. Масштабирование выполняется так, чтобы максимум (или экстремум) имел уровень A:
           C = A / g(N_peak)

    Это алгебраически эквивалентно записи из статьи, но реализовано в более устойчивом вычислительном виде.

    :param A: целевой уровень масштабирования
    :param A1: коэффициент роста первого логарифмического члена
    :param A2: коэффициент второго логарифмического члена
    :param k1: коэффициент динамики первого логарифмического члена
    :param k2: коэффициент динамики второго логарифмического члена
    :return: коэффициент масштабирования C
    """
    n_peak: float = logarithmic_model_peak_cycle(
        A1=A1,
        A2=A2,
        k1=k1,
        k2=k2,
    )

    if n_peak <= -1.0 / max(k1, k2):
        raise ValueError("Получено некорректное N_peak: логарифм выходит за область определения.")

    g_peak: float = float(
        logarithmic_ppr_raw(
            n_cycles=n_peak,
            A1=A1,
            A2=A2,
            k1=k1,
            k2=k2,
        )
    )

    if np.isclose(g_peak, 0.0):
        raise ValueError("Невозможно вычислить C: значение базовой функции в точке экстремума равно 0.")

    C: float = A / g_peak
    return float(C)


def logarithmic_ppr_normalized(
    n_cycles: ArrayLike,
    A: float,
    A1: float,
    A2: float,
    k1: float,
    k2: float,
) -> Union[np.ndarray, float]:
    """
    Расчёт PPR по нормированной логарифмической модели.

    Формула:
        f2(N) = [A1 * ln(k1 * N + 1) - A2 * ln(k2 * N + 1)] * C

    где C подбирается так, чтобы экстремум функции достигал уровня A.

    :param n_cycles: число циклов нагружения N
    :param A: целевой уровень масштабирования функции
    :param A1: коэффициент роста первого логарифмического члена
    :param A2: коэффициент второго логарифмического члена
    :param k1: коэффициент динамики первого логарифмического члена
    :param k2: коэффициент динамики второго логарифмического члена
    :return: массив значений PPR(N)
    """
    C: float = logarithmic_model_scale_factor(
        A=A,
        A1=A1,
        A2=A2,
        k1=k1,
        k2=k2,
    )

    raw: Union[np.ndarray, float] = logarithmic_ppr_raw(
        n_cycles=n_cycles,
        A1=A1,
        A2=A2,
        k1=k1,
        k2=k2,
    )

    raw_arr: np.ndarray = _to_numpy_1d(raw)
    f2: np.ndarray = raw_arr * C

    return _return_scalar_if_needed(n_cycles, f2)