from typing import Any, List, Tuple
from uuid import uuid4

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

import pdcast.types as tc
from pdcast import options

RTOL = options.RTOL
ATOL = options.ATOL

rng = np.random.default_rng()


def boolean_mocks(length) -> List[Tuple[Series, Any]]:
    return [  # type: ignore
        # Null
        (pd.Series([None] * length), pd.Int8Dtype()),
        # Boolean
        # (pd.Series(np.zeros(length)), np.bool_),
        # (pd.Series(np.ones(length)), np.bool_),
        (pd.Series(rng.integers(0, 2, length)), np.bool_),
        (pd.Series(np.random.choice([1 + 1e-12, 1e-12], length)), np.bool_),
        # Boolean - Nullable
        (pd.Series(np.random.choice([1, 0, None], length)).astype(float), pd.Int8Dtype()),  # type: ignore
        (pd.Series(np.random.choice([1 + 1e-12, 1e-12, None], length)).astype(float), pd.Int8Dtype()),  # type: ignore
    ]


def integer_mocks(length) -> List[Tuple[Series, Any]]:
    unsigned_ints = [
        rng.integers(0, 255, length),
        rng.integers(0, 65_000, length),
        rng.integers(0, 4_294_967_295, length),
        rng.integers(0, 18_446_744_073_709, length),
    ]
    signed_ints = [
        rng.integers(-128, 127, length),
        rng.integers(-32_768, 32_767, length),
        rng.integers(-2_147_483_648, 2_147_483_647, length),
        np.array(rng.integers(-2_147_483_648, 2_147_483_647, length)) * 1e9,
    ]

    return [
        # Integer - Unsigned
        *[
            (pd.Series(x, dtype=pd.Int64Dtype()), y)
            for x, y in zip(unsigned_ints, tc.UINT_TYPES)
        ],
        # Integer - Unsigned - Nullable
        *[
            (pd.Series(np.where(rng.uniform(size=length) > 0.5, x, None), dtype=pd.Int64Dtype()), y)
            for x, y in zip(unsigned_ints, tc.UINT_NULLABLE_TYPES)
        ],
        # Integer - Signed
        *[
            (pd.Series(x, dtype=pd.Int64Dtype()), y)
            for x, y in zip(signed_ints, tc.INT_TYPES)
        ],
        # Integer - Signed - Nullable
        *[
            (pd.Series(np.where(rng.uniform(size=length) > 0.5, x, None), dtype=pd.Int64Dtype()), y)
            for x, y in zip(signed_ints, tc.INT_NULLABLE_TYPES)
        ],
    ]


def float_mocks(length) -> List[Tuple[Series, Any]]:
    floats = [
        np.random.uniform(-65503.9, 65499.9, length).astype(np.float16).astype(np.float64),
        np.random.uniform(-3.4028235e38, 3.4028235e38, length).astype(np.float32).astype(np.float64),
        np.random.uniform(-1.7976931348623157e200, 1.7976931348623157e200, length),
    ]
    return [
        # Float
        *[(pd.Series(x), y) for x, y in zip(floats, tc.FLOAT_TYPES)]
    ]


# Categorical
def categorical_mocks(length) -> List[Tuple[Series, Any]]:
    unique_ids = [uuid4() for _ in range(length)]
    unique_nullable_ids = np.where(rng.uniform(size=length) > 0.5, unique_ids, None)
    nearly_unique_1 = np.where(rng.uniform(size=length) > 0.9, "a", unique_ids)
    nearly_unique_2 = np.where(rng.uniform(size=length) > 0.7, "a", unique_ids)
    return [
        # Repeated
        (pd.Series(np.random.choice(["a", "b", "c", "d"], length)), pd.CategoricalDtype()),
        # Repeated - Nullable
        (pd.Series(np.random.choice(["a", "b", "c", None], length)), pd.CategoricalDtype()),
        # Unique
        (pd.Series(unique_ids), np.object_),
        # Unique - Nullable
        (pd.Series(unique_nullable_ids), np.object_),
        # Nearly Unique - Above 80% Threshold
        (pd.Series(nearly_unique_1), np.object_),
        # Nearly Unique - Below 80% Threshold
        (pd.Series(nearly_unique_2), pd.CategoricalDtype()),
    ]


def other_mocks(length):
    return [
        (pd.Series([np.complex_(x) for x in range(length)]), np.complex_)
    ]


def series_mocks(length) -> List[Tuple[Series, Any]]:
    mocks: List[Tuple[Series, Any]] = [
        *boolean_mocks(length),
        *integer_mocks(length),
        *float_mocks(length),
        *categorical_mocks(length),
        *other_mocks(length),
    ]
    return mocks


def dataframe_mock(length) -> Tuple[DataFrame, dict]:
    mocks = [
        boolean_mocks(length),
        integer_mocks(length),
        float_mocks(length),
        categorical_mocks(length),
        other_mocks(length),
    ]
    input_series = [y[0] for x in mocks for y in x]
    expected_types = [y[1] for x in mocks for y in x]
    df = pd.concat(input_series, axis=1)
    schema = {k: v for k, v in zip(df.columns, expected_types)}
    return df, schema  # type: ignore


series_date_types = [
    (pd.Series(pd.period_range("2021-01-01", "2021-12-31"))),
    (pd.Series(pd.date_range("2021-01-01", "2021-12-31"))),
    (pd.Series(pd.timedelta_range("1 Day", periods=365))),
]


def series_numpy_only(length):
    return [
        # Integer - Unsigned - Nullable
        (pd.Series(np.where(rng.uniform(size=length) > 0.5, rng.integers(0, 255, length), None), dtype=float), 
        np.float16),
        # Categorical = Repeated
        (pd.Series(np.random.choice(["a", "b", "c", "d"], length)), np.object_),
    ]


def frames_equal(
    a: DataFrame, b: DataFrame, rtol: float = None, atol: float = None
) -> bool:
    rtol = rtol or RTOL
    atol = atol or ATOL
    for i in range(a.shape[1]):
        a_col = a.iloc[:, i]
        b_col = b.iloc[:, i]
        if not series_equal(a_col, b_col, rtol=rtol, atol=atol):
            return False
    return True


def series_equal(
    a: Series, b: Series, rtol: float = None, atol: float = None
) -> bool:
    rtol = rtol or RTOL
    atol = atol or ATOL
    cond1 = a.name == b.name
    cond2 = a.shape == b.shape
    if not (cond1 and cond2):
        return False
    try:
        if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
            return False
    except TypeError:
        if not a.dropna().tolist() == b.dropna().tolist():
            return False
    return True


def dicts_equal(d1: dict, d2: dict):
    if d1.keys() != d2.keys():
        return False
    for k1, v1 in d1.items():
        v2 = d2[k1]
        if v1 != v2 and type(v1) != type(v2):
            print(f"v1: {repr(v1)} v2: {repr(v2)}")
            return False
    return True
