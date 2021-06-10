from typing import Tuple
from uuid import uuid4

import numpy as np
import pandas as pd

import pdcast.core as core

rng = np.random.default_rng()


def boolean_mocks(length):
    return [
        # Null
        (pd.Series([None] * length), pd.Int8Dtype()),

        # Boolean
        (pd.Series(np.zeros(length)), np.bool_),
        (pd.Series(np.ones(length)), np.bool_),
        (pd.Series(rng.integers(0, 1, length)), np.bool_),
        (pd.Series(np.random.choice([1 + 1e-12, 1e-12], length)), np.bool_),

        # Boolean - Nullable
        (pd.Series(np.random.choice([1, 0, None], length)).astype(float), pd.Int8Dtype()),
        (pd.Series(np.random.choice([1 + 1e-12, 1e-12, None], length)).astype(float), pd.Int8Dtype()),
    ]


def integer_mocks(length):
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
        *[(pd.Series(x, dtype=pd.Int64Dtype()), y) for x, y in zip(unsigned_ints, core.UINT_TYPES)],
        # Integer - Unsigned - Nullable
        *[
            (pd.Series(np.where(rng.uniform(size=length) > 0.5, x, None), dtype=pd.Int64Dtype()), y)
            for x, y in zip(unsigned_ints, core.UINT_NULLABLE_TYPES)
        ],
        # Integer - Signed
        *[(pd.Series(x, dtype=pd.Int64Dtype()), y) for x, y in zip(signed_ints, core.INT_TYPES)],
        # Integer - Signed - Nullable
        *[
            (pd.Series(np.where(rng.uniform(size=length) > 0.5, x, None), dtype=pd.Int64Dtype()), y)
            for x, y in zip(signed_ints, core.INT_NULLABLE_TYPES)
        ],
    ]


def float_mocks(length):
    floats = [
        np.random.uniform(-65503.9, 65499.9, length).astype(np.float16).astype(np.float64),
        np.random.uniform(-3.4028235e38, 3.4028235e38, length).astype(np.float32).astype(np.float64),
        np.random.uniform(-1.7976931348623157e200, 1.7976931348623157e200, length),
    ]
    return [
        # Float
        *[(pd.Series(x), y) for x, y in zip(floats, core.FLOAT_TYPES)]
    ]


# Categorical
def categorical_mocks(length):
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


def dataframe_mock(length) -> Tuple[pd.DataFrame, dict]:
    mocks = [boolean_mocks(length), integer_mocks(length), float_mocks(length), categorical_mocks(length)]
    input_series = [y[0] for x in mocks for y in x]
    expected_types = [y[1] for x in mocks for y in x]
    df = pd.concat(input_series, axis=1)
    schema = {k: v for k, v in zip(df.columns, expected_types)}
    return df, schema
