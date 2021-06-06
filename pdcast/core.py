from typing import Any, Dict

import numpy as np
import pandas as pd

ALL_NAN_TYPE = pd.Int8Dtype()

BOOLEAN_TYPES = [np.bool_, pd.Int8Dtype()]

UINT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]

UINT_NULLABLE_TYPES = [
    pd.UInt8Dtype(),
    pd.UInt16Dtype(),
    pd.UInt32Dtype(),
    pd.UInt64Dtype(),
]

INT_TYPES = [np.int8, np.int16, np.int32, np.int64]

INT_NULLABLE_TYPES = [
    pd.Int8Dtype(),
    pd.Int16Dtype(),
    pd.Int32Dtype(),
    pd.Int64Dtype(),
]

FLOAT_TYPES = [np.float16, np.float32, np.float64, np.float128]


class ValidTypeFound(Exception):
    pass


def smallest_viable_type(srs: pd.Series, cat_thresh=0.95, sample_size=None):
    """Determine smallest viable type for Pandas Series."""

    original_dtype = srs.dtype
    valid_type = None

    def assign_valid_type(data_type):
        nonlocal valid_type
        valid_type = data_type
        raise ValidTypeFound

    def first_valid_type(srs, candidate_types):
        for data_type in candidate_types:
            if type_cast_valid(srs, data_type):
                assign_valid_type(data_type)

    if sample_size and srs.size > sample_size:
        srs = srs.sample(n=sample_size)  # type: ignore

    try:
        if np.issubdtype(original_dtype, np.number):
            val_range = srs.min(), srs.max()
            val_min, val_max = val_range[0], val_range[1]
            is_signed = val_range[0] < 0
            is_nullable = srs.isna().sum() > 0
            is_decimal = not np.isclose(np.nansum(np.mod(srs.values, 1)), 0)

            if not is_decimal:
                # Identify 1 / 0 flags and cast to `np.bool`
                if (
                    close_to_0_or_1(val_min)
                    and close_to_0_or_1(val_max)
                    and srs.dropna().unique().shape[0] <= 2
                ):
                    first_valid_type(srs, BOOLEAN_TYPES)

                if is_nullable:
                    if is_signed:
                        first_valid_type(srs, INT_NULLABLE_TYPES)

                    # Unsigned
                    else:
                        # Other integer types
                        first_valid_type(srs, UINT_NULLABLE_TYPES)

                # Not nullable
                else:
                    if is_signed:
                        first_valid_type(srs, INT_TYPES)
                    # Unsigned
                    else:
                        first_valid_type(srs, UINT_TYPES)
            # Float type
            first_valid_type(srs, FLOAT_TYPES)

        elif srs.isna().all():
            assign_valid_type(ALL_NAN_TYPE)

        # Non-numeric
        else:
            unique_count = srs.unique().shape[0]
            srs_length = srs.size
            unique_pct = unique_count / srs_length
            # Cast to `categorical` if percentage of unique value less than threshold
            if unique_pct < cat_thresh:
                assign_valid_type(pd.CategoricalDtype())
            assign_valid_type(np.object_)

    except ValidTypeFound:
        return valid_type

    # Keep original type if nothing else works
    return original_dtype


def minimum_viable_schema(
    df: pd.DataFrame, cat_thresh=0.8, sample_size=10_000
) -> Dict[str, Any]:
    """Determine minimum viable schema for Pandas DataFrame."""
    df = df.copy()
    if sample_size:
        # Use head and tail in case DataFrame is sorted
        df = sample_head_tail(df)
    schema = {
        str(col): smallest_viable_type(srs, cat_thresh=cat_thresh)
        for col, srs in df.iteritems()
    }
    return schema


def sample_head_tail(df: pd.DataFrame, sample_size: int = 10_000):
    """Sample from head and tail of DataFrame."""
    half_sample = sample_size // 2
    return df[:half_sample].append(df[-half_sample:])


def close_to_0_or_1(x: np.number) -> bool:
    """Check if `x` is close to zero or one."""
    return np.isclose(x, 0) or np.isclose(x, 1)


def type_cast_valid(srs: pd.Series, data_type) -> bool:
    """Check `srs` can be cast to `data_type` without loss of information."""
    srs_new = srs.astype(data_type)
    if isinstance(srs.dtype, np.number) and isinstance(data_type, np.number):
        return np.allclose(srs_new, srs, equal_nan=True)
    else:
        return pd.Series(srs == srs_new).all()
