from typing import Any, Dict, Hashable, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from pandas._typing import FrameOrSeries

# TODO: Change string numerics to numeric types (optional)

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

# Default tolerance from numpy inexact numeric comparison
# See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html
RTOL = 1e-05
ATOL = 1e-08


class _ValidTypeFound(Exception):
    pass


def smallest_viable_type(
    srs: Series,
    cat_thresh=0.8,
    sample_size=None,
    rtol: float = None,
    atol: float = None,
):
    """Determine smallest viable type for Pandas Series."""

    original_dtype = srs.dtype
    valid_type = None

    def assign_valid_type(data_type):
        nonlocal valid_type
        valid_type = data_type
        raise _ValidTypeFound

    def first_valid_type(srs, candidate_types):
        for data_type in candidate_types:
            if type_cast_valid(srs, data_type):
                assign_valid_type(data_type)

    if sample_size and srs.size > sample_size:
        srs = srs.sample(n=sample_size)  # type: ignore

    try:
        if pd.api.types.is_numeric_dtype(original_dtype):
            val_range = srs.min(), srs.max()
            val_min, val_max = val_range[0], val_range[1]
            is_signed = val_range[0] < 0
            is_nullable = srs.isna().any()

            srs_mod_one = np.mod(srs.fillna(0), 1)
            is_decimal = not all(close_to_val(srs_mod_one, 0))

            if not is_decimal:
                # Identify 1 / 0 flags and cast to `np.bool`
                if (
                    close_to_0_or_1(val_min, rtol, atol)
                    and close_to_0_or_1(val_max, rtol, atol)
                    and srs.dropna().unique().shape[0] <= 2
                ):
                    # Convert values close to zero to exactly zero and values close to one to exactly one
                    # so that Pandas allows recast to int type
                    srs = np.where(close_to_val(srs, 0, rtol, atol), 0, srs)
                    srs = np.where(close_to_val(srs, 1, rtol, atol), 1, srs)
                    srs = Series(srs)

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
            unique_count = srs.dropna().unique().shape[0]
            srs_length = srs.dropna().shape[0]
            unique_pct = unique_count / srs_length
            # Cast to `categorical` if percentage of unique value less than threshold
            if unique_pct < cat_thresh:
                assign_valid_type(pd.CategoricalDtype())
            assign_valid_type(np.object_)

    except _ValidTypeFound:
        return valid_type

    # Keep original type if nothing else works
    return original_dtype


def infer_schema(
    data: FrameOrSeries,
    include: Iterable[Hashable] = None,
    exclude: Iterable[Hashable] = None,
    cat_thresh: float = 0.8,
    sample_size: int = 10_000,
    rtol: float = None,
    atol: float = None,
) -> Dict[Any, Any]:
    """Infer minimum viable schema."""
    data = data.copy()
    # Use head and tail in case data is sorted
    if sample_size and data.shape[0] > sample_size:
        data = sample_head_tail(data)
    if isinstance(data, Series):
        schema = {data.name: smallest_viable_type(data)}
    if isinstance(data, DataFrame):
        target_cols = include or data.columns
        if exclude:
            target_cols = [col for col in target_cols if col not in set(exclude)]
        schema = {
            col: (
                smallest_viable_type(srs, cat_thresh=cat_thresh, rtol=rtol, atol=atol)
                if col in set(target_cols)
                else srs.dtype
            )
            for col, srs in data.iteritems()
        }
    else:
        raise TypeError(data)
    return schema


def coerce_df(df: DataFrame, schema: dict) -> FrameOrSeries:
    df = df.copy()
    try:
        df = df.astype(schema)  # type: ignore
    except TypeError:
        for col, dtype in schema.items():
            df[col] = coerce_series(df[col], dtype)
    return df


def coerce_series(srs: Series, dtype: Any) -> FrameOrSeries:
    srs = srs.copy()
    try:
        srs = srs.astype(dtype)  # type: ignore
    except TypeError:
        if (
            pd.api.types.is_integer_dtype(dtype)
            and not pd.api.types.is_integer_dtype(srs)
            and srs.dropna().shape[0]
        ):
            srs = srs.round(0).astype(dtype)  # type: ignore
        else:
            srs = srs.astype(dtype)  # type: ignore
    return srs


def downcast(
    data: FrameOrSeries,
    include: Iterable[Hashable] = None,
    exclude: Iterable[Hashable] = None,
    return_schema: bool = False,
    cat_thresh: float = 0.8,
    sample_size: int = 10_000,
    rtol: float = None,
    atol: float = None,
) -> Union[DataFrame, Series, Tuple[FrameOrSeries, dict]]:
    """Infer and apply minimum viable schema."""
    data = data.copy()
    schema = infer_schema(
        data,
        include=include,
        exclude=exclude,
        cat_thresh=cat_thresh,
        sample_size=sample_size,
        rtol=rtol,
        atol=atol,
    )
    if isinstance(data, Series):
        dtype = schema[data.name]
        data = coerce_series(data, dtype)
    else:
        data = coerce_df(data, schema)
    if return_schema:
        return data, schema
    return data


def sample_head_tail(data: FrameOrSeries, sample_size: int = 10_000):
    """Sample from head and tail of DataFrame."""
    if data.shape[0] > sample_size:
        half_sample = sample_size // 2
        data = data[:half_sample].append(data[-half_sample:])
    return data


def type_cast_valid(
    srs: Series, data_type: Any, rtol: float = None, atol: float = None
) -> bool:
    """Check `srs` can be cast to `data_type` without loss of information."""
    try:
        srs_new = srs.astype(data_type)
    except TypeError:
        return False
    sdtype = srs.dtype
    rtol = rtol or RTOL
    atol = atol or ATOL
    if is_numeric_typelike(sdtype) and is_numeric_typelike(data_type):
        return np.allclose(srs_new, srs, equal_nan=True, rtol=rtol, atol=atol)
    else:
        return Series(srs == srs_new).all()


def is_numeric_typelike(x) -> bool:
    return (isinstance(x, type) and issubclass(x, np.number)) or (
        isinstance(x, np.dtype) and np.issubdtype(x, np.number)
    )


def close_to_val(
    srs: Series, val: Union[int, float], rtol: float = None, atol: float = None
) -> Series:
    rtol = rtol or RTOL
    atol = atol or ATOL
    try:
        return np.isclose(srs, val, rtol=rtol, atol=atol)
    except TypeError:
        return pd.Series(srs == val)


def check_frames_equal(
    a: DataFrame, b: DataFrame, rtol: float = None, atol: float = None
) -> bool:
    rtol = rtol or RTOL
    atol = atol or ATOL
    cond1 = all(a.columns == b.columns)
    cond2 = a.shape == b.shape
    if not (cond1 and cond2):
        return False
    for i in range(a.shape[1]):
        a_col = a.iloc[:, i]
        b_col = b.iloc[:, i]
        try:
            if not np.allclose(a_col, b_col, rtol=rtol, atol=atol):
                return False
        except TypeError:
            if not a_col.tolist() == b_col.tolist():
                return False
    return True


def close_to_0_or_1(x: np.number, rtol: float = None, atol: float = None) -> bool:
    """Check if `x` is close to zero or one."""
    rtol = rtol or RTOL
    atol = atol or ATOL
    try:
        return np.isclose(x, 0, rtol=rtol, atol=atol) or np.isclose(
            x, 1, rtol=rtol, atol=atol
        )
    except TypeError:
        return x == 0 or x == 1
