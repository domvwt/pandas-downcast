# -*- coding: utf-8 -*-
"""Core functions for downcasting Pandas DataFrames and Series."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from pandas._typing import FrameOrSeries
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import pdcast.types as tc


@dataclass
class Options:
    """Config options for `pdcast`."""

    RTOL = 1e-05
    """Default relative tolerance for numpy inexact numeric comparison
    See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html"""

    ATOL = 1e-08
    """Default absolute tolerance for numpy inexact numeric comparison
    See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html"""


options = Options()


class _ValidTypeFound(Exception):
    """ """


def infer_dtype(
    series: Series,
    cat_thresh: float = 0.8,
    rtol: float = None,
    atol: float = None,
):
    """Determine smallest viable type for Pandas Series.

    Args:
        series (Series): Pandas Series.
        cat_thresh (float): Categorical value threshold. (Default value = 0.8)
            Non-numeric variables with proportion of unique values less than
            `cat_thresh` will be cast to Categorical type.
        rtol (float): Absolute tolerance for numeric equality. (Default value = None)
        atol (float): Relative tolerance for numeric equality. (Default value = None)

    Returns:
        Smallest viable Numpy or Pandas data type for `series`.

    """

    original_dtype = series.dtype
    valid_type = None

    def assign_valid_type(data_type):
        nonlocal valid_type
        valid_type = data_type
        raise _ValidTypeFound

    def first_valid_type(srs, candidate_types):
        for data_type in candidate_types:
            if type_cast_valid(srs, data_type):
                assign_valid_type(data_type)

    try:
        if pd.api.types.is_numeric_dtype(original_dtype):
            val_range = series.min(), series.max()
            val_min, val_max = val_range[0], val_range[1]
            is_signed = val_range[0] < 0
            is_nullable = series.isna().any()

            srs_mod_one = np.mod(series.fillna(0), 1)
            is_decimal = not all(close_to_val(srs_mod_one, 0))

            if not is_decimal:
                # Identify 1 / 0 flags and cast to `np.bool`
                if (
                    close_to_0_or_1(val_min, rtol, atol)
                    and close_to_0_or_1(val_max, rtol, atol)
                    and series.dropna().unique().shape[0] <= 2
                ):
                    # Convert values close to zero to exactly zero and values close to one to
                    # exactly one so that Pandas allows recast to int type
                    series = np.where(close_to_val(series, 0, rtol, atol), 0, series)
                    series = np.where(close_to_val(series, 1, rtol, atol), 1, series)
                    series = Series(series)

                    first_valid_type(series, tc.BOOLEAN_TYPES)

                if is_nullable:
                    if is_signed:
                        first_valid_type(series, tc.INT_NULLABLE_TYPES)

                    # Unsigned
                    else:
                        # Other integer types
                        first_valid_type(series, tc.UINT_NULLABLE_TYPES)

                # Not nullable
                else:
                    if is_signed:
                        first_valid_type(series, tc.INT_TYPES)
                    # Unsigned
                    else:
                        first_valid_type(series, tc.UINT_TYPES)
            # Float type
            first_valid_type(series, tc.FLOAT_TYPES)

        elif series.isna().all():
            assign_valid_type(tc.ALL_NAN_TYPE)

        # Non-numeric
        else:
            unique_count = series.dropna().unique().shape[0]
            srs_length = series.dropna().shape[0]
            unique_pct = unique_count / srs_length
            # Cast to `categorical` if percentage of uniques value less than threshold
            if unique_pct < cat_thresh:
                assign_valid_type(pd.CategoricalDtype())
            assign_valid_type(np.object_)

    except _ValidTypeFound:
        return valid_type

    # Keep original type if nothing else works
    except TypeError:
        pass

    return original_dtype


def infer_schema(
    data: FrameOrSeries,
    include: Iterable[Hashable] = None,
    exclude: Iterable[Hashable] = None,
    sample_size: int = 10_000,
    infer_dtype_kws: Dict[str, Any] = None,
) -> Dict[Any, Any]:
    """Infer minimum viable schema for `data`.

    Args:
        data (FrameOrSeries): Pandas DataFrame or Series.
        include (Iterable[Hashable]): Columns to include. (Default value = None)
            Excludes all other columns if defined.
        exclude (Iterable[Hashable]): Columns to exclude. (Default value = None)
        sample_size (int): Number of records to take from head and tail. (Default value = 10_000)
        infer_dtype_kws (Dict[Any, Any]): Keyword arguments for `infer_dtype`. (Default value = None)

    Returns:
        Dict[str, Any]: Inferred schema.

    """
    if not isinstance(data, (DataFrame, Series)):
        raise TypeError(type(data))
    data = data.copy()
    infer_dtype_kws = infer_dtype_kws or {}
    # Use head and tail in case data is sorted
    if sample_size and data.shape[0] > sample_size:
        data = take_head_and_tail(data)
    if isinstance(data, Series):
        schema = {data.name: infer_dtype(data, **infer_dtype_kws)}
    else:  # DataFrame
        target_cols = include or data.columns
        if exclude:
            target_cols = [col for col in target_cols if col not in set(exclude)]
        schema = {
            col: (
                infer_dtype(srs, **infer_dtype_kws)
                if col in set(target_cols)
                else srs.dtype
            )
            for col, srs in data.iteritems()
        }
    return schema


def coerce_df(df: DataFrame, schema: dict) -> FrameOrSeries:
    """Coerce DataFrame to `schema`.

    Args:
        df (DataFrame): Pandas DataFrame.
        schema (dict): Target schema.

    Returns:
        DataFrame: Pandas DataFrame with `schema`.

    """
    df = df.copy()
    try:
        df = df.astype(schema)  # type: ignore
    except TypeError:
        for col, dtype in schema.items():
            df[col] = coerce_series(df[col], dtype)
    return df


def coerce_series(series: Series, dtype: Any) -> FrameOrSeries:
    """Coerce Series to `dtype`.

    Args:
        series (Series): Pandas Series.
        dtype (Any): Target data type.

    Returns:
        Series: Pandas Series of type `dtype`.

    """
    series = series.copy()
    try:
        series = series.astype(dtype)  # type: ignore
    except TypeError:
        # TypeError thrown when converting float to int without rounding
        series = series.round(0).astype(dtype)  # type: ignore
    return series


def downcast(
    data: FrameOrSeries,
    include: Iterable[Hashable] = None,
    exclude: Iterable[Hashable] = None,
    return_schema: bool = False,
    sample_size: int = 10_000,
    infer_dtype_kws: Dict[str, Any] = None,
) -> Union[DataFrame, Series, Tuple[FrameOrSeries, dict]]:
    """Infer and apply minimum viable schema.

    Args:
        data (FrameOrSeries):
        include (Iterable[Hashable]): Columns to include. (Default value = None)
            Excludes all other columns if defined.
        exclude (Iterable[Hashable]): Columns to exclude. (Default value = None)
        return_schema (bool): Return inferred schema if True. (Default value = False)
        sample_size (int): Number of records to take from head and tail. (Default value = 10_000)
        infer_dtype_kws (Dict[str, Any]): Keyword arguments for `infer_dtype`. (Default value = None)

    Returns:
        FrameOrSeries: Downcast Pandas DataFrame or Series.
        Dict[Any, Any]: Inferred schema. (if `return_schema` is True)

    """
    data = data.copy()
    schema = infer_schema(
        data,
        include=include,
        exclude=exclude,
        sample_size=sample_size,
        infer_dtype_kws=infer_dtype_kws,
    )
    if isinstance(data, Series):
        dtype = schema[data.name]
        data = coerce_series(data, dtype)
    elif isinstance(data, DataFrame):
        data = coerce_df(data, schema)
    if return_schema:
        return data, schema
    return data


def take_head_and_tail(data: FrameOrSeries, sample_size: int = 10_000) -> FrameOrSeries:
    """Take head and tail of DataFrame.

    Args:
        data (FrameOrSeries): Pandas DataFrame or Series.
        sample_size (int): Number of records to take. (Default value = 10_000)

    Returns:
        FrameOrSeries: Resampled `data`.

    """
    if data.shape[0] > sample_size:
        half_sample = sample_size // 2
        data = data[:half_sample].append(data[-half_sample:])
    return data


def type_cast_valid(
    series: Series, data_type: Any, rtol: float = None, atol: float = None
) -> bool:
    """Check `series` can be cast to `data_type` without loss of information.

    Args:
        series (Series): Pandas Series.
        data_type (Any): Any data type.
        rtol (float): Absolute tolerance for numeric equality. (Default value = None)
        atol (float): Relative tolerance for numeric equality. (Default value = None)

    Returns:
        bool: True if type if valid, False otherwise.

    """
    try:
        srs_new = series.astype(data_type)
    except TypeError:
        return False
    sdtype = series.dtype
    rtol = rtol or options.RTOL
    atol = atol or options.ATOL
    if is_numeric_typelike(sdtype) and is_numeric_typelike(data_type):
        return np.allclose(srs_new, series, equal_nan=True, rtol=rtol, atol=atol)
    return Series(series == srs_new).all()


def is_numeric_typelike(dtype) -> bool:
    """Check whether `dtype` is a numeric type or class.

    Args:
        dtype: Any data type class or object.

    Returns:
        bool: True if numeric type, False otherwise.

    """
    return (isinstance(dtype, type) and issubclass(dtype, np.number)) or (
        isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.number)
    )


def close_to_val(
    series: Series, val: Union[int, float], rtol: float = None, atol: float = None
) -> Series:
    """Check all `series` values close to `val`.

    Args:
        series (Series): Pandas Series.
        val (Union[int, float]): Value for comparison.
        rtol (float): Absolute tolerance for numeric equality. (Default value = None)
        atol (float): Relative tolerance for numeric equality. (Default value = None)

    Returns:
        bool: True if all close to `val`, False otherwise.

    """
    rtol = rtol or options.RTOL
    atol = atol or options.ATOL
    try:
        return np.isclose(series, val, rtol=rtol, atol=atol)
    except TypeError:
        return pd.Series(series == val)


def close_to_0_or_1(num: np.number, rtol: float = None, atol: float = None) -> bool:
    """Check if `num` is close to zero or one.

    Args:
        num (np.number): Number for comparison.
        rtol (float): Absolute tolerance for numeric equality. (Default value = None)
        atol (float): Relative tolerance for numeric equality. (Default value = None)

    Returns:
        bool: True if all close to 0 or 1, False otherwise.

    """
    rtol = rtol or options.RTOL
    atol = atol or options.ATOL
    return np.isclose(num, 0, rtol=rtol, atol=atol) or np.isclose(
        num, 1, rtol=rtol, atol=atol
    )
