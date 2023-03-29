# -*- coding: utf-8 -*-
"""Core functions for downcasting Pandas DataFrames and Series."""

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series

import pdcast.types as tc

try:
    from typing import Literal, TypeVar
except ImportError:
    from typing import TypeVar  # type: ignore

    from typing_extensions import Literal  # type: ignore


PANDAS_VERSION = tuple(int(x) for x in pd.__version__.split(".")[:2])


T = TypeVar("T", DataFrame, Series)


@dataclass
class Options:
    """Config options for `pdcast`."""

    RTOL = 1e-05
    """Default relative tolerance for numpy inexact numeric comparison
    See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html"""

    ATOL = 1e-08
    """Default absolute tolerance for numpy inexact numeric comparison
    See: https://numpy.org/doc/stable/reference/generated/numpy.allclose.html"""

    numpy_dtypes_only = False


options = Options()


class _ValidTypeFound(Exception):
    ...


def infer_dtype(
    series: Series,
    cat_thresh: float = 0.8,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    numpy_dtypes_only: Optional[bool] = None,
) -> Any:
    """Determine smallest viable type for Pandas Series.

    Args:
        series: Pandas Series.
        cat_thresh: Categorical value threshold. (Default value = 0.8)
            Non-numeric variables with proportion of unique values less than
            `cat_thresh` will be cast to Categorical type.
        rtol: Absolute tolerance for numeric equality. (Default value = None)
        atol: Relative tolerance for numeric equality. (Default value = None)
        numpy_dtypes_only: Use only Numpy dtypes for schema. (Default value = None)

    Returns:
        Smallest viable Numpy or Pandas data type for `series`.

    """

    original_dtype = series.dtype
    valid_type = None

    numpy_dtypes_only = numpy_dtypes_only or options.numpy_dtypes_only

    def assign_valid_type(data_type) -> None:
        nonlocal valid_type
        valid_type = data_type
        raise _ValidTypeFound

    def first_valid_type(srs, candidate_types) -> None:
        for data_type in candidate_types:
            if type_cast_valid(srs, data_type):
                assign_valid_type(data_type)

    try:
        if pd.api.types.is_numeric_dtype(original_dtype):
            val_range = series.min(), series.max()
            val_min, val_max = val_range[0], val_range[1]
            is_signed = val_range[0] < 0
            is_nullable = series.isna().any()

            srs_mod_one = np.mod(series.fillna(0), 1)  # type: ignore
            is_decimal = not all(close_to_val(srs_mod_one, 0))

            if not is_decimal:
                # Identify 1 / 0 flags and cast to `np.bool`
                if (
                    close_to_0_or_1(val_min, rtol, atol)
                    and close_to_0_or_1(val_max, rtol, atol)
                    and series.dropna().unique().shape[0] == 2
                ):
                    # Convert values close to zero to exactly zero and values close to one to
                    # exactly one so that Pandas allows recast to int type
                    series = np.where(close_to_val(series, 0, rtol, atol), 0, series)
                    series = np.where(close_to_val(series, 1, rtol, atol), 1, series)
                    series = Series(series)

                    first_valid_type(series, tc.BOOLEAN_TYPES)

                if is_nullable and not numpy_dtypes_only:
                    if is_signed:
                        first_valid_type(series, tc.INT_NULLABLE_TYPES)

                    # Unsigned
                    else:
                        # Other integer types
                        first_valid_type(series, tc.UINT_NULLABLE_TYPES)

                # Not nullable
                elif not is_nullable:
                    if is_signed:
                        first_valid_type(series, tc.INT_TYPES)
                    # Unsigned
                    else:
                        first_valid_type(series, tc.UINT_TYPES)
            # Float type
            first_valid_type(series, tc.FLOAT_TYPES)

        elif series.isna().all():
            assign_valid_type(tc.ALL_NAN_TYPE)

        # Non-numeric (and not a date)
        elif original_dtype == object:
            if not numpy_dtypes_only:
                unique_count = series.dropna().unique().shape[0]
                srs_length = series.dropna().shape[0]
                unique_pct = unique_count / srs_length
                # Cast to `categorical` if percentage of unique values less than threshold
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
    data: T,
    include: Optional[Iterable[Hashable]] = None,
    exclude: Optional[Iterable[Hashable]] = None,
    sample_size: Optional[int] = None,
    numpy_dtypes_only: Optional[bool] = None,
    infer_dtype_kws: Optional[Dict[str, Any]] = None,
) -> Dict[Any, Any]:
    """Infer minimum viable schema for `data`.

    Args:
        data: Pandas DataFrame or Series.
        include: Columns to include. (Default value = None)
            Excludes all other columns if defined.
        exclude: Columns to exclude. (Default value = None)
        sample_size: Number of records to take from head and tail. (Default value = None)
        numpy_dtypes_only: Use only Numpy dtypes for schema. (Default value = None)
        infer_dtype_kws: Keyword arguments for `infer_dtype`. (Default value = None)

    Returns:
        Inferred schema.

    """
    if not isinstance(data, (DataFrame, Series)):
        raise TypeError(type(data))
    data = data.copy()
    infer_dtype_kws = infer_dtype_kws or {}
    # Use head and tail in case data is sorted
    if sample_size and data.shape[0] > sample_size:
        data = take_head_and_tail(data)  # type: ignore
    if isinstance(data, Series):
        schema = {
            data.name: infer_dtype(
                data, numpy_dtypes_only=numpy_dtypes_only, **infer_dtype_kws
            )
        }
    else:  # DataFrame
        target_cols: Iterable[Hashable] = include or data.columns  # type: ignore
        if exclude:
            target_cols = [col for col in target_cols if col not in set(exclude)]
        if PANDAS_VERSION < (1, 5):
            schema = {
                col: (
                    infer_dtype(
                        srs, numpy_dtypes_only=numpy_dtypes_only, **infer_dtype_kws
                    )
                    if col in set(target_cols)
                    else srs.dtype
                )
                for col, srs in data.iteritems()  # type: ignore
            }
        else:
            schema = {
                col: (
                    infer_dtype(
                        srs, numpy_dtypes_only=numpy_dtypes_only, **infer_dtype_kws  # type: ignore
                    )
                    if col in set(target_cols)
                    else srs.dtype  # type: ignore
                )
                for col, srs in data.items()
            }
    return schema


def coerce_df(df: DataFrame, schema: Dict[Hashable, Any]) -> DataFrame:
    """Coerce DataFrame to `schema`.

    Args:
        df: Pandas DataFrame.
        schema: Target schema.

    Returns:
        Pandas DataFrame with `schema`.

    """
    df = df.copy()
    try:
        df = df.astype(schema)  # type: ignore
    except TypeError:
        for col, dtype in schema.items():
            df[col] = coerce_series(Series(df.loc[:, col]), dtype)  # type: ignore
    return df


def coerce_series(series: Series, dtype: Any) -> Series:
    """Coerce Series to `dtype`.

    Args:
        series (Series): Pandas Series.
        dtype (Any): Target data type.

    Returns:
        Series: Pandas Series of type `dtype`.

    """
    series = series.copy()
    try:
        series = series.astype(dtype)
    except TypeError:
        # TypeError thrown when converting float to int without rounding
        series = series.round(0).astype(dtype)
    return series


@overload
def downcast(  # type: ignore
    data: T,
    include: Optional[Iterable[Hashable]] = None,
    exclude: Optional[Iterable[Hashable]] = None,
    return_schema: Literal[False] = False,
    sample_size: Optional[int] = None,
    numpy_dtypes_only: bool = False,
    infer_dtype_kws: Optional[Dict[str, Any]] = None,
) -> T:
    ...


@overload
def downcast(
    data: T,
    include: Optional[Iterable[Hashable]] = None,
    exclude: Optional[Iterable[Hashable]] = None,
    return_schema: Literal[True] = True,
    sample_size: Optional[int] = None,
    numpy_dtypes_only: bool = False,
    infer_dtype_kws: Optional[Dict[str, Any]] = None,
) -> Tuple[T, Dict[Hashable, Any]]:
    ...


def downcast(
    data: T,
    include: Optional[Iterable[Hashable]] = None,
    exclude: Optional[Iterable[Hashable]] = None,
    return_schema: bool = False,
    sample_size: Optional[int] = None,
    numpy_dtypes_only: bool = False,
    infer_dtype_kws: Optional[Dict[str, Any]] = None,
) -> Union[T, Tuple[T, Dict[Hashable, Any]]]:
    """Infer and apply minimum viable schema.

    Args:
        data: Pandas DataFrame or Series to downcast.
        include: Columns to include. (Default value = None)
            Excludes all other columns if defined.
        exclude: Columns to exclude. (Default value = None)
        return_schema: Return inferred schema if True. (Default value = False)
        sample_size: Number of records to take from head and tail. (Default value = None)
        numpy_dtypes_only: Use only Numpy dtypes for schema. (Default value = None)
        infer_dtype_kws: Keyword arguments for `infer_dtype`. (Default value = None)

    Returns:
        Downcast Pandas DataFrame or Series.
        Inferred schema. (if `return_schema` is True)

    """
    data = data.copy()
    schema = infer_schema(
        data,
        include=include,
        exclude=exclude,
        sample_size=sample_size,
        numpy_dtypes_only=numpy_dtypes_only,
        infer_dtype_kws=infer_dtype_kws,
    )
    if isinstance(data, Series):
        dtype = schema[data.name]
        data = coerce_series(data, dtype)
    elif isinstance(data, DataFrame):  # DataFrame
        data = coerce_df(data, schema)
    if return_schema:
        return data, schema
    return data


def take_head_and_tail(data: T, sample_size: int = 10_000) -> T:
    """Take head and tail of DataFrame or Series.

    Args:
        data: Pandas DataFrame or Series.
        sample_size: Number of records to take. (Default value = 10_000)

    Returns:
        Resampled `data`.

    """
    rows = data.shape[0]
    if rows > sample_size:
        half_sample = sample_size // 2
        data = pd.concat([data.iloc[:half_sample], data.iloc[-half_sample:]])  # type: ignore
    return data


def type_cast_valid(
    series: Series,
    data_type: Any,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> bool:
    """Check `series` can be cast to `data_type` without loss of information.

    Args:
        series: Pandas Series.
        data_type: Any data type.
        rtol: Absolute tolerance for numeric equality. (Default value = None)
        atol: Relative tolerance for numeric equality. (Default value = None)

    Returns:
        True if type is valid, False otherwise.

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
    return bool(Series(series == srs_new).all())


def is_numeric_typelike(dtype) -> bool:
    """Check whether `dtype` is a numeric type or class.

    Args:
        dtype: Any data type class or object.

    Returns:
        True if numeric type, False otherwise.

    """
    return (isinstance(dtype, type) and issubclass(dtype, np.number)) or (
        isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.number)
    )


def close_to_val(
    series: Series,
    val: Union[int, float],
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> Series:
    """Check all `series` values close to `val`.

    Args:
        series: Pandas Series.
        val: Value for comparison.
        rtol: Absolute tolerance for numeric equality. (Default value = None)
        atol: Relative tolerance for numeric equality. (Default value = None)

    Returns:
        True if all close to `val`, False otherwise.

    """
    rtol = rtol or options.RTOL
    atol = atol or options.ATOL
    try:
        return np.isclose(series, val, rtol=rtol, atol=atol)
    except TypeError:
        return pd.Series(series == val)


def close_to_0_or_1(
    num: np.number, rtol: Optional[float] = None, atol: Optional[float] = None
) -> bool:
    """Check if `num` is close to zero or one.

    Args:
        num: Number for comparison.
        rtol: Absolute tolerance for numeric equality. (Default value = None)
        atol: Relative tolerance for numeric equality. (Default value = None)

    Returns:
        True if all close to 0 or 1, False otherwise.

    """
    rtol = rtol or options.RTOL
    atol = atol or options.ATOL
    return bool(
        np.isclose(num, 0, rtol=rtol, atol=atol)
        or np.isclose(num, 1, rtol=rtol, atol=atol)
    )
