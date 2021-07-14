import numpy as np
import pandas as pd
import pytest
from pandas.core.frame import DataFrame

import pdcast as pdc
from pdcast.core import (
    coerce_df,
    downcast,
    infer_dtype,
    take_head_and_tail,
    type_cast_valid,
)
from tests.conftest import (
    boolean_mocks,
    categorical_mocks,
    dataframe_mock,
    dicts_equal,
    float_mocks,
    frames_equal,
    integer_mocks,
    series_date_types,
    series_equal,
    series_mocks,
    series_numpy_only,
)


@pytest.mark.parametrize("test_input,expected", boolean_mocks(1000))
def test_smallest_viable_type_bool(test_input, expected):
    output = infer_dtype(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", integer_mocks(1000))
def test_smallest_viable_type_int(test_input, expected):
    output = infer_dtype(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", float_mocks(1000))
def test_smallest_viable_type_float(test_input, expected):
    output = infer_dtype(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", categorical_mocks(1000))
def test_smallest_viable_type_categorical(test_input, expected):
    output = infer_dtype(test_input)
    assert output == expected


def test_infer_schema():
    input_df, expected = dataframe_mock(1000)
    output = pdc.infer_schema(input_df)
    assert len(output) == len(expected)
    assert dicts_equal(output, expected)


def test_infer_schema_big_df():
    input_df, expected = dataframe_mock(50_000)
    output = pdc.infer_schema(input_df)
    assert len(output) == len(expected)
    assert dicts_equal(output, expected)


def test_downcast_dataframe():
    input_df, expected_schema = dataframe_mock(1000)
    expected_df = coerce_df(input_df, expected_schema)
    output_df: DataFrame = downcast(input_df, return_schema=False)  # type: ignore
    output_schema = output_df.dtypes.to_dict()
    assert frames_equal(output_df, expected_df)  # type: ignore
    assert dicts_equal(output_schema, expected_schema)


def test_downcast_series():
    input_df, expected_schema = dataframe_mock(1000)
    for name, input_series in input_df.iteritems():
        output_series, output_schema = pdc.downcast(input_series, return_schema=True)
        series_schema = {name: expected_schema[name]}
        expected_series = pdc.coerce_series(input_series, series_schema)
        assert dicts_equal(output_schema, series_schema)
        assert series_equal(output_series, expected_series)  # type: ignore


def test_downcast_bad_type():
    input_list = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        pdc.downcast(input_list)  # type: ignore


def test_infer_schema_exclude():
    input_df, expected_schema = dataframe_mock(1000)
    include = [2, 4, 6]
    exclude = [x for x in range(input_df.shape[1]) if x not in include]
    output_schema = pdc.infer_schema(input_df, exclude=exclude)
    expected_schema = input_df.dtypes.to_dict()
    expected_schema.update({2: np.bool_, 4: pd.Int8Dtype(), 6: np.uint16})
    assert dicts_equal(expected_schema, output_schema)


def test_infer_schema_include():
    input_df, expected_schema = dataframe_mock(1000)
    include = [2, 4, 6]
    output_schema = pdc.infer_schema(input_df, include=include)
    expected_schema = input_df.dtypes.to_dict()
    expected_schema.update({2: np.bool_, 4: pd.Int8Dtype(), 6: np.uint16})
    assert dicts_equal(expected_schema, output_schema)


@pytest.mark.parametrize("series,expected_dtype", series_mocks(1000))
def test_infer_schema_series(series, expected_dtype):
    name = "series_name"
    series.name = name
    output_schema = pdc.infer_schema(series)
    expected_schema = {name: expected_dtype}
    assert dicts_equal(output_schema, expected_schema)


def test_infer_schema_bad_type():
    input_list = [1, 2, 3, 4, 5]
    with pytest.raises(TypeError):
        pdc.infer_schema(input_list)  # type: ignore


def test_type_cast_valid():
    input_series = pd.Series([float(x * 1.2) for x in range(10)])
    new_dtype = pd.Int8Dtype()
    result = type_cast_valid(input_series, new_dtype)
    assert not result


def test_head_and_tail_bypass():
    input_df, _ = dataframe_mock(1000)
    output_df: DataFrame = take_head_and_tail(input_df)  # type: ignore
    assert frames_equal(input_df, output_df)


@pytest.mark.parametrize("input_srs", series_date_types)
def test_datetypes(input_srs):
    input_dtype = input_srs.dtype
    output_dtype = infer_dtype(input_srs)
    assert input_dtype == output_dtype


@pytest.mark.parametrize("input_srs,expected", series_numpy_only(1000))
def test_numpy_only(input_srs, expected):
    output_dtype = infer_dtype(input_srs, numpy_dtypes_only=True)
    assert output_dtype == expected
