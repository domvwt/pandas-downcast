import pytest

import pdcast as pc
import pdcast.core as core
from pdcast.core import coerce_df, downcast, smallest_viable_type
from tests.conftest import (boolean_mocks, categorical_mocks, dataframe_mock,
                            float_mocks, integer_mocks)


@pytest.mark.parametrize("test_input,expected", boolean_mocks(1000))
def test_smallest_viable_type_bool(test_input, expected):
    output = smallest_viable_type(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", integer_mocks(1000))
def test_smallest_viable_type_int(test_input, expected):
    output = smallest_viable_type(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", float_mocks(1000))
def test_smallest_viable_type_float(test_input, expected):
    output = smallest_viable_type(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", categorical_mocks(1000))
def test_smallest_viable_type_categorical(test_input, expected):
    output = smallest_viable_type(test_input)
    assert output == expected


def test_infer_schema():
    input_df, expected = dataframe_mock(1000)
    output = pc.infer_schema(input_df)
    assert len(output) == len(expected)
    for k, expected_type in expected.items():
        output_type = output[k]
        assert output_type == expected_type


def test_infer_schema_big_df():
    input_df, expected = dataframe_mock(50_000)
    output = pc.infer_schema(input_df)
    assert len(output) == len(expected)
    for k, expected_type in expected.items():
        output_type = output[k]
        assert output_type == expected_type


def test_downcast():
    input_df, expected_schema = dataframe_mock(1000)
    expected_df = coerce_df(input_df, expected_schema)
    output_df, output_schema = downcast(input_df, return_schema=True)
    assert core.check_frames_equal(output_df, expected_df)  # type: ignore
    assert output_schema == expected_schema
