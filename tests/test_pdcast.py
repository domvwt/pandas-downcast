import pytest

import pdcast as pc
from tests.conftest import boolean_mocks, float_mocks, integer_mocks


@pytest.mark.parametrize("test_input,expected", boolean_mocks)
def test_smallest_viable_type_bool(test_input, expected):
    output = pc.smallest_viable_type(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", integer_mocks)
def test_smallest_viable_type_int(test_input, expected):
    output = pc.smallest_viable_type(test_input)
    assert output == expected


@pytest.mark.parametrize("test_input,expected", float_mocks)
def test_smallest_viable_type_float(test_input, expected):
    output = pc.smallest_viable_type(test_input)
    assert output == expected
