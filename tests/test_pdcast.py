import numpy as np
import pandas as pd
import pytest

import pdcast as pc

series_mocks = [
    (pd.Series(np.zeros(12)), np.bool_),
    (pd.Series(np.ones(12)), np.bool_),
    (pd.Series([1, 0] * 6), np.bool_),
    (pd.Series([1, 0, None] * 4), pd.Int8Dtype()),
    (pd.Series([None] * 12), pd.Int8Dtype()),
]


@pytest.mark.parametrize("test_input,expected", series_mocks)
def test_smallest_viable_type(test_input, expected):
    output = pc.smallest_viable_type(test_input)
    assert output == expected
