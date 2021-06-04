import datetime as dt
import itertools as it
from pprint import pprint
from typing import Dict, Iterable, List
from uuid import uuid4

import numpy as np  # type: ignore
import pandas as pd

from pdcast import minimum_viable_schema

np.random.seed(1234)

N = int(2e5)

# ---------- Generate Data ----------

# Basics
data: List[Iterable] = [
    np.ones(N),
    np.zeros(N),
    np.random.choice(
        [0, 1],
        size=N,
    ),
]


# Numeric signed and unsigned arrays of different orders
def numeric_arrays_of_order(n: int, order_max: int = 40, signed: bool = False):
    arr_list = []
    for order in range(order_max + 1):
        stop = 10 ** order
        start = -stop if signed else 0
        arr1 = np.linspace(start=start, stop=stop, num=n)
        arr2 = np.round(arr1, 0)
        arr_list.append(arr1)
        arr_list.append(arr2)
    return arr_list


unsigned_arrays = numeric_arrays_of_order(N)
signed_arrays = numeric_arrays_of_order(N, signed=True)


# String types with varying carinality
def string_array(n: int, categories: int = None):
    if categories:
        return np.array(
            [
                x[1]
                for x in zip(range(n), it.cycle([uuid4() for _ in range(categories)]))
            ]
        )
    return np.array([str(uuid4()) for _ in range(n)])


unique_strings = string_array(N)
categorical_strings = [
    string_array(N, categories=5),
    string_array(N, categories=int(N * 0.80)),
    string_array(N, categories=int(N * 0.90)),
    string_array(N, categories=int(N * 0.95)),
    string_array(N, categories=int(N * 0.99)),
]

data += unsigned_arrays + signed_arrays + [unique_strings] + categorical_strings

data_dict: Dict[str, Iterable] = {
    str(idx).rjust(3, "0"): col for idx, col in enumerate(data)
}

df00: pd.DataFrame = pd.DataFrame(data_dict).sample(frac=1)  # type: ignore


start = dt.datetime.now()
# TODO: get optimal cardinality thresh
schema = minimum_viable_schema(df00, cat_thresh=0.8)
end = dt.datetime.now()
start_size = round(np.sum(df00.memory_usage(deep=True)) / 1024 / 1024, 2)
df01 = df00.astype(schema)
end_size = round(np.sum(df01.memory_usage(deep=True)) / 1024 / 1024, 2)
pprint(schema)
print(f"Start size (MB): {start_size}MB")
print(f"End size: {end_size}MB")
print(f"Size reduction: {100 * round(1 - (end_size / start_size), 2)}%")
print(f"Duration: {end - start}")
# new_numeric = df01.select_dtypes(include=np.number)
# assert pd.testing.assert_frame_equal(
#     df00[new_numeric.columns], new_numeric, check_dtype=False
# )
