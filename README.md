Pandas Downcast
===============

[![image](https://img.shields.io/pypi/v/pandas-downcast.svg)](https://pypi.python.org/pypi/pandas-downcast)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pandas-downcast.svg)](https://pypi.python.org/pypi/pandas-downcast/)
[![Build Status](https://travis-ci.com/domvwt/pandas-downcast.svg?branch=main)](https://travis-ci.com/domvwt/pandas-downcast)
[![codecov](https://codecov.io/gh/domvwt/pandas-downcast/branch/main/graph/badge.svg?token=TQPLURKQ9Z)](https://codecov.io/gh/domvwt/pandas-downcast)

Safely infer minimum viable schema for Pandas `DataFrame` and `Series`.

## Installation
```bash
pip install pandas-downcast
```

## Dependencies
* python >= 3.6
* pandas
* numpy

## License
[MIT](https://opensource.org/licenses/MIT)

## Usage
```python
import pdcast as pdc

import numpy as np
import pandas as pd

data = {
    "integers": np.linspace(1, 100, 100),
    "floats": np.linspace(1, 1000, 100).round(2),
    "booleans": np.random.choice([1, 0], 100),
    "categories": np.random.choice(["foo", "bar", "baz"], 100),
}

df = pd.DataFrame(data)

# Downcast DataFrame to minimum viable schema.
df_downcast = pdc.downcast(df)

# Infer minimum schema from DataFrame.
schema = pdc.infer_schema(df)

# Coerce DataFrame to schema - required if converting float to Pandas Integer.
df_new = pdc.coerce_df(df)
```

## Additional Notes
Smaller types == smaller memory footprint.
```python
df.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 4 columns):
#  #   Column      Non-Null Count  Dtype  
# ---  ------      --------------  -----  
#  0   integers    100 non-null    float64
#  1   floats      100 non-null    float64
#  2   booleans    100 non-null    int64  
#  3   categories  100 non-null    object 
# dtypes: float64(2), int64(1), object(1)
# memory usage: 3.2+ KB

df_downcast.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 4 columns):
#  #   Column      Non-Null Count  Dtype   
# ---  ------      --------------  -----   
#  0   integers    100 non-null    uint8   
#  1   floats      100 non-null    float32 
#  2   booleans    100 non-null    bool    
#  3   categories  100 non-null    category
# dtypes: bool(1), category(1), float32(1), uint8(1)
# memory usage: 932.0 bytes
```

Numerical data types will be downcast if the resulting values are within tolerance of the original values.
For details on tolerance for numeric comparison, see the notes on [`np.allclose`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html).
```python
print(df.head())
#    integers  floats  booleans categories
# 0       1.0    1.00         1        foo
# 1       2.0   11.09         0        baz
# 2       3.0   21.18         1        bar
# 3       4.0   31.27         0        bar
# 4       5.0   41.36         0        foo

print(df_downcast.head())
#    integers     floats  booleans categories
# 0         1   1.000000      True        foo
# 1         2  11.090000     False        baz
# 2         3  21.180000      True        bar
# 3         4  31.270000     False        bar
# 4         5  41.360001     False        foo


print(pdc.options.ATOL)
# >>> 1e-08

print(pdc.options.RTOL)
# >>> 1e-05
```
Tolerance can be set at module level or passed in function arguments:
```python
pdc.options.ATOL = 1e-10
pdc.options.RTOL = 1e-10
df_downcast_new = pdc.downcast(df)
```
Or
```python
infer_dtype_kws = {
    "ATOL": 1e-10,
    "RTOL": 1e-10
}
df_downcast_new = pdc.downcast(df, infer_dtype_kws=infer_dtype_kws)
```
The `floats` column is now kept as `float64` to meet the tolerance requirement. 
Values in the `integers` column are still safely cast to `uint8`.
```python
df_downcast_new.info()
# <class 'pandas.core.frame.DataFrame'>
# RangeIndex: 100 entries, 0 to 99
# Data columns (total 4 columns):
#  #   Column      Non-Null Count  Dtype   
# ---  ------      --------------  -----   
#  0   integers    100 non-null    uint8   
#  1   floats      100 non-null    float64 
#  2   booleans    100 non-null    bool    
#  3   categories  100 non-null    category
# dtypes: bool(1), category(1), float64(1), uint8(1)
# memory usage: 1.3 KB
```

## Example
The following example shows how downcasting data often leads to size reductions of **greater than 70%**, depending on the original types.

```python
import pdcast as pdc
import seaborn as sns

df_dict = {df: sns.load_dataset(df) for df in sns.get_dataset_names()}

results = []

for name, df in df_dict.items():
    mem_usage_pre = df.memory_usage(deep=True).sum()
    df_post = pdc.downcast(df)
    mem_usage_post = df_post.memory_usage(deep=True).sum()
    shrinkage = int((1 - (mem_usage_post / mem_usage_pre)) * 100)
    results.append({"dataset": name, "size_pre": mem_usage_pre, "size_post": mem_usage_post, "shrink_pct": shrinkage})

results_df = pd.DataFrame(results).sort_values("shrink_pct", ascending=False).reset_index()
print(results_df)
```
```
           dataset  size_pre  size_post  shrink_pct
0             fmri    213232      14776          93
1          titanic    321240      28162          91
2        attention      5888        696          88
3         penguins     75711       9131          87
4             dots    122240      17488          85
5           geyser     21172       3051          85
6           gammas    500128     108386          78
7         anagrams      2048        456          77
8          planets    112663      30168          73
9         anscombe      3428        964          71
10            iris     14728       5354          63
11        exercise      3302       1412          57
12         flights      3616       1888          47
13             mpg     75756      43842          42
14            tips      7969       6261          21
15        diamonds   3184588    2860948          10
16  brain_networks   4330642    4330642           0
17     car_crashes      5993       5993           0
```