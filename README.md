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
results_df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>dataset</th>
      <th>size_pre</th>
      <th>size_post</th>
      <th>shrink_pct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fmri</td>
      <td>213232</td>
      <td>14776</td>
      <td>93</td>
    </tr>
    <tr>
      <th>1</th>
      <td>titanic</td>
      <td>321240</td>
      <td>28162</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>attention</td>
      <td>5888</td>
      <td>696</td>
      <td>88</td>
    </tr>
    <tr>
      <th>3</th>
      <td>penguins</td>
      <td>75711</td>
      <td>9131</td>
      <td>87</td>
    </tr>
    <tr>
      <th>4</th>
      <td>dots</td>
      <td>122240</td>
      <td>17488</td>
      <td>85</td>
    </tr>
    <tr>
      <th>5</th>
      <td>geyser</td>
      <td>21172</td>
      <td>3051</td>
      <td>85</td>
    </tr>
    <tr>
      <th>6</th>
      <td>gammas</td>
      <td>500128</td>
      <td>108386</td>
      <td>78</td>
    </tr>
    <tr>
      <th>7</th>
      <td>anagrams</td>
      <td>2048</td>
      <td>456</td>
      <td>77</td>
    </tr>
    <tr>
      <th>8</th>
      <td>planets</td>
      <td>112663</td>
      <td>30168</td>
      <td>73</td>
    </tr>
    <tr>
      <th>9</th>
      <td>anscombe</td>
      <td>3428</td>
      <td>964</td>
      <td>71</td>
    </tr>
    <tr>
      <th>10</th>
      <td>iris</td>
      <td>14728</td>
      <td>5354</td>
      <td>63</td>
    </tr>
    <tr>
      <th>11</th>
      <td>exercise</td>
      <td>3302</td>
      <td>1412</td>
      <td>57</td>
    </tr>
    <tr>
      <th>12</th>
      <td>flights</td>
      <td>3616</td>
      <td>1888</td>
      <td>47</td>
    </tr>
    <tr>
      <th>13</th>
      <td>mpg</td>
      <td>75756</td>
      <td>43842</td>
      <td>42</td>
    </tr>
    <tr>
      <th>14</th>
      <td>tips</td>
      <td>7969</td>
      <td>6261</td>
      <td>21</td>
    </tr>
    <tr>
      <th>15</th>
      <td>diamonds</td>
      <td>3184588</td>
      <td>2860948</td>
      <td>10</td>
    </tr>
    <tr>
      <th>16</th>
      <td>brain_networks</td>
      <td>4330642</td>
      <td>4330642</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>car_crashes</td>
      <td>5993</td>
      <td>5993</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
