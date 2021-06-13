Pandas Downcast
===============

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
import pdcast as pc

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
df_downcast = pc.downcast(df)

# Infer minimum schema from DataFrame.
schema = pc.infer_schema(df)

# Coerce DataFrame to schema - required if converting float to Pandas Integer.
df_new = pc.coerce_df(df)
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


print(pc.options.ATOL)
# >>> 1e-08

print(pc.options.RTOL)
# >>> 1e-05
```
Tolerance can be set at module level or passed in function arguments:
```python
pc.options.ATOL = 1e-10
pc.options.RTOL = 1e-10
df_downcast_new = pc.downcast(df)
```
Or
```python
infer_dtype_kws = {
    "ATOL": 1e-10,
    "RTOL": 1e-10
}
df_downcast_new = pc.downcast(df, infer_dtype_kws=infer_dtype_kws)
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

