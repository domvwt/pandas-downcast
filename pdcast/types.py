import numpy as np
import pandas as pd

ALL_NAN_TYPE = pd.Int8Dtype()

BOOLEAN_TYPES = [np.bool_, pd.Int8Dtype()]

UINT_TYPES = [np.uint8, np.uint16, np.uint32, np.uint64]

UINT_NULLABLE_TYPES = [
    pd.UInt8Dtype(),
    pd.UInt16Dtype(),
    pd.UInt32Dtype(),
    pd.UInt64Dtype(),
]

INT_TYPES = [np.int8, np.int16, np.int32, np.int64]

INT_NULLABLE_TYPES = [
    pd.Int8Dtype(),
    pd.Int16Dtype(),
    pd.Int32Dtype(),
    pd.Int64Dtype(),
]

FLOAT_TYPES = [np.float16, np.float32, np.float64, np.float128]
