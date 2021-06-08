import numpy as np
import pandas as pd

boolean_mocks = [
    # Null
    (pd.Series([None] * 12), pd.Int8Dtype()),

    # Boolean
    (pd.Series(np.zeros(12)), np.bool_),
    (pd.Series(np.ones(12)), np.bool_),
    (pd.Series([1, 0] * 6), np.bool_),
    (pd.Series([1 + 1e-12, 1e-12] * 6), np.bool_),

    # Boolean - Nullable
    (pd.Series([1, 0, None] * 4), pd.Int8Dtype()),
    (pd.Series([1 + 1e-12, 1e-12, None] * 4), pd.Int8Dtype()),
]

integer_mocks = [
    # Integer - Unsigned
    (pd.Series([0, 10, 100, 255] * 3), np.uint8),
    (pd.Series([0, 10, 100, 65_000] * 3), np.uint16),
    (pd.Series([0, 10, 100, 4_294_967_295] * 3), np.uint32),
    (pd.Series([0, 10, 100, 18_446_744_073_709_551_615] * 3), np.uint64),
    (pd.Series([0, 10.00, 100.00, 255] * 3), np.uint8),
    (pd.Series([0, 10.00, 100.00, 65_000] * 3), np.uint16),
    (pd.Series([0, 10.00, 100.00, 4_294_967_295] * 3), np.uint32),
    # (pd.Series([0, 10.00, 100.00, 18_446_744_073_709_551_615] * 3), np.uint64),
    (pd.Series([0, 10.00, 100.00, 18_446_744_073_709_550_591] * 3), np.uint64),

    # Integer - Unsigned - Nullable
    (pd.Series([0, 10, None, 255] * 3), pd.UInt8Dtype()),
    (pd.Series([0, 10, None, 65_000] * 3), pd.UInt16Dtype()),
    (pd.Series([0, 10, None, 4_294_967_295] * 3), pd.UInt32Dtype()),
    # (pd.Series([0, 10, None, 18_446_744_073_709_551_615] * 3), pd.UInt64Dtype()),
    (pd.Series([0, 10, None, 18_446_744_073_709_550_000] * 3), pd.UInt64Dtype()),

    # Integer - Signed
    (pd.Series([-128, -10.00, 100.00, 127] * 3), np.int8),
    (pd.Series([-32_768, -10.00, 100.00, 32_767] * 3), np.int16),
    (pd.Series([-2_147_483_648, -10.00, 100.00, 2_147_483_647] * 3), np.int32),
    # (pd.Series([-9_223_372_036_854_775_808, -10.00, 100.00, 9_223_372_036_854_775_807] * 3), np.int64),
    (pd.Series([-9_223_372_036_854_775_000, -10.00, 100.00, 9_223_372_036_854_775_000] * 3), np.int64),

    # Integer - Signed - Nullable
    (pd.Series([-128, -10.00, None, 127] * 3), pd.Int8Dtype()),
    (pd.Series([-32_768, -10.00, None, 32_767] * 3), pd.Int16Dtype()),
    (pd.Series([-2_147_483_648, -10.00, None, 2_147_483_647] * 3), pd.Int32Dtype()),
    # (pd.Series([-9_223_372_036_854_775_808, -10.00, 100.00, 9_223_372_036_854_775_807] * 3), pd.Int64Dtype),
    (pd.Series([-9_223_372_036_854_775_000, -10.00, None, 9_223_372_036_854_775_000] * 3), pd.Int64Dtype()),
]

float_mocks = [
    # Float
    (pd.Series([-0.990234, 0.990234, 0, None]), np.float16),
    (pd.Series([-3.4028235e38, 3.4028235e38, 0, None]), np.float32),
    (pd.Series([-1.7976931348623157e308, 1.7976931348623157e308, 0, None]), np.float64),
]

# Categorical
