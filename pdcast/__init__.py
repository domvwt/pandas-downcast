# -*- coding: utf-8 -*-
"""Pandas Downcast
===============

Safely infer minimum viable schema for Pandas DataFrames and Series.

Examples:
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
    df_new = pdc.coerce_df(df, schema)

"""

__author__ = """Dominic Thorn"""
__email__ = "dominic.thorn@gmail.com"
__version__ = "1.2.1"

from pdcast.core import coerce_df, coerce_series, downcast, infer_schema, options
