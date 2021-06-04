from pdcast import _INSTALLED_MODULES

if "sklearn" in _INSTALLED_MODULES:
    from typing import Any, Dict, Optional

    from pandas import DataFrame
    from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
    from sklearn.exceptions import NotFittedError  # type: ignore

    import pdcast

    class PandasDowncaster(BaseEstimator, TransformerMixin):
        """Apply minimum viable schema to Pandas DataFrame."""

        schema_: Optional[Dict[str, Any]] = None

        def fit(self, X: DataFrame, y=None, **fit_params) -> None:

            if not isinstance(X, DataFrame):
                raise TypeError(type(X))

            df = X.copy()
            self.schema_ = pdcast.minimum_viable_schema(df)

        def transform(self, X: DataFrame, y=None, **transform_params) -> DataFrame:

            if not isinstance(X, DataFrame):
                raise TypeError(type(X))

            if not hasattr(self, "schema_"):
                raise NotFittedError(str(self))

            df = X.copy()
            return df.astype(self.schema_)  # type: ignore

        def fit_transform(self, X: DataFrame, y=None, **fit_params) -> DataFrame:
            return super().fit_transform(X, y=y, **fit_params)
