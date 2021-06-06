__author__ = """Dominic Thorn"""
__email__ = "dominic.thorn@gmail.com"
__version__ = "0.1.0.dev1"

from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path
from typing import Set as _Set

from pdcast.core import minimum_viable_schema, smallest_viable_type

_MODULE_PATH: _Path = _Path(__file__).parent.absolute()


_OPTIONAL_DEPENDENCIES: _Set[str] = {"sklearn"}

_INSTALLED_MODULES: _Set[str] = {
    x.name for x in [_find_spec(dep) for dep in _OPTIONAL_DEPENDENCIES] if x
}


if "sklearn" in _INSTALLED_MODULES:
    from pdcast.transformer import PandasDowncaster
