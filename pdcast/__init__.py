__author__ = """Dominic Thorn"""
__email__ = "dominic.thorn@gmail.com"
__version__ = "0.1.0.dev1"

from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path
from typing import Set as _Set

from pdcast.core import downcast, infer_schema

_MODULE_PATH: _Path = _Path(__file__).parent.absolute()
