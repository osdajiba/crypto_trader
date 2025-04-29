#!/usr/bin/env python3
# src/portfolio/assets/__init__.py

from .base import Asset
from .factory import AssetFactory
from .spot import Spot
from .future import Future

# Import other asset types if they've been implemented
try:
    from .bond import Bond
except ImportError:
    pass

try:
    from .fund import Fund
except ImportError:
    pass

try:
    from .option import Option
except ImportError:
    pass

__all__ = [
    'Asset',
    'AssetFactory',
    'Spot',
    'Future',
]

# Add other asset types to __all__ if they're imported
import sys
if 'Bond' in sys.modules.get(__name__, {}).__dict__:
    __all__.append('Bond')
if 'Fund' in sys.modules.get(__name__, {}).__dict__:
    __all__.append('Fund')
if 'Option' in sys.modules.get(__name__, {}).__dict__:
    __all__.append('Option')