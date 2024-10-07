## Memory management in [NumPy](https://numpy.org)*

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/inaccel/numpy-allocator/master?labpath=NumPy-Allocator.ipynb)
[![PyPI version](https://badge.fury.io/py/numpy-allocator.svg)](https://badge.fury.io/py/numpy-allocator)

**NumPy is a trademark owned by [NumFOCUS](https://numfocus.org).*

#### Customize Memory Allocators

Α metaclass is used to override the internal data memory routines. The metaclass has four optional fields:

```python
>>> import ctypes
>>> import ctypes.util
>>> import numpy_allocator
>>> my = ctypes.CDLL(ctypes.util.find_library('my'))
>>> class my_allocator(metaclass=numpy_allocator.type):
...     _calloc_ = ctypes.addressof(my.calloc_func)
...     _free_ = ctypes.addressof(my.free_func)
...     _malloc_ = ctypes.addressof(my.malloc_func)
...     _realloc_ = ctypes.addressof(my.realloc_func)
...
```

#### An example using the allocator

```python
>>> import numpy as np
>>> with my_allocator:
...     a = np.array([1, 2, 3])
...
>>> my_allocator.handles(a)
True
```
