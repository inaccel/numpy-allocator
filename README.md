## Memory management in [NumPy](https://numpy.org)*

[![Binder](https://mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/v2/gh/inaccel/numpy-allocator/master?labpath=NumPy-Allocator.ipynb)
[![PyPI version](https://badge.fury.io/py/numpy-allocator.svg)](https://badge.fury.io/py/numpy-allocator)

**NumPy is a trademark owned by [NumFOCUS](https://numfocus.org).*

#### Customize Memory Allocators

Α metaclass is used to override the internal data memory routines. The metaclass has four optional fields:

```python
>>> import ctypes
>>> import numpy_allocator
>>> my = ctypes.CDLL('libmy.so')
>>> class my_allocator(metaclass=numpy_allocator.type):
...     _calloc_ = my.calloc_func
...     _free_ = my.free_func
...     _malloc_ = my.malloc_func
...     _realloc_ = my.realloc_func
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
