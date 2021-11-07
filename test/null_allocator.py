from ctypes import *
import numpy_allocator


@CFUNCTYPE(c_void_p, c_size_t, c_size_t)
def null_calloc(nelem, elsize):
    return None


@CFUNCTYPE(None, c_void_p, c_size_t)
def null_free(ptr, size):
    pass


@CFUNCTYPE(c_void_p, c_size_t)
def null_malloc(size):
    return None


@CFUNCTYPE(c_void_p, c_void_p, c_size_t)
def null_realloc(ptr, new_size):
    return None


class null_allocator(metaclass=numpy_allocator.type):

    _calloc_ = null_calloc

    _free_ = null_free

    _malloc_ = null_malloc

    _realloc_ = null_realloc


def main():
    import numpy as np

    with np.testing.assert_raises(MemoryError):
        with null_allocator:
            np.ndarray(())


if __name__ == '__main__':
    main()
