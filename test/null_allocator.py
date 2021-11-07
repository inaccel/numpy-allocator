from ctypes import *
import numpy_allocator


class null_allocator(metaclass=numpy_allocator.type):
    @CFUNCTYPE(c_void_p, c_size_t, c_size_t)
    def _calloc_(nelem, elsize):
        return None

    @CFUNCTYPE(None, c_void_p, c_size_t)
    def _free_(ptr, size):
        pass

    @CFUNCTYPE(c_void_p, c_size_t)
    def _malloc_(size):
        return None

    @CFUNCTYPE(c_void_p, c_void_p, c_size_t)
    def _realloc_(ptr, new_size):
        return None


def main():
    import numpy as np

    with np.testing.assert_raises(MemoryError):
        with null_allocator:
            np.ndarray(())


if __name__ == '__main__':
    main()
