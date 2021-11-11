from ctypes import *
from mmap import PAGESIZE
import numpy_allocator

std = CDLL(None)

std.free.argtypes = [c_void_p]
std.free.restype = None

std.memalign.argtypes = [c_size_t, c_size_t]
std.memalign.restype = c_void_p

std.memcpy.argtypes = [c_void_p, c_void_p, c_size_t]
std.memcpy.restype = c_void_p

std.memset.argtypes = [c_void_p, c_int, c_size_t]
std.memset.restype = c_void_p

std.realloc.argtypes = [c_void_p, c_size_t]
std.realloc.restype = c_void_p


class page_aligned_allocator(metaclass=numpy_allocator.type):
    @CFUNCTYPE(c_void_p, c_size_t, c_size_t)
    def _calloc_(nelem, elsize):
        result = std.memalign(PAGESIZE, nelem * elsize)
        if result:
            result = std.memset(result, 0, nelem * elsize)
        return result

    @CFUNCTYPE(c_void_p, c_size_t)
    def _malloc_(size):
        return std.memalign(PAGESIZE, size)

    @CFUNCTYPE(c_void_p, c_void_p, c_size_t)
    def _realloc_(ptr, new_size):
        result = std.realloc(ptr, new_size)
        if result and result % PAGESIZE != 0:
            tmp = result
            result = std.memalign(PAGESIZE, new_size)
            if result:
                result = std.memcpy(result, tmp, new_size)
            std.free(tmp)
        return result


def main():
    import numpy as np

    with page_aligned_allocator:
        print(page_aligned_allocator)

        np.core.test()


if __name__ == '__main__':
    main()
