from ctypes import *
from mmap import PAGESIZE
from numpy_allocator import base_allocator

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


@CFUNCTYPE(c_void_p, c_size_t, c_size_t)
def aligned_calloc(nelem, elsize):
    result = std.memalign(PAGESIZE, nelem * elsize)
    result = std.memset(result, 0, nelem * elsize)
    return result


@CFUNCTYPE(c_void_p, c_size_t)
def aligned_malloc(size):
    return std.memalign(PAGESIZE, size)


@CFUNCTYPE(c_void_p, c_void_p, c_size_t)
def aligned_realloc(ptr, new_size):
    result = std.realloc(ptr, new_size)
    if result % PAGESIZE != 0:
        tmp = result
        result = std.memalign(PAGESIZE, new_size)
        result = std.memcpy(result, tmp, new_size)
        std.free(tmp)
    return result


class aligned_allocator(metaclass=base_allocator):

    _calloc_ = aligned_calloc

    _malloc_ = aligned_malloc

    _realloc_ = aligned_realloc


def main():
    import numpy as np

    with aligned_allocator:
        np.core.test()


if __name__ == '__main__':
    main()
