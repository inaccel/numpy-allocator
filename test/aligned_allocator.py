from ctypes import *
from mmap import PAGESIZE
from numpy_allocator import base_allocator

std = CDLL(None)

std.memalign.argtypes = [c_size_t, c_size_t]
std.memalign.restype = c_void_p

std.memcpy.argtypes = [c_void_p, c_void_p, c_size_t]
std.memcpy.restype = c_void_p

std.memset.argtypes = [c_size_t, c_size_t]
std.memset.restype = c_void_p

std.realloc.argtypes = [c_void_p, c_size_t]
std.realloc.restype = c_void_p


@CFUNCTYPE(c_void_p, c_size_t)
def aligned_alloc(size):
    return std.memalign(PAGESIZE, size)


@CFUNCTYPE(c_void_p, c_void_p, c_size_t)
def aligned_realloc(ptr, size):
    result = std.realloc(ptr, size)
    if result % PAGESIZE != 0:
        tmp = result
        result = std.memalign(PAGESIZE, size)
        result = std.memcpy(result, tmp, size)
        std.free(tmp)
    return result


@CFUNCTYPE(c_void_p, c_size_t, c_size_t)
def aligned_zeroed_alloc(nelems, elsize):
    result = std.memalign(PAGESIZE, nelems * elsize)
    result = std.memset(result, 0, nelems * elsize)
    return result


class aligned_allocator(metaclass=base_allocator):

    _alloc_ = aligned_alloc

    _realloc_ = aligned_realloc

    _zeroed_alloc_ = aligned_zeroed_alloc


def main():
    import numpy as np

    with aligned_allocator:
        np.core.test()


if __name__ == '__main__':
    main()
