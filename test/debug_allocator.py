from ctypes import *
from numpy_allocator import base_allocator

std = CDLL(None)

std.calloc.argtypes = [c_size_t, c_size_t]
std.calloc.restype = c_void_p

std.free.argtypes = [c_void_p]
std.free.restype = None

std.malloc.argtypes = [c_size_t]
std.malloc.restype = c_void_p

std.realloc.argtypes = [c_void_p, c_size_t]
std.realloc.restype = c_void_p


@CFUNCTYPE(c_void_p, c_size_t, c_size_t)
def PyDataMem_CallocFunc(nelem, elsize):
    result = std.calloc(nelem, elsize)
    print('%x calloc(%d, %d)' % (result, nelem, elsize))
    return result


@CFUNCTYPE(None, c_void_p, c_size_t)
def PyDataMem_FreeFunc(ptr, size):
    std.free(ptr)
    print('free(%x)' % ptr)


@CFUNCTYPE(c_void_p, c_size_t)
def PyDataMem_MallocFunc(size):
    result = std.malloc(size)
    print('%x malloc(%d)' % (result, size))
    return result


@CFUNCTYPE(c_void_p, c_void_p, c_size_t)
def PyDataMem_ReallocFunc(ptr, new_size):
    result = std.realloc(ptr, new_size)
    print('%x realloc(%x, %d)' % (result, ptr, new_size))
    return result


class debug_allocator(metaclass=base_allocator):

    _calloc_ = PyDataMem_CallocFunc

    _free_ = PyDataMem_FreeFunc

    _malloc_ = PyDataMem_MallocFunc

    _realloc_ = PyDataMem_ReallocFunc


def main():
    import numpy as np

    with debug_allocator:
        a = np.arange(15).reshape((3, 5))

    assert debug_allocator.handles(a)

    b = np.array([6, 7, 8])

    assert not debug_allocator.handles(b)


if __name__ == '__main__':
    main()