from ctypes import *
from numpy_allocator import base_allocator

std = CDLL(None)

std.malloc.argtypes = [c_size_t]
std.malloc.restype = c_void_p

std.memcpy.argtypes = [c_void_p, c_void_p, c_size_t]
std.memcpy.restype = c_void_p

std.free.argtypes = [c_void_p]
std.free.restype = None

std.realloc.argtypes = [c_void_p, c_size_t]
std.realloc.restype = c_void_p

std.calloc.argtypes = [c_size_t, c_size_t]
std.calloc.restype = c_void_p


@CFUNCTYPE(c_void_p, c_size_t)
def PyDataMem_AllocFunc(size):
    result = std.malloc(size)
    print('%x malloc(%d)' % (result, size))
    return result


@CFUNCTYPE(c_void_p, c_void_p, c_void_p, c_size_t)
def PyDataMem_CopyFunc(dst, src, size):
    result = std.memcpy(dst, src, size)
    print('%x memcpy(%x, %x, %d)' % (result, dst, src, size))
    return result


@CFUNCTYPE(None, c_void_p, c_size_t)
def PyDataMem_FreeFunc(ptr, size):
    std.free(ptr)
    print('free(%x)' % ptr)


@CFUNCTYPE(c_void_p, c_void_p, c_size_t)
def PyDataMem_ReallocFunc(ptr, size):
    result = std.realloc(ptr, size)
    print('%x realloc(%x, %d)' % (result, ptr, size))
    return result


@CFUNCTYPE(c_void_p, c_size_t, c_size_t)
def PyDataMem_ZeroedAllocFunc(nelems, elsize):
    result = std.calloc(nelems, elsize)
    print('%x calloc(%d, %d)' % (result, nelems, elsize))
    return result


class debug_allocator(metaclass=base_allocator):

    _alloc_ = PyDataMem_AllocFunc

    _free_ = PyDataMem_FreeFunc

    # _host2obj_ = PyDataMem_CopyFunc

    # _obj2host_ = PyDataMem_CopyFunc

    # _obj2obj_ = PyDataMem_CopyFunc

    _realloc_ = PyDataMem_ReallocFunc

    _zeroed_alloc_ = PyDataMem_ZeroedAllocFunc


def main():
    import numpy as np

    with debug_allocator:
        a = np.arange(15).reshape((3, 5))

    assert debug_allocator.handles(a)

    b = np.array([6, 7, 8])

    assert not debug_allocator.handles(b)


if __name__ == '__main__':
    main()
