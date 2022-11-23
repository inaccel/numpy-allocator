from ctypes import CDLL, c_size_t, c_void_p
from ctypes.util import find_library
import numpy_allocator

std = CDLL(find_library('c'))

std.calloc.argtypes = [c_size_t, c_size_t]
std.calloc.restype = c_void_p

std.free.argtypes = [c_void_p]
std.free.restype = None

std.malloc.argtypes = [c_size_t]
std.malloc.restype = c_void_p

std.realloc.argtypes = [c_void_p, c_size_t]
std.realloc.restype = c_void_p


class debug_allocator(metaclass=numpy_allocator.type):

    def _calloc_(nelem, elsize):
        result = std.calloc(nelem, elsize)
        if result:
            print('0x%x calloc(%d, %d)' % (result, nelem, elsize))
        else:
            print('calloc(%d, %d)' % (nelem, elsize))
        return result

    def _free_(ptr, size):
        std.free(ptr)
        print('free(0x%x)' % ptr)

    def _malloc_(size):
        result = std.malloc(size)
        if result:
            print('0x%x malloc(%d)' % (result, size))
        else:
            print('malloc(%d)' % size)
        return result

    def _realloc_(ptr, new_size):
        result = std.realloc(ptr, new_size)
        if result:
            print('0x%x realloc(0x%x, %d)' % (result, ptr, new_size))
        else:
            print('realloc(0x%x, %d)' % (ptr, new_size))
        return result


def main():
    import numpy as np

    with debug_allocator:
        a = np.arange(15).reshape((3, 5))

    assert debug_allocator.handles(a)

    b = np.array([6, 7, 8])

    assert not debug_allocator.handles(b)


if __name__ == '__main__':
    main()
