from ctypes import *
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


class aligned_allocator(numpy_allocator.object):

    def __init__(self, alignment):
        self.alignment = alignment

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, self.alignment)

    def _calloc_(self, nelem, elsize):
        result = std.memalign(self.alignment, nelem * elsize)
        if result:
            result = std.memset(result, 0, nelem * elsize)
        return result

    def _malloc_(self, size):
        return std.memalign(self.alignment, size)

    def _realloc_(self, ptr, new_size):
        result = std.realloc(ptr, new_size)
        if result and result % self.alignment != 0:
            tmp = result
            result = std.memalign(self.alignment, new_size)
            if result:
                result = std.memcpy(result, tmp, new_size)
            std.free(tmp)
        return result


def main():
    from mmap import PAGESIZE
    import numpy as np

    with aligned_allocator(PAGESIZE) as page_aligned_allocator:
        print(page_aligned_allocator)

        np.core.test()


if __name__ == '__main__':
    main()
