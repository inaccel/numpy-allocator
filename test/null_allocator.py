import numpy_allocator


class null_allocator(metaclass=numpy_allocator.type):

    def _calloc_(nelem, elsize):
        return None

    def _free_(ptr, size):
        pass

    def _malloc_(size):
        return None

    def _realloc_(ptr, new_size):
        return None


def main():
    import numpy as np

    with np.testing.assert_raises(MemoryError):
        with null_allocator:
            np.ndarray(())


if __name__ == '__main__':
    main()
