import numpy_allocator


class default_allocator(metaclass=numpy_allocator.type):
    _handler_ = numpy_allocator.default_handler


def main():
    import numpy as np

    with default_allocator:
        print(np.core.multiarray.get_handler_name(),
              np.core.multiarray.get_handler_version())


if __name__ == '__main__':
    main()
