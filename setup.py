from setuptools import Extension, setup

import numpy

setup(
    name='numpy-allocator',
    version='1.0.0',
    description='numpy allocator',
    author='InAccel',
    author_email='info@inaccel.com',
    url='https://github.com/inaccel/numpy-allocator',
    ext_modules=[
        Extension(
            name='numpy_allocator',
            sources=[
                'numpy_allocator.c',
            ],
            include_dirs=[
                numpy.get_include(),
            ],
        ),
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    license='Apache-2.0',
    install_requires=[
        'numpy',
    ],
    python_requires='>=3.7',
)
