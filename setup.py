from setuptools import Extension, setup

import numpy


def README():
    with open('README.md') as md:
        return md.read()


setup(
    name='numpy-allocator',
    use_scm_version=True,
    description='Configurable memory allocations',
    long_description=README(),
    long_description_content_type='text/markdown',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    license='Apache-2.0',
    platforms=[
        'Linux',
    ],
    install_requires=[
        'numpy>=1.22.0',
    ],
    python_requires='>=3.8',
)
