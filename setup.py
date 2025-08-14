
# setup.py
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "tsp",
        ["tsp.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native", "-ffast-math"],
        language="c",
    )
]

setup(
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
        'initializedcheck': False,
        'nonecheck': False,
    })
)