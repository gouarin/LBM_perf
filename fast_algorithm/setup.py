#!/usr/bin/env python
from distutils.core import setup, Extension
from Cython.Distutils import build_ext

d2q9 = Extension(name = "d2q9_nsnxny_cython",
                     sources=["d2q9_nsnxny_cython.pyx"],
                     extra_compile_args = ['-O3', '-fopenmp'],
                     extra_link_args= ['-fopenmp'])

d2q9_v2 = Extension(name = "d2q9_nxnyns_cython",
                     sources=["d2q9_nxnyns_cython.pyx"],
                     extra_compile_args = ['-O3', '-fopenmp'],
                     extra_link_args= ['-fopenmp'])

setup(ext_modules = [d2q9, d2q9_v2],
      cmdclass = {'build_ext': build_ext}
  )
