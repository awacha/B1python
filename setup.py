#!/usb/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("B1python.cythonized", ["src/cythonized.pyx"])]

setup(name='B1python',version='0.1', author='Andras Wacha',
      description='Python macros for (A)SAXS evaluation',
      packages=['B1python'],
      package_dir={'B1python': 'src'},
      package_data={'B1python': ['calibrationfiles/*']},
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules
      )
      