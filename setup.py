#!/usb/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("B1python.c_asamacros", ["src/c_asamacros.pyx"]),
               Extension("B1python.c_asaxseval",["src/c_asaxseval.pyx"]),
               Extension("B1python.c_B1io",["src/c_B1io.pyx"]),
               Extension("B1python.c_B1macros",["src/c_B1macros.pyx"]),
               Extension("B1python.c_guitools",["src/c_guitools.pyx"]),
               Extension("B1python.c_fitting",["src/c_fitting.pyx"]),
               Extension("B1python.c_utils",["src/c_utils.pyx"]),
               Extension("B1python.c_utils2d",["src/c_utils2d.pyx"]),
               Extension("B1python.c_xanes",["src/c_xanes.pyx"]),
               Extension("B1python.c_unstable",["src/c_unstable.pyx"]),
               Extension("B1python.distdist",["src/distdist.pyx"]),
               ]

setup(name='B1python',version='0.2', author='Andras Wacha',
      author_email='awacha@gmail.com',url='http://github.com/awacha/B1python',
      description='Python macros for (A)SAXS evaluation',
      packages=['B1python'],
      package_dir={'B1python': 'src'},
      package_data={'B1python': ['calibrationfiles/*']},
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules
      )
      