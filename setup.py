#!/usb/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from distutils.sysconfig import get_python_lib, get_python_inc
import os

VERSION='0.8.2'

incdirs=list(set([get_python_lib(0,0),get_python_lib(0,1),get_python_lib(1,0),get_python_lib(1,1),get_python_inc(0),get_python_inc(1)]))

npy_incdirs=[os.path.join(x,'numpy/core/include') for x in incdirs]
incdirs.extend(npy_incdirs)

try:
    f=open('src/__init__.py','rt')
    lines=f.readlines()
    f.close()
    verline=[l for l in lines if l.strip().startswith('VERSION')][0]
    verline=verline.split('=')[1].strip()[1:-1]
    if verline==VERSION:
        raise RuntimeError # to quit this try block
    f1=open('src/__init__.py','w+t')
    for l in lines:
        if l.strip().startswith('VERSION'):
            l='VERSION="%s"\n' % VERSION
        f1.write(l)
    f1.close() 
    print ""
    print "+---------------------------------------%s------------+" % ('-'*len(VERSION))
    print "| UPDATED VERSION IN src/__init__.py to %s !!!!!!!!!! |" % VERSION
    print "+---------------------------------------%s------------+" % ('-'*len(VERSION))
    print ""
except IOError:
    print "Cannot update VERSION in src/__init__.py"
except RuntimeError:
    pass

ext_modules = [Extension("B1python.c_asamacros", ["src/c_asamacros.pyx"],include_dirs=incdirs),
               Extension("B1python.c_asaxseval",["src/c_asaxseval.pyx"],include_dirs=incdirs),
               Extension("B1python.c_B1io",["src/c_B1io.pyx"],include_dirs=incdirs),
               Extension("B1python.c_B1macros",["src/c_B1macros.pyx"],include_dirs=incdirs),
               Extension("B1python.c_guitools",["src/c_guitools.pyx"],include_dirs=incdirs),
               Extension("B1python.c_fitting",["src/c_fitting.pyx"],include_dirs=incdirs),
               Extension("B1python.c_utils",["src/c_utils.pyx"],include_dirs=incdirs),
               Extension("B1python.c_utils2d",["src/c_utils2d.pyx"],include_dirs=incdirs),
               Extension("B1python.c_xanes",["src/c_xanes.pyx"],include_dirs=incdirs),
               Extension("B1python.c_saxssim",["src/c_saxssim.pyx"],include_dirs=incdirs),
               Extension("B1python.c_unstable",["src/c_unstable.pyx"],include_dirs=incdirs),
               ]

setup(name='B1python',version=VERSION, author='Andras Wacha',
      author_email='awacha@gmail.com',url='http://github.com/awacha/B1python',
      description='Python macros for (A)SAXS evaluation',
      packages=['B1python'],
      py_modules=['B1python.asamacros','B1python.asaxseval','B1python.B1io',
                'B1python.B1macros','B1python.guitools','B1python.fitting','B1python.utils',
                'B1python.utils2d','B1python.xanes','B1python.unstable','B1python.saxssim'],
      package_dir={'B1python': 'src'},
      package_data={'B1python': ['calibrationfiles/*']},
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules,
      scripts = ['src/B1guitool.py']
      )
      