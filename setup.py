#!/usb/bin/env python

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

VERSION='0.4.2'

print ""
print "+----------------------------------------%s------------+" % ('-'*len(VERSION))
print "| UPDATING VERSION IN src/__init__.py to %s !!!!!!!!!! |" % VERSION
print "+----------------------------------------%s------------+" % ('-'*len(VERSION))
print ""
f=open('src/__init__.py','rt')
lines=f.readlines()
f.close()
f1=open('src/__init__.py','w+t')
for l in lines:
    if l.strip().startswith('VERSION'):
        l='VERSION=%s\n' % VERSION
    f1.write(l)
f1.close() 

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
               Extension("B1python.c_saxssim",["src/c_saxssim.pyx"])
               ]

setup(name='B1python',version=VERSION, author='Andras Wacha',
      author_email='awacha@gmail.com',url='http://github.com/awacha/B1python',
      description='Python macros for (A)SAXS evaluation',
      packages=['B1python'],
      package_dir={'B1python': 'src'},
      package_data={'B1python': ['calibrationfiles/*']},
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules
      )
      