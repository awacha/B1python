#!/usb/bin/env python

from distutils.core import setup, Extension

radint_ng_module = Extension('_radint_ng',
                             sources=['radint_ng_wrap.c', 'radint_ng.c'],)
                             
setup(name='radint_ng',version='0.1', author='Andras Wacha',
      description='radint, programmed in C',
      ext_modules=[radint_ng_module],
      py_modules=['radint_ng']
      )