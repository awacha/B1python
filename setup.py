#!/usb/bin/env python

from distutils.core import setup

setup(name='B1python',version='0.1', author='Andras Wacha',
      description='Python macros for (A)SAXS evaluation',
      packages=['B1python'],
      package_dir={'B1python': 'src'},
      package_data={'B1python': ['calibrationfiles/*']},
      )
      