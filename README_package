"""B1python: various Pythonic functions for Small-Angle X-ray Scattering
analysis. Created by Andras Wacha for beamline B1 @HASYLAB/DESY, Hamburg,
Germany, but hopefully other scientists can benefit from these too. These
functions are partially based on Ulla Vainio's Matlab(R) scripts for B1 data
analysis.

Legal note: I donate these scripts to the public. You may freely use,
    distribute, modify the sources, as long as the legal notices remain in
    place, and no functionality is removed. However, using it as a main part
    for commercial (ie. non-free) software is not allowed (you got it free,
    you should give it free). The author(s) take no warranty on this program,
    nor on the outputs (like the copyleft from GNU). However, if you modify
    this, find a bug, suggest a new feature to be implemented, please feel
    free to contact the authors (Andras Wacha: awacha at gmail dot com).

Note for developers: If you plan to enhance this program, please be so kind to
    contact the original author (Andras Wacha: awacha at gmail dot com). I ask
    this because I am happy when I hear that somebody finds my work useful for
    his/her tasks. And for the coding style: please comment every change by
    your monogram/nickname and the date. And you should add your name,
    nickname and e-mail address to the authors clause in this notice as well.
    You deserve it.

General concepts:

    As it was already said, these functions are based on Matlab(R) scripts.
        It was kept iln mind therefore to retain Compatibility to the Matlab(R)
        version more or less. However, we are in Python and it would be
        foolish not to use the possibilities and tools it provides. In the
        next lines I note the differences.
    
    the numbering of the pixels of the detector starts from 1, in case of the
        beam coordinates, thus the values from the Matlab(R)-style
        intnorm*.log files are usable without modification. On contrary to
        Matlab(R), Python counts the indices from 0, so to get the real value
        of the beam coordinates, one should look at the pixel bcx-1,bcy-1. All
        functions in this file which need the beam coordinates as input or
        supply them as output, handle this automatically.
    
    the radially averaged datasets are in a data dictionary with fields 'q',
        'Intensity', 'Error' and possible 'Area'. The approach is similar to
        that of the Matlab(R) version, however in Python, dictionary is the
        best suited container.
        
    the mask matrices and corrected 2D data are saved to and loaded from mat
        files.
        
    if a function classifies measurements depending on energies, it uses
        always the uncalibrated (apparent) energies, which were set up in the
        measurement program at the beamline.
        
    the difference between header and param structures (dictionaries in
        Python) are planned to be completely removed. In the Matlab(R) version
        the fields of a header structure consist a subset of the ones of the
        param structure. When the data evaluation routines get implemented,
        the header dictionary extracted from the input datasets will get
        extended during the evaluation run by newly calculated values.

Dependencies:
    This set of functions depend---apart from the standard Python library---on
    various 3rd party modules. A complete list of these:
        matplotlib (pylab)
        scipy
    
A final note: functions labelled by EXPERIMENTAL!!!! in the online help-text
    are REALLY experimental. When I say experimental, I mean experimental,
    possibly not fully implemented code. No kidding. They are not thoroughly
    tested, so use on your own risk. They may not do what you expect, or they
    won't do anything at all. You have been warned. But if you are really
    curious, you can look at their source code... :-)
"""


# NOTE: never ever use the "from spam import *" formalism in this file! This can
# break things very much (eg. the sum() function from numpy overloads the
# built-in sum(). So be very cautious!



