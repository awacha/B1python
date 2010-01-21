import numpy as np
cimport numpy as np

ctypedef np.double_t DOUBLE_T

def trapezoidshapefunction(double lengthbase, double lengthtop, np.ndarray[DOUBLE_T, ndim=1] x not None):
    cdef np.ndarray[np.double_t, ndim=1] T
    if len(x)<2:
        T=np.array([1],dtype=np.double)
        return T
    cdef int xlength=x.size
    cdef unsigned long i    
    T=np.zeros(xlength,dtype=np.double)

    for i from 0 <= i <xlength:
        if x[i]<=-lengthtop/2.0:
            T[i]=4.0/(lengthbase**2-lengthtop**2)*x[i]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
        elif x[i]<=lengthtop/2.0:
            T[i]=2.0/(lengthbase+lengthtop)
        else:
            T[i]=-4.0/(lengthbase**2-lengthtop**2)*x[i]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
#    indslopeleft=(x<=-lengthtop/2.0)
#    indsloperight=(x>=lengthtop/2.0)
#    indtop=(x<=lengthtop/2.0)&(x>=-lengthtop/2.0)
#    T[indsloperight]=-4.0/(lengthbase**2-lengthtop**2)*x[indsloperight]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
#    T[indtop]=2.0/(lengthbase+lengthtop)
#    T[indslopeleft]=4.0/(lengthbase**2-lengthtop**2)*x[indslopeleft]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
    return T

def smearingmatrix(int pixelmin, int pixelmax, double beamcenter, double pixelsize,
                   double lengthbaseh, double lengthtoph, double lengthbasev=0,
                   double lengthtopv=0, int beamnumh=1024, int beamnumv=1):
    """Calculate the smearing matrix for the given geometry.
    
    Inputs: (pixels and pixel coordinates are counted from 0. The direction
        of the detector is assumed to be vertical.)
        pixelmin: the smallest pixel to take into account
        pixelmax: the largest pixel to take into account
        beamcenter: pixel coordinate of the primary beam
        pixelsize: the size of pixels, in micrometers
        lengthbaseh: the length of the base of the horizontal beam profile
        lengthtoph: the length of the top of the horizontal beam profile
        lengthbasev: the length of the base of the vertical beam profile
        lengthtopv: the length of the top of the vertical beam profile
        beamnumh: the number of elementary points of the horizontal beam
            profile
        beamnumv: the number of elementary points of the vertical beam profile
            Give 1 if you only want to take the length of the slit into
            account.
    
    Output:
        The smearing matrix. This is an upper triangular matrix. Desmearing
        of a column vector of the measured intensities can be accomplished by
        multiplying by the inverse of this matrix.
    """
    pixelsize=pixelsize/1e3
    # coordinates of the pixels
    cdef np.ndarray pixels=np.arange(pixelmin,pixelmax+1,dtype=np.double)
    
    # distance of each pixel from the beam in pixel units
    cdef np.ndarray[np.double_t, ndim=1] x
    cdef np.ndarray[np.double_t, ndim=1] yb
    cdef np.ndarray[np.double_t, ndim=1] xb
    cdef double deltah
    cdef double centerh
    cdef double deltav
    cdef double centerv
    cdef np.ndarray[np.double_t, ndim=1] H
    cdef np.ndarray[np.double_t, ndim=1] V
    cdef double P
    cdef double center
    cdef np.ndarray[np.double_t, ndim=2] A
    cdef double tmp
    cdef int ind1
    cdef double prop
    cdef unsigned long i
    cdef unsigned long j
    cdef unsigned long ix, iy
    cdef unsigned long lenpixels
    
    lenpixels=len(pixels)
    x=np.absolute(pixels-beamcenter);
    # horizontal and vertical coordinates of the beam-profile in mm.
    if beamnumh>1:
        yb=np.linspace(-max(lengthbaseh/pixelsize,lengthtoph/pixelsize)/2.0,max(lengthbaseh,lengthtoph)/2.0,beamnumh)
        deltah=(yb[-1]-yb[0])*1.0/beamnumh
        centerh=2.0/(lengthbaseh+lengthtoph)
    else:
        beamnumh=1
        yb=np.array([0],dtype=np.double)
        deltah=1
        centerh=1
    if beamnumv>1:
        xb=np.linspace(-max(lengthbasev,lengthtopv)/2.0,max(lengthbasev,lengthtopv)/2.0,beamnumv)
        deltav=(xb[-1]-xb[0])*1.0/beamnumv
        centerv=2.0/(lengthbasev+lengthtopv)
    else:
        beamnumv=1
        xb=np.array([0],dtype=np.double)
        deltav=1
        centerv=1
    #beam profile vector (trapezoid centered at the origin. Only a half of it
    # is taken into account)
    H=trapezoidshapefunction(lengthbaseh,lengthtoph,yb)
    V=trapezoidshapefunction(lengthbasev,lengthtopv,xb)
    center=centerh*centerv
    # scale y to detector pixel units
    A=np.zeros((len(x),len(x)),dtype=np.double)
    for i from 0 <=i< lenpixels:
        A[i,i]+=center
        for ix from 0 <=ix<beamnumv:
            for iy from 0<=iy<beamnumh:
                P=H[iy]*V[ix]
                tmp=np.sqrt((i-xb[ix])**2+yb[iy]**2)
                ind1=np.floor(tmp)
                prop=tmp-ind1
                if ind1>=lenpixels:
                    continue
                A[i,ind1]+=P*(1-prop)
                if ind1<lenpixels-1:
                    A[i,ind1+1]+=P*prop
    A=A*deltah*deltav
    return A
