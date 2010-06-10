import numpy as np
cimport numpy as np
from stdlib cimport malloc, free

cdef extern from "math.h":
    double sqrt(double)
    double floor(double)
    
cdef double *ctrapezoidshapefunction(double lengthbase,double lengthtop, double *x, Py_ssize_t lenx):
    cdef double * T
    cdef Py_ssize_t i
    cdef double lb2mlt2
    if x==NULL:
        return NULL
    lb2mlt2=lengthbase**2-lengthtop**2
    if lenx<2:
        lenx=1
    T=<double *>malloc(lenx*sizeof(double))
    if lenx==1:
        T[0]=1
        return T
    for i from 0<=i<lenx:
        if x[i]<=-lengthtop/2.0:
            T[i]=(4.0*x[i]+lengthbase*2.0)/lb2mlt2
        elif x[i]<lengthtop/2.0:
            T[i]=2.0/(lengthbase+lengthtop)
        else:
            T[i]=(-4.0*x[i]+lengthbase*2.0)/lb2mlt2
    for i from 0<=i<lenx:
        if (x[i]<=-lengthbase/2) or (x[i]>=lengthbase/2):
            T[i]=0
    return T

cdef inline double mymax(double a, double b):
    if a>b:
        return a
    else:
        return b 

cdef inline double mymin(double a, double b):
    if a>b:
        return b
    else:
        return a 
    
    
def smearingmatrix(Py_ssize_t pixelmin, Py_ssize_t pixelmax, double beamcenter,
                    double pixelsize, double lengthbaseh, double lengthtoph,
                    double lengthbasev=0, double lengthtopv=0,
                    Py_ssize_t beamnumh=1024,Py_ssize_t beamnumv=1):
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
    cdef Py_ssize_t npixels
    cdef double *x
    cdef double * yb
    cdef double * xb    
    cdef double deltah,deltav,centerh,centerv
    cdef double *H
    cdef double *V
    cdef double center
    cdef Py_ssize_t i,j,ix,iy
    cdef np.ndarray[np.double_t,ndim=2] A
    cdef double tmp
    cdef long ind1
    cdef double prop
    cdef double p
    cdef double left
    # coordinates of the pixels
    npixels=pixelmax-pixelmin+1
    x=<double*>malloc(npixels*sizeof(double))
    for i from 0<=i<npixels:
        x[i]=pixelmin+i
    # horizontal and vertical coordinates of the beam-profile in mm.
    if beamnumv<1:
        beamnumv=1
    if beamnumh<1:
        beamnumh=1
    yb=<double *>malloc(beamnumh*sizeof(double))
    yb[0]=0
    xb=<double *>malloc(beamnumv*sizeof(double))
    xb[0]=0
    
    if beamnumh>1:
        left=-mymax(lengthbaseh,lengthtoph)/2.0
        deltah=-2*left/(beamnumh-1)
        for iy from 0<=iy<beamnumh:
            yb[iy]=left+iy*deltah
        centerh=2.0/(lengthbaseh+lengthtoph)
    else:
        deltah=1
        centerh=1
    if beamnumv>1:
        left=-mymax(lengthbasev,lengthtopv)/2.0
        deltav=-2*left/(beamnumv-1)
        for ix from 0<=ix<beamnumv:
            xb[ix]=left+ix*deltav
        centerv=2.0/(lengthbasev+lengthtopv)
    else:
        deltav=1
        centerv=1
    #beam profile vector (trapezoid centered at the origin. Only a half of it
    # is taken into account)
    H=ctrapezoidshapefunction(lengthbaseh,lengthtoph,yb,beamnumh)
    V=ctrapezoidshapefunction(lengthbasev,lengthtopv,xb,beamnumv)
    center=centerh*centerv
    # scale y to detector pixel units
    A=np.zeros((npixels,npixels),dtype=np.double)
    for i from 0<=i<npixels:
        A[i,i]+=center
        for ix from 0<=ix<beamnumv:
            for iy from 0<=iy<beamnumh:
                tmp=sqrt((<double>i-xb[ix]/pixelsize*1e3)*(<double>i-xb[ix]/pixelsize*1e3)+(yb[iy]/pixelsize*1e3)*(yb[iy]/pixelsize*1e3))
                ind1=<long>floor(tmp)
                prop=tmp-ind1
                if ind1>=npixels:
                    continue
                p=H[iy]*V[ix]
                A[i,ind1]+=p*(1-prop)
                if ind1<npixels-1:
                    A[i,ind1+1]+=p*prop
    A=A*deltah*deltav
    free(x)
    free(xb)
    free(yb)
    free(H)
    free(V)
    return A
    

def trapezoidshapefunction(lengthbase,lengthtop,x):
    x=np.array(x)
    if len(x)<2:
        return np.array(1)
    T=np.zeros(x.shape)
    indofflimits=(x<=-lengthbase/2.0)|(x>=lengthbase/2.0)
    indslopeleft=(x<=-lengthtop/2.0)
    indsloperight=(x>=lengthtop/2.0)
    indtop=(x<=lengthtop/2.0)&(x>=-lengthtop/2.0)
    T[indsloperight]=-4.0/(lengthbase**2-lengthtop**2)*x[indsloperight]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
    T[indtop]=2.0/(lengthbase+lengthtop)
    T[indslopeleft]=4.0/(lengthbase**2-lengthtop**2)*x[indslopeleft]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
    T[indofflimits]=0
    return T
   
#----------------RETIRED MACROS----------------------

def smearingmatrix1(pixelmin,pixelmax,beamcenter,pixelsize,lengthbaseh,
                   lengthtoph,lengthbasev=0,lengthtopv=0,beamnumh=1024,
                   beamnumv=1):
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
    # coordinates of the pixels
    pixels=np.arange(pixelmin,pixelmax+1)
    # distance of each pixel from the beam in pixel units
    x=np.absolute(pixels-beamcenter);
    # horizontal and vertical coordinates of the beam-profile in mm.
    if beamnumh>1:
        yb=np.linspace(-max(lengthbaseh,lengthtoph)/2.0,max(lengthbaseh,lengthtoph)/2.0,beamnumh)
        deltah=(yb[-1]-yb[0])*1.0/beamnumh
        centerh=2.0/(lengthbaseh+lengthtoph)
    else:
        yb=np.array([0])
        deltah=1
        centerh=1
    if beamnumv>1:
        xb=np.linspace(-max(lengthbasev,lengthtopv)/2.0,max(lengthbasev,lengthtopv)/2.0,beamnumv)
        deltav=(xb[-1]-xb[0])*1.0/beamnumv
        centerv=2.0/(lengthbasev+lengthtopv)
    else:
        xb=np.array([0])
        deltav=1
        centerv=1
    Xb,Yb=np.meshgrid(xb,yb)
    #beam profile vector (trapezoid centered at the origin. Only a half of it
    # is taken into account)
    H=trapezoidshapefunction(lengthbaseh,lengthtoph,yb)
    V=trapezoidshapefunction(lengthbasev,lengthtopv,xb)
    P=np.kron(H,V)
    center=centerh*centerv
    # scale y to detector pixel units
    Yb=Yb/pixelsize*1e3
    Xb=Xb/pixelsize*1e3
    A=np.zeros((len(x),len(x)))
    for i in range(len(x)):
        A[i,i]+=center
        tmp=np.sqrt((i-Xb)**2+Yb**2)
        ind1=np.floor(tmp).astype('int').flatten()
        prop=tmp.flatten()-ind1
        indices=(ind1<len(pixels))        
        ind1=ind1[indices].flatten()
        prop=prop[indices].flatten()
        p=P[indices].flatten()
        for j in range(len(ind1)):
            A[i,ind1[j]]+=p[j]*(1-prop[j])
            if ind1[j]<len(pixels)-1:
                A[i,ind1[j]+1]+=p[j]*prop[j]
    A=A*deltah*deltav
#    pylab.imshow(A)
#    pylab.colorbar()
#    pylab.gcf().show()
    return A
