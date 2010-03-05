import numpy as np
cimport numpy as np
import scipy.signal
from stdlib cimport malloc,free

cdef extern from "math.h":
    double sqrt(double)
    double M_PI
    double sin(double)
    double atan(double)

cdef extern from "stdlib.h":
    void *calloc(Py_ssize_t,Py_ssize_t)

def indirectdesmear(np.ndarray[np.double_t, ndim=1] m not None,
                    np.ndarray[np.double_t, ndim=1] I not None,
                    np.ndarray[np.double_t, ndim=1] px not None,
                    np.ndarray[np.double_t, ndim=1] P not None,
                    np.ndarray[np.double_t, ndim=1] qx not None,
                    np.ndarray[np.double_t, ndim=1] Q not None,
                    np.ndarray[np.double_t, ndim=1] wx not None,
                    np.ndarray[np.double_t, ndim=1] W not None,
                    double hperl,
                    double rmax,
                    Py_ssize_t Nr,
                    Py_ssize_t Nsplines,
                    Py_ssize_t Bsplineorder=3):
    """Indirect desmear according to Glatter

    Inputs:
        m: abscissae for I (pixels)
        I: counts in each q-bin
        px: abscissae for P (expressed in detector pixel units!)
        P: beam length profile
        qx: abscissae for Q (expressed in detector pixel units!)
        Q: beam height profile
        wx: abscissae for W (wavelength units)
        W: wavelength spread function
        hperl: pixel size divided by the sample-detector distance
        rmax: estimated highest distance
        Nr: number of r points for the support of each spline (only for
            the positive points)
        Nsplines: number of splines to be used
        Bsplineorder: the order of B-splines (default: 3)
        
    """
    cdef np.ndarray[np.double_t, ndim=1] spl
    cdef double *spl1
    cdef double * splr
    cdef np.ndarray[np.double_t, ndim=1] r
    cdef Py_ssize_t i,j,k,l,n,o, lenP,lenQ,lenW,lenm
    cdef double factor
    cdef double qr
    cdef double *m1, *px1, *P1, *qx1, *Q1,*wx1,*W1
    cdef double *smearsplines1

    # check lengths
    lenP=len(P)
    if len(px)!=len(P):
        raise ValueError('px and P should be of the same shape')
    lenQ=len(Q)
    if len(qx)!=len(Q):
        raise ValueError('qx and Q should be of the same shape')
    lenW=len(W)
    if len(wx)!=len(W):
        raise ValueError('wx and W should be of the same shape')
    lenm=len(m)
    if len(m)!=len(I):
        raise ValueError('m and I should be of the same shape')
        
    #allocate memory for C-style copies of numpy arrays
    m1=<double*>malloc(lenm*sizeof(double))
    px1=<double*>malloc(lenP*sizeof(double))
    P1=<double*>malloc(lenP*sizeof(double))
    qx1=<double*>malloc(lenQ*sizeof(double))
    Q1=<double*>malloc(lenQ*sizeof(double))
    wx1=<double*>malloc(lenW*sizeof(double))
    W1=<double*>malloc(lenW*sizeof(double))    

    # copy numpy arrays to c-style
    for i from 0<=i<lenm:
        m1[i]=m[i]
    for i from 0<=i<lenP:
        px1[i]=px[i]
        P1[i]=P[i]
    for i from 0<=i<lenQ:
        qx1[i]=qx[i]
        Q1[i]=Q[i]
    for i from 0<=i<lenW:
        wx1[i]=wx[i]
        W1[i]=W[i]

    #the abscissae of the splines            
    splr=<double*>malloc(sizeof(double)*Nr)
    #the length of the support of the B-spline
    factor=(Bsplineorder+1)
    #set the abscissae from -factor/2 to factor/2
    for i from 0<=i<Nr:
        splr[i]=-factor/2 + factor*i/(Nr-1)
    #the same as above, as a numpy array
    r=np.linspace(-Bsplineorder*0.5-0.5,Bsplineorder*0.5+0.5,Nr)
    # calculate the B-spline  function
    spl=scipy.signal.bspline(r,Bsplineorder)
    # create a C copy of it
    spl1=<double*>malloc(sizeof(double)*len(spl))
    for i from 0<=i<len(spl):
        spl1[i]=spl[i]
    #this will contain the smeared values
    smearsplines=np.zeros((lenm,Nsplines))
    # the C version of this
    smearsplines1=<double*>malloc(lenm*Nsplines*sizeof(double))
    for i from 0<=i<lenm*Nsplines:
        smearsplines1[i]=0
        
    for j from 0<=j<lenP: # for each point in the beam length profile
        print "j:",j
        for k from 0<=k<lenQ: # for each point in the beam width profile
            for l from 0<=l<lenW: # for each point in the multiwavelength profile
                for n from 0<=n<lenm: # for each detector point
                    # calculate the q-value corresponding to this point
                    qprime=4*M_PI*sin(0.5*atan(sqrt((m1[n]-qx1[k])**2+px1[j]**2)*hperl))/wx1[l]
                    #qprime=4*M_PI*0.5*sqrt((m1[n]-qx1[k])**2+px1[j]**2)*hperl/wx1[l]
                    for o from 0<=o<Nr: # for each point of the spline function
                        for i from 0<=i<Nsplines: # for each spline
                            qr=qprime*(splr[o]+i*rmax/(Nsplines-1))
                            if (qr!=0):
                                factor=sin(qr)/qr
                            else:
                                factor=1
                            smearsplines1[n+i*Nsplines]+=Q1[k]*P1[j]*W1[l]*spl[o]*factor
    for n from 0<=n<lenm:
        for i from 0<=i<Nsplines:
            smearsplines[n,i]=smearsplines1[n+i*Nsplines]
    free(m1)
    free(P1)
    free(px1)
    free(qx1)
    free(Q1)
    free(wx1)
    free(W1)
    free(splr)
    free(smearsplines1)
    return smearsplines,spl
