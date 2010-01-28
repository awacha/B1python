import numpy as np
cimport numpy as np
from stdlib cimport *


cdef extern from "math.h":
    double cos(double)
    double sin(double)
    double sqrt(double)
    double atan(double)
    double M_PI
    double NAN

#cdef double M_PI=3.14159265358979323846
cdef double HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units
    
def test(np.ndarray[np.double_t, ndim=2] data):
    if data is None:
        print "test(): data is None!"
    else:
        print "test(): data is not None!"
    
def polartransform(np.ndarray[np.double_t, ndim=2] data not None, np.ndarray[np.double_t, ndim=1] r, np.ndarray[np.double_t, ndim=1] phi, double origx, double origy):
    """Calculates a matrix of a polar representation of the image.
    
    Inputs:
        data: the 2D matrix
        r: vector of polar radii
        phi: vector of polar angles
        origx: the x (row) coordinate of the origin
        origy: the y (column) coordinate of the origin
    Outputs:
        pdata: a matrix of len(phi) rows and len(r) columns which contains the
            polar representation of the image.
    """
    cdef np.ndarray[np.double_t,ndim=2] pdata
    cdef Py_ssize_t lenphi,lenr
    cdef Py_ssize_t i,j
    cdef double x,y
    cdef Nrows,Ncols
    
    Nrows=data.shape[0]
    Ncols=data.shape[1]
    lenphi=len(phi)
    lenr=len(r)
    pdata=np.zeros((lenphi,lenr))
    
    for i from 0<=i<lenphi:
        for j from 0<=j<lenr:
            x=origx-1+r[j]*cos(phi[i]);
            y=origy-1+r[j]*sin(phi[i]);
            if (x>=0) and (y>=0) and (x<Nrows) and (y<Ncols):
                pdata[i,j]=data[x,y];
    return pdata

def radintC(np.ndarray[np.double_t,ndim=2] data not None,
            np.ndarray[np.double_t,ndim=2] dataerr not None,
            double energy, double distance, res,
            double bcx, double bcy,
            np.ndarray[np.uint8_t, ndim=2] mask not None,
            np.ndarray[np.double_t, ndim=1] q=None,
            bint shutup=True):
    """Do radial integration on 2D scattering images
    
    Inputs:
        data: the intensity matrix
        dataerr: the error matrix (of the same size as data). Should not be
            zero!
        energy: the (calibrated) beam energy (eV)
        distance: the distance from the sample to the detector (mm)
        res: pixel size in mm-s. Both x and y (row and column) direction can
            be given if wished, in a list with two elements. A scalar value
            means that the pixel size is equal in both directions
        bcx: the coordinate of the beam center in the x (row) direction,
            starting from ZERO
        bcy: the coordinate of the beam center in the y (column) direction,
            starting from ZERO
        mask: the mask matrix (of the same size as data). Nonzero is masked,
            zero is not masked
        q: the q points at which the integration is requested. Note that 
            auto-guessing is not yet supported! It should be defined in 1/Angstroems.
        shutup: if True, work quietly (do not print messages).
        
    Outputs: four ndarrays.
        the q vector
        the intensity vector
        the error vector
        the area vector
    """
    cdef double xres,yres
    cdef Py_ssize_t M,N
    cdef np.ndarray[np.double_t, ndim=1] qout
    cdef np.ndarray[np.double_t, ndim=1] Intensity
    cdef np.ndarray[np.double_t, ndim=1] Error
    cdef np.ndarray[np.double_t, ndim=1] Area
    cdef Py_ssize_t ix,iy
    cdef Py_ssize_t l
    cdef double x,y,q1
    cdef double * qmin
    cdef double * qmax
    cdef double qmin1
    cdef double qmax1
    cdef double qstep1
    cdef Py_ssize_t K
    cdef Py_ssize_t lowescape, hiescape, masked, zeroerror
    
    if type(res)!=type([]):
        res=[res,res];
    if len(res)==1:
        res=[res[0], res[0]]
    if len(res)>2:
        raise ValueError('res should be a scalar or a nonempty vector of length<=2')
    
    xres=res[0]
    yres=res[1]
    
    M=data.shape[0] # number of rows
    N=data.shape[1] # number of columns

    if dataerr.shape[0]!=M or dataerr.shape[1]!=N:
        raise ValueError('data and dataerr should be of the same shape')
    if mask.shape[0]!=M or mask.shape[1]!=N:
        raise ValueError('data and mask should be of the same shape')

    
    if not shutup:
        print "Creating D matrix...",
#    # if the q-scale was not supplied, create one.
    if q is None:
        if not shutup:
            print "Creating q-scale...",
        qmin1=0
        qmax1=0
        for ix from 0<=ix<M:
            for iy from 0<=iy<N:
                if mask[ix,iy]:
                    continue
                x=((ix-bcx)*xres)
                y=((iy-bcy)*yres)
                q1=4*M_PI*sin(0.5*atan(sqrt(x*x+y*y)/distance))*energy/HC
                if (qmax1==0):
                    qmax1=q1
                if (qmin1==0):
                    qmin1=q1
                if (q1>qmax1):
                    qmax1=q1
                if (q1<qmin1):
                    qmin1=q1
        qstep1=(qmax1-qmin1)/(sqrt(M*M+N*N))
        qout=np.arange(qmin1,qmax1,qstep1,dtype=np.double)
        if not shutup:
            print "done"
    else:
        qout=q
    K=len(qout)
    # initialize the output vectors
    Intensity=np.zeros(K,dtype=np.double)
    Error=np.zeros(K,dtype=np.double)
    Area=np.zeros(K,dtype=np.double)
    if not shutup:
        print "Integrating..."
    # set the bounds of the q-bins in qmin and qmax
    qmin=<double *>malloc(K*sizeof(double))
    qmax=<double *>malloc(K*sizeof(double))
    for l from 0<=l<K:
        if l==0:
            qmin[l]=qout[0]
        else:
            qmin[l]=0.5*(qout[l]+qout[l-1])
        if l==K-1:
            qmax[l]=qout[len(qout)-1]
        else:
            qmax[l]=0.5*(qout[l]+qout[l+1])
    lowescape=0
    hiescape=0
    masked=0
    zeroerror=0
    for ix from 0<=ix<M: #rows
        for iy from 0<=iy<N: #columns
            if mask[ix,iy]:
                masked+=1
                continue
            if dataerr[ix,iy]==0:
                zeroerror+=1
                continue
            x=((ix-bcx)*xres)
            y=((iy-bcy)*yres)
            q1=4*M_PI*sin(0.5*atan(sqrt(x*x+y*y)/distance))*energy/HC
            if q1<qmin[0]:
                lowescape+=1
                continue
            if q1>qmax[K-1]:
                hiescape+=1
                continue
            # go through every q-bin
            for l from 0<=l<K:
                if (q1<=qmax[l]):
                    Intensity[l]+=data[ix,iy]
                    Error[l]+=dataerr[ix,iy]**2
#                    Intensity[l]+=data[ix,iy]/(dataerr[ix,iy]**2)
#                    Error[l]+=1/(dataerr[ix,iy]**2)
                    Area[l]+=1
                    break
    free(qmin)
    free(qmax)
    for l from 0<=l<K:
        if Area[l]>0:
            Intensity[l]/=Area[l]
            Error[l]/=Area[l]
            Error[l]=sqrt(Error[l])
#        if Error[l]>0:        
#            Intensity[l]=Intensity[l]/Error[l] # Error[l] is sum_i(1/sigma^2_i)
#            Error[l]=1/sqrt(Error[l])
#        else:
#            Intensity[l]=NAN
#            Error[l]=NAN
    if not shutup:
        print "done"
    print "Finished integration."
    print "Lowescape: ",lowescape
    print "Hiescape: ",hiescape
    print "Masked: ",masked
    print "ZeroError: ",zeroerror
    return qout,Intensity,Error,Area # return
