import numpy as np
cimport numpy as np
from stdlib cimport *


cdef extern from "math.h":
    double cos(double)
    double sin(double)
    double sqrt(double)
    double atan(double)
    double floor(double)
    double atan2(double,double)
    double M_PI
    double NAN
    double INFINITY
    double ceil(double)
    double fmod(double,double)

cdef double HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units
    
def polartransform(np.ndarray[np.double_t, ndim=2] data not None,
                   np.ndarray[np.double_t, ndim=1] r,
                   np.ndarray[np.double_t, ndim=1] phi,
                   double origx, double origy):
    """Calculates a matrix of a polar representation of the image.
    
    Inputs:
        data: the 2D matrix
        r: vector of polar radii
        phi: vector of polar angles (degrees)
        origx: the x (row) coordinate of the origin, starting from 1
        origy: the y (column) coordinate of the origin, starting from 1
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
            bint shutup=True, bint returnavgq=False):
    """Do radial integration on 2D scattering images. Now this takes the
        functional determinant dq/dr into account.
    
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
            starting from 1
        bcy: the coordinate of the beam center in the y (column) direction,
            starting from 1
        mask: the mask matrix (of the same size as data). Nonzero is masked,
            zero is not masked
        q: the q points at which the integration is requested. It should be
            defined in 1/Angstroems.
        shutup: if True, work quietly (do not print messages).
        returnavgq: if True, returns the average q value for each bin, ie. the
            average of the q-values corresponding to the centers of the pixels
            which fell into each q-bin.
        
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
    cdef double * qmax
    cdef double *weight
    cdef double w
    cdef double qmin1
    cdef double qmax1
    cdef double qstep1
    cdef double rho
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
        q=np.arange(qmin1,qmax1,qstep1,dtype=np.double)
        if not shutup:
            print "done"
    else:
        pass
        # do nothing, as q already contains the q-values
    K=len(q)
    # initialize the output vectors
    Intensity=np.zeros(K,dtype=np.double)
    Error=np.zeros(K,dtype=np.double)
    Area=np.zeros(K,dtype=np.double)
    qout=np.zeros(K,dtype=np.double)
    if not shutup:
        print "Integrating..."
    # set the bounds of the q-bins in qmin and qmax
    qmax=<double *>malloc(K*sizeof(double))
    weight=<double *>malloc(K*sizeof(double))
    for l from 0<=l<K:
        weight[l]=0
        if l==K-1:
            qmax[l]=q[K-1]
        else:
            qmax[l]=0.5*(q[l]+q[l+1])
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
            rho=sqrt(x*x+y*y)/distance
            q1=4*M_PI*sin(0.5*atan(rho))*energy/HC
            if q1<q[0]:
                lowescape+=1
                continue
            if q1>qmax[K-1]:
                hiescape+=1
                continue
            # go through every q-bin
            w=(2*M_PI*energy/HC/distance)**2*(2+rho**2+2*sqrt(1+rho**2))/( (1+rho**2+sqrt(1+rho**2))**2*sqrt(1+rho**2) )
            for l from 0<=l<K:
                if (q1<=qmax[l]):
                    qout[l]+=q1
                    Intensity[l]+=data[ix,iy]*w
                    Error[l]+=dataerr[ix,iy]**2*w
                    Area[l]+=1
                    weight[l]+=w
                    break
    free(qmax)
    free(weight)
    for l from 0<=l<K:
        if Area[l]>0:
            if weight[l]<=0:
                print "Area is not zero but weight is nonpositive at index",l
            qout[l]/=weight[l]
            Intensity[l]/=weight[l]
            Error[l]/=weight[l]
            Error[l]=sqrt(Error[l])
    if not shutup:
        print "Finished integration."
        print "Lowescape: ",lowescape
        print "Hiescape: ",hiescape
        print "Masked: ",masked
        print "ZeroError: ",zeroerror
    if returnavgq:
        return qout,Intensity,Error,Area
    else:
        return q,Intensity,Error,Area
    
def azimintpixC(np.ndarray[np.double_t, ndim=2] data not None,
                np.ndarray[np.double_t, ndim=2] error,
                orig,
                np.ndarray[np.uint8_t, ndim=2] mask not None,
                Ntheta=100,
                double dmin=0,
                double dmax=INFINITY):
    """Perform azimuthal integration of image.

    Inputs:
        data: matrix to average
        error: error matrix. If not applicable, set it to None
        orig: vector of beam center coordinates, starting from 1.
        mask: mask matrix; 1 means masked, 0 means non-masked
        Ntheta: number of desired points on the abscissa
        dmin: the lower bound of the circle stripe (expressed in pixel units)
        dmax: the upper bound of the circle stripe (expressed in pixel units)

    Outputs: theta,I,[E],A
        theta: theta-range, in radians
        I: intensity points
        E: error values (returned only if the "error" argument was not None)
        A: effective area points
    """
    cdef Py_ssize_t Ntheta1
    cdef np.ndarray[np.double_t, ndim=1] theta
    cdef np.ndarray[np.double_t, ndim=1] I
    cdef np.ndarray[np.double_t, ndim=1] E
    cdef np.ndarray[np.double_t, ndim=1] A
    cdef Py_ssize_t ix,iy
    cdef Py_ssize_t M,N
    cdef Py_ssize_t index
    cdef double bcx, bcy
    cdef double d,x,y,phi
    cdef int errornone
    cdef Py_ssize_t escaped

    Ntheta1=<Py_ssize_t>floor(Ntheta)
    M=data.shape[0]
    N=data.shape[1]
    if (mask.shape[0]!=M) or (mask.shape[1]!=N):
        raise ValueError, "The size and shape of data and mask should be the same."
    bcx=orig[0]-1
    bcy=orig[1]-1
    
    theta=np.linspace(0,2*np.pi,Ntheta1) # the abscissa of the results
    I=np.zeros(Ntheta1,dtype=np.double) # vector of intensities
    A=np.zeros(Ntheta1,dtype=np.double) # vector of effective areas
    E=np.zeros(Ntheta1,dtype=np.double)

    errornone=(error is None)
    escaped=0
    for ix from 0<=ix<M:
        for iy from 0<=iy<N:
            if mask[ix,iy]:
                continue
            x=ix-bcx
            y=iy-bcy
            d=sqrt(x**2+y**2)
            if (d<dmin) or (d>dmax):
                continue
            phi=atan2(y,x)
            index=<Py_ssize_t>floor(phi/(2*M_PI)*Ntheta1)
            if index>=Ntheta1:
                escaped=escaped+1
                continue
            I[index]+=data[ix,iy]
            if not errornone:
                E[index]+=error[ix,iy]**2
            A[index]+=1
    print "Escaped: ",escaped
    for index from 0<=index<Ntheta1:
        if A[index]>0:
            I[index]/=A[index]
            if not errornone:
                E[index]=sqrt(E[index]/A[index])
    if errornone:
        return theta,I,A
    else:
        return theta,I,E,A
 
def imageintC(np.ndarray[np.double_t,ndim=2] data not None,
              orig,
              np.ndarray[np.double_t,ndim=2] mask not None,
              fi=None, dfi=None):
    """Perform radial averaging on the image.
    
    Inputs:
        data: matrix to average
        orig: vector of beam center coordinates, starting from 1.
        mask: mask matrix; 1 means masked, 0 means non-masked
        fi: starting angle for sector averaging, in degrees
        dfi: angle extent for sector averaging, in degrees
    Outputs:
        vector of integrated values
        vector of effective areas
        
    Note: based on the work of Mika Torkkeli
    """
    cdef double bcx,bcy,fi1,dfi1
    cdef Py_ssize_t i,j,d
    cdef Py_ssize_t Nrow,Ncol,Nbins
    cdef double x,y,phi
    cdef int sectormode
    cdef double * C1
    cdef double * NC1
    # X: row (0-th dimension), Y: column (1st dimension)
    bcx=orig[0]-1
    bcy=orig[1]-1
    Nrow=data.shape[0]
    Ncol=data.shape[1]
    Nbins=<Py_ssize_t>ceil(sqrt(Nrow**2+Ncol**2))+1
    C1=<double*>malloc(Nbins*sizeof(double))
    NC1=<double*>malloc(Nbins*sizeof(double))
    for i from 0<=i<Nbins:
        C1[i]=0
        NC1[i]=0
    if (fi is not None) and (dfi is not None):
        sectormode=1
        fi1=fi*M_PI/180.0
        dfi1=dfi*M_PI/180.0
    else:
        sectormode=0
        fi1=0
        dfi1=2*M_PI
    for i from 0<=i<Nrow:
        for j from 0<=j<Ncol:
            if mask[i,j]:
                continue
            x=i-bcx
            y=j-bcy
            if sectormode:
                phi=fmod(atan2(y,x)-fi1+10*M_PI,2*M_PI)
                if phi>dfi1:
                    continue
            d=<Py_ssize_t>floor(sqrt(x*x+y*y))
            C1[d]+=data[i,j]
            NC1[d]+=1
    #find the last nonzero bin
    for d from 0<=d<Nbins:
        if NC1[Nbins-1-d]>0:
            break
    #create return np.ndarrays
    C=np.zeros(d,dtype=np.double)
    NC=np.zeros(d,dtype=np.double)
    #copy results
    for i from 0<=i<Nbins:
        if (NC1[i]>0):
            C[i]=C1[i]/NC1[i]
            NC[i]=NC1[i]
    #free allocated variables
    free(C1)
    free(NC1)
    #return
    return C,NC
