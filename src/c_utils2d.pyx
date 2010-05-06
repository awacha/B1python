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
    double exp(double)
    double M_PI
    double NAN
    double INFINITY
    double ceil(double)
    double fmod(double,double)
    int isfinite(double)

cdef double HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

cdef gaussian(double x0, double sigma, double x):
    return 1/sqrt(2*M_PI*sigma*sigma)*exp(-(x-x0)**2/(2*sigma**2))
    
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
            bint shutup=True, bint returnavgq=False, phi0=None, dphi=None,
            returnmask=False, double fuzzy_FWHM=0, bint symmetric_sector=False):
    """
    def radintC(np.ndarray[np.double_t,ndim=2] data not None,
            np.ndarray[np.double_t,ndim=2] dataerr not None,
            double energy, double distance, res,
            double bcx, double bcy,
            np.ndarray[np.uint8_t, ndim=2] mask not None,
            np.ndarray[np.double_t, ndim=1] q=None,
            bint shutup=True, bint returnavgq=False, phi0=None, dphi=None,
            returnmask=False, double fuzzy_FWHM=0, bint symmetric_sector=False):

        Do radial integration on 2D scattering images. Now this takes the
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
        phi0: starting angle if sector integration is requested. Expressed
            in radians. Set it to None if simple radial averaging is needed.
        dphi: arc angle if sector integration is requested. Expressed in
            radians. Set it to None if simple radial averaging is needed.
        returnmask: True if the effective mask matrix is to be returned
            (0 for pixels taken into account, 1 for all the others).
        fuzzy_FWHM: FWHM for the Gauss weighing function for fuzzy integration
            (where pixels are weighed according to their distance in q
            from the desired q value of the bin.
        symmetric_sector: True if the mirror part of the sector should be taken
            into account as well on sector integration. Ie. pixels falling into
            sectors of width dphi, and starting at phi0 and pi+phi0, respectively,
            will be accounted for.
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
    cdef double phi0a,dphia
    cdef Py_ssize_t K
    cdef Py_ssize_t lowescape, hiescape, masked, zeroerror
    cdef int sectorint
    cdef np.ndarray[np.uint8_t,ndim=2] maskout
    cdef double g,w1
    cdef double symmetric_sector_periodicity
   
    if type(res)!=type([]) and type(res)!=type(()) and type(res)!=np.ndarray:
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

    if (phi0 is not None) and (dphi is not None):
        phi0a=phi0
        dphia=dphi
        sectorint=1
    else:
        phi0a=0
        dphia=3*M_PI
        sectorint=0

    if symmetric_sector:
        symmetric_sector_periodicity=1
    else:
        symmetric_sector_periodicity=2

    if returnmask:
        maskout=np.ones([data.shape[0],data.shape[1]],dtype=np.uint8)
    
    if not shutup:
        print "Creating D matrix...",
    # if the q-scale was not supplied, create one.
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
    if fuzzy_FWHM<=0:
        qmax=<double *>malloc(K*sizeof(double))
    else:
        qmax=NULL
    weight=<double *>malloc(K*sizeof(double))
    for l from 0<=l<K:
        weight[l]=0
        if fuzzy_FWHM<=0:
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
            if not (isfinite(data[ix,iy]) and isfinite(dataerr[ix,iy])):
                continue
            x=((ix-bcx)*xres)
            y=((iy-bcy)*yres)
            if sectorint:
                phi=atan2(y,x)
                if fmod(phi-phi0a+M_PI*10,symmetric_sector_periodicity*M_PI)>dphia:
                    continue
            rho=sqrt(x*x+y*y)/distance
            q1=4*M_PI*sin(0.5*atan(rho))*energy/HC
            if q1<q[0]:
                lowescape+=1
                continue
            if q1>q[K-1]:
                hiescape+=1
                continue
            #weight, corresponding to the Jacobian determinant
            w=(2*M_PI*energy/HC/distance)**2*(2+rho**2+2*sqrt(1+rho**2))/( (1+rho**2+sqrt(1+rho**2))**2*sqrt(1+rho**2) )
            # go through every q-bin
            for l from 0<=l<K:
                if fuzzy_FWHM<=0: # traditional integration
                    if (q1>qmax[l]): # if the q-value corresponding to the center of the current pixel
                                     # is greater than the upper limit of the l-eth bin, then skip to
                                     # the next bin.
                        continue
                    # If q1 becomes smaller than equal to the top of the current bin, do not skip, but
                    # calculate this pixel into this bin with weight w
                    w1=w 
                else: # if "fuzzy" integration is preferred, the weight from the Jacobian is multiplied
                      # by the weight from the Gaussian
                    w1=w*gaussian(q[l],0.5*fuzzy_FWHM,q1)
                # now we have a weight. We can reach this point in only two ways:
                #   1) traditional integration and bin #l is the bin in which the pixel falls
                #  or
                #   2) fuzzy integration. Each pixel is calculated into each bin, but with
                #      different weight.
                qout[l]+=q1*w1
                Intensity[l]+=data[ix,iy]*w1
                Error[l]+=dataerr[ix,iy]**2*w1
                Area[l]+=1
                weight[l]+=w1
                if returnmask:
                    maskout[ix,iy]=0
                if fuzzy_FWHM<=0: # we must do this for the traditional integration,
                                  # because omitting this would cause the pixel to be
                                  # calculated into the next bin as well.
                    break
    #normalize the results
    for l from 0<=l<K:
        if Area[l]>0:
            if weight[l]<=0:
                print "Area is not zero but weight is nonpositive at index",l,"; w=",weight[l],"; A=",Area[l]
                Intensity[l]=0
                Error[l]=0
                Area[l]=0
            else:
                qout[l]/=weight[l]
                Intensity[l]/=weight[l]
                Error[l]/=weight[l]
                Error[l]=sqrt(Error[l])
    if fuzzy_FWHM<=0:
        free(qmax)
    free(weight)
    if not shutup:
        print "Finished integration."
        print "Lowescape: ",lowescape
        print "Hiescape: ",hiescape
        print "Masked: ",masked
        print "ZeroError: ",zeroerror
    if returnavgq:
        if returnmask:
            return qout,Intensity,Error,Area,maskout
        else:
            return qout,Intensity,Error,Area
    else:
        if returnmask:
            return q,Intensity,Error,Area,maskout
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
    #print "Escaped: ",escaped
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
    C=np.zeros(Nbins-d,dtype=np.double)
    NC=np.zeros(Nbins-d,dtype=np.double)
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

def azimintqC(np.ndarray[np.double_t, ndim=2] data not None,
              np.ndarray[np.double_t, ndim=2] error, # error can be None
              double energy,
              double dist,
              res,
              orig,
              np.ndarray[np.uint8_t, ndim=2] mask not None,
              Ntheta=100,
              double qmin=0,
              double qmax=INFINITY,bint returnmask=False):
    """Perform azimuthal integration of image, with respect to q values

    Inputs:
        data: matrix to average
        error: error matrix. If not applicable, set it to None
        energy: photon energy
        dist: sample-detector distance
        res: resolution of the detector (mm/pixel)
        orig: vector of beam center coordinates, starting from 1.
        mask: mask matrix; 1 means masked, 0 means non-masked
        Ntheta: number of desired points on the abscissa
        qmin: the lower bound of the circle stripe (expressed in q units)
        qmax: the upper bound of the circle stripe (expressed in q units)
        returnmask: if True, a mask is returned, only the pixels taken into
            account being unmasked (0). 
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
    cdef double q
    cdef int errornone
    cdef Py_ssize_t escaped
    cdef double resx,resy
    cdef np.ndarray[np.uint8_t, ndim=2] maskout
    
    if type(res)!=type([]) and type(res)!=type(()) and type(res)!=np.ndarray:
        res=[res,res];
    if len(res)==1:
        res=[res[0], res[0]]
    if len(res)>2:
        raise ValueError('res should be a scalar or a nonempty vector of length<=2')
    
    resx=res[0]
    resy=res[1]
    

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
    if returnmask:
        maskout=np.ones([data.shape[0],data.shape[1]],dtype=np.uint8)

    errornone=(error is None)
    escaped=0
    for ix from 0<=ix<M:
        for iy from 0<=iy<N:
            if mask[ix,iy]:
                continue
            x=(ix-bcx)*resx
            y=(iy-bcy)*resy
            d=sqrt(x**2+y**2)
            q=4*M_PI*sin(0.5*atan2(d,dist))*energy/HC
            if (q<qmin) or (q>qmax):
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
            if returnmask:
                maskout[ix,iy]=0
    #print "Escaped: ",escaped
    for index from 0<=index<Ntheta1:
        if A[index]>0:
            I[index]/=A[index]
            if not errornone:
                E[index]=sqrt(E[index]/A[index])
    if errornone:
        if returnmask:
            return theta,I,A,maskout
        else:
            return theta,I,A
    else:
        if returnmask:
            return theta,I,E,A,maskout
        else:
            return theta,I,E,A
