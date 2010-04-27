import numpy as np
cimport numpy as np

from stdlib cimport *

HC=12398.419

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double sqrt(double)
    double M_PI
    
cdef inline double fsphere(double q, double R):
    """Scattering factor of a sphere
    
    Inputs:
        q: q value(s) (scalar or an array of arbitrary size and shape)
        R: radius (scalar)
        
    Output:
        the values of the scattering factor in an array of the same shape as q
    """
    if q==0:
        return R**3/3
    else:
        return 1/q**3*(sin(q*R)-q*R*cos(q*R))

def Ctheorsphere2D(np.ndarray[np.double_t, ndim=2] data not None,
                   double dist, double wavelength, Py_ssize_t Nx,
                   Py_ssize_t Ny, double pixelsizex, double pixelsizey,
                   double dx=0,double dy=0,bint headerout=False,
                   Py_ssize_t FSN=0,Title=''):
    """Ctheorsphere2D(np.ndarray[np.double_t, ndim=2] data not None,
                   double dist, double wavelength, Py_ssize_t Nx,
                   Py_ssize_t Ny, double pixelsizex, double pixelsizey,
                   double dx=0,double dy=0,bint headerout=False,
                   Py_ssize_t FSN=0,Title=''):
    
    Calculate theoretical scattering of a sphere-structure in a virtual
    transmission SAXS setup.
    
    Inputs:
        data: numpy array representing the sphere structure data. It should have
            at least 5 columns, which are interpreted as x, y, z, R, rho, thus
            the center of the sphere, its radius and average electron density.
            Superfluous columns are disregarded. x is horizontal, y is vertical.
            z points towards the detector.
        dist: sample-to-detector distance, usually in mm.
        wavelength: wavelength of the radiation used, usually in Angstroems
        Nx, Ny: the width and height of the virtual 2D detector, in pixels.
        pixelsizex, pixelsizey: the horizontal and vertical pixel size, usually
            expressed in millimetres (the same as dist)
        dx, dy: horizontal and vertical displacement of the detector. If these
            are 0, the beam falls at the centre. Of the same dimension as dist
            and pixelsize.
        headerout: True if a header structure is to be returned (with beam
            beam center, distance, wavelength etc.)
        FSN: File Sequence Number to write in header.
        Title: Sample title to write in header.

    Output:
        a 2D numpy array containing the scattering image. The row coordinate is
        the vertical coordinate (rows are horizontal).
    """
    cdef double *x,*y,*z,*R,*rho
    cdef np.ndarray[np.double_t, ndim=2] output
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t numspheres
    cdef double tmp,I
    cdef double qx,qy,qz,q
    cdef double sx,sy,sz
    
    if data.shape[1]<5:
        raise ValueError('the number of columns in the input matrix should be 5.')
    output=np.zeros((Ny,Nx),dtype=np.double)
    numspheres=data.shape[0]
    x=<double*>malloc(sizeof(double)*numspheres)
    y=<double*>malloc(sizeof(double)*numspheres)
    z=<double*>malloc(sizeof(double)*numspheres)
    R=<double*>malloc(sizeof(double)*numspheres)
    rho=<double*>malloc(sizeof(double)*numspheres)
    for i from 0<=i<numspheres:
        x[i]=data[i,0]
        y[i]=data[i,1]
        z[i]=data[i,2]
        R[i]=data[i,3]
        rho[i]=data[i,4]
    for i from 0<=i<Nx:
        for j from 0<=j<Ny:
            sx=((i-Nx/2.0)*pixelsizex+dx)
            sy=((j-Ny/2.0)*pixelsizey+dy)
            sz=dist
            tmp=sqrt(sx**2+sy**2+sz**2)
            if tmp==0:
                qx=0
                qy=0
                qz=0
            else:
                qx=sx/tmp*2*M_PI/wavelength
                qy=sy/tmp*2*M_PI/wavelength
                qz=(sz/tmp-1)*2*M_PI/wavelength
            q=sqrt(qx**2+qy**2+qz**2)
            I=0
            for k from 0<=k<numspheres:
                I+=fsphere(q,R[k])**2*(rho[k]**2)
                for l from k<l<numspheres:
                    tmp=(x[k]-x[l])*qx+(y[k]-y[l])*qy+(z[k]-z[l])*qz
                    I+=fsphere(q,R[k])*fsphere(q,R[l])*2*rho[k]*rho[l]*cos(tmp)
            output[j,i]=I
        #print "column",i,"/",Nx,"done"
    free(rho); free(R); free(z); free(y); free(x);
    if headerout:
        return output,{'BeamPosX':(0.5*Ny+1)-dy/pixelsizey,\
                       'BeamPosY':(0.5*Nx+1)-dx/pixelsizex,'Dist':dist,\
                       'EnergyCalibrated':HC/wavelength,'PixelSizeX':pixelsizex,\
                       'PixelSizeY':pixelsizey,\
                       'PixelSize':sqrt(pixelsizex*pixelsizex+pixelsizey*pixelsizey),\
                       'FSN':FSN,'Title':Title}
    else:
        return output
def Ctheorspheregas(np.ndarray[np.double_t, ndim=1] qrange not None,
                    np.ndarray[np.double_t, ndim=1] R not None,
                    np.ndarray[np.double_t, ndim=1] rho not None):
    """Ctheorspheregas(np.ndarray[np.double_t, ndim=1] qrange not None,
                    np.ndarray[np.double_t, ndim=1] R not None,
                    np.ndarray[np.double_t, ndim=1] rho not None):
    
    Calculate the theoretical scattering intensity of a spatially
        uncorrelated sphere structure
    
    Inputs:
        qrange: np.ndarray of q values
        R, rho: one-dimensional np.ndarrays of the same
            lengths, containing the radii and electron-
            densities of the spheres, respectively.
            
    Output:
        a vector of the same size that of qrange. It contains the scattering
        intensities.
    """
    cdef Py_ssize_t i,j
    cdef Py_ssize_t lenq,lensphere
    cdef double q
    cdef double d
    lenq=len(qrange)
    lensphere=len(R)
    if lensphere!=len(rho):
        raise ValueError('argument x and rho should be of the same length!')
    Intensity=np.zeros(lenq,dtype=np.double)
    for i from 0<=i<lenq:
        q=qrange[i]
        Intensity[i]=0
        for j from 0<=j<lensphere:
            Intensity[i]+=(rho[j]*fsphere(q,R[j]))**2
    return Intensity            

def Ctheorspheres(np.ndarray[np.double_t, ndim=1] qrange not None,
                  np.ndarray[np.double_t, ndim=1] x not None,
                  np.ndarray[np.double_t, ndim=1] y not None,
                  np.ndarray[np.double_t, ndim=1] z not None,
                  np.ndarray[np.double_t, ndim=1] R not None,
                  np.ndarray[np.double_t, ndim=1] rho not None):
    """def Ctheorspheres(np.ndarray[np.double_t, ndim=1] qrange not None,
                  np.ndarray[np.double_t, ndim=1] x not None,
                  np.ndarray[np.double_t, ndim=1] y not None,
                  np.ndarray[np.double_t, ndim=1] z not None,
                  np.ndarray[np.double_t, ndim=1] R not None,
                  np.ndarray[np.double_t, ndim=1] rho not None):
    
    Calculate the theoretical scattering intensity of the sphere structure
    
    Inputs:
        qrange: np.ndarray of q values
        x, y, z, R, rho: one-dimensional np.ndarrays of the same
            lengths, containing x, y, z coordinates, radii and electron-
            densities of the spheres, respectively.
            
    Output:
        a vector of the same size that of qrange. It contains the scattering
        intensities.
    """
    cdef double *I
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t lenq,lensphere
    cdef double q
    cdef double d
    cdef double factor1,factor
    lenq=len(qrange)
    lensphere=len(x)
    if lensphere!=len(y):
        raise ValueError('argument x and y should be of the same length!')
    if lensphere!=len(z):
        raise ValueError('argument x and z should be of the same length!')
    if lensphere!=len(R):
        raise ValueError('argument x and R should be of the same length!')
    if lensphere!=len(rho):
        raise ValueError('argument x and rho should be of the same length!')
    I=<double*>malloc(lenq*sizeof(double))
    for i from 0<=i<lenq:
        q=qrange[i]
        for j from 0<=j<lensphere:
            factor1=rho[j]*fsphere(q,R[j])
            I[i]=factor1**2
            for k from 1<=k<i:
                d=sqrt((x[j]-x[k])**2+(y[j]-y[k])**2+(z[j]-z[k])**2)
                if (d==0):
                    factor=1
                else:
                    factor=sin(d*q)/(d*q)
                I[i]+=rho[k]*fsphere(q,R[k])*factor1*factor
    Intensity=np.zeros(lenq,dtype=np.double)
    for i from 0<=i<lenq:
        Intensity[i]=I[i]
    free(I)
    return Intensity            

def Coffchipbinning(np.ndarray[np.double_t, ndim=2] M, Py_ssize_t xlen, Py_ssize_t ylen):
    """
    """
    cdef Py_ssize_t i,i1,j,j1
    cdef Py_ssize_t Nx,Ny
    cdef np.ndarray[np.double_t,ndim=2] N
    
    Nx=M.shape[0]/xlen
    Ny=M.shape[1]/ylen
    
    N=np.zeros((Nx,Ny),np.double)
    for i from 0<=i<Nx:
        print "i==",i
        for i1 from 0<=i1<xlen:
            for j from 0<=j<Ny:
                for j1 from 0<=j1<ylen:
                    N[i,j]+=M[i*xlen+i1,j*ylen+j1]
    return N/(xlen*ylen)
