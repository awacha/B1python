import numpy as np
cimport numpy as np

from stdlib cimport *

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double sqrt(double)

cdef inline double fsphere(double q, double R):
    """Scattering factor of a sphere
    
    Inputs:
        q: q value(s) (scalar or an array of arbitrary size and shape)
        R: radius (scalar)
        
    Output:
        the values of the scattering factor in an array of the same shape as q
    """
    return 1/q**3*(sin(q*R)-q*R*cos(q*R))

def Ctheorspheregas(np.ndarray[np.double_t, ndim=1] qrange not None,
                    np.ndarray[np.double_t, ndim=1] R not None,
                    np.ndarray[np.double_t, ndim=1] rho not None):
    """Calculate the theoretical scattering intensity of a spatially
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
    """Calculate the theoretical scattering intensity of the sphere structure
    
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
