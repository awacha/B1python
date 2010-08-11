import numpy as np
import warnings
cimport numpy as np

from stdlib cimport *
cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double M_PI
    double fabs(double)


cdef inline double gaussian_fast(double x, double x0, double std):
    return 1/sqrt(2*M_PI*std*std)*exp(-(x-x0)*(x-x0)/(2*std*std))

def multigauss(np.ndarray[np.double_t, ndim=1] p not None,
                np.ndarray[np.double_t, ndim=1] x not None,
                np.ndarray[np.double_t, ndim=1] y not None,
                np.ndarray[np.double_t, ndim=1] yerror not None):
    """def multigauss(np.ndarray[np.double_t, ndim=1] p not None,
                np.ndarray[np.double_t, ndim=1] x not None,
                np.ndarray[np.double_t, ndim=1] y not None,
                np.ndarray[np.double_t, ndim=1] yerror not None):

    Fitting function, usable in scipy.optimize.leastsq. This
        model function fits multiple Gaussian peaks.
        
    Inputs:
        p: array of parameters. First value is a constant offset.
            After it, position/scattering/amplitude for gaussian peaks.
        x: array of x data
        y: array of y data
        yerrr: array of y error data (weighing)
    
    Output:
        (y-p[0]-Gaussian(x-p[1])*p[2]-Gaussian(x-p[3])*p[4]-...)/yerror
    """
    cdef Py_ssize_t i,j
    cdef np.ndarray[np.double_t, ndim=1] output
    output=np.zeros(len(x),np.double)
    for i from 0<=i<(len(p)-1)/3:
        for j from 0<=j<len(output):
            output[j]+=fabs(p[3*i+3])*gaussian_fast(x[j],p[3*i+1],p[3*i+2])
    for i from 0<=i<len(output):
        output[i]+=p[0]-y[i]
        output[i]/=yerror[i]
    return output
    
