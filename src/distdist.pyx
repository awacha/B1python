import numpy as np
cimport numpy as np
from stdlib cimport *

cdef extern from "math.h":
    double sqrt(double)
    double M_PI
    double sin(double)
    
cdef extern from "stdlib.h":
    unsigned long RAND_MAX
    unsigned long rand()

def ddistcylinder(double R, double h,np.ndarray[np.double_t,ndim=1] d not None,Py_ssize_t NMC):
    """Calculate the distance distribution function p(r) for a cylinder.
    
    Inputs:
        R: radius
        h: height
        d: vector of the values for r
        NMC: number of Monte-Carlo steps
    
    Outputs:
        a vector, of the same size as d. Normalized that its integral with respect
            to d is the square of the volume of the particle
    """
    cdef Py_ssize_t i,j,lend
    cdef double xa,ya,za,xb,yb,zb
    cdef double d1
    cdef double *myd
    cdef double *myresult
    cdef np.ndarray[np.double_t, ndim=1] result
    

    lend=len(d)
    result=np.zeros(lend,dtype=np.double)
    myd=<double*>malloc(sizeof(double)*lend)
    myresult=<double*>malloc(sizeof(double)*lend)
    
    for i from 0<=i<lend:
        myd[i]=d[i]
        myresult[i]=0
    
    for i from 0<=i<NMC:
        xa=rand()/<double>RAND_MAX*2*R-R
        ya=rand()/<double>RAND_MAX*2*R-R
        if (xa*xa+ya*ya)>R*R:
            i-=1
            continue
        za=rand()/<double>RAND_MAX*h-h/2
        xb=rand()/<double>RAND_MAX*2*R-R
        yb=rand()/<double>RAND_MAX*2*R-R
        if (xb*xb+yb*yb)>R*R:
            i-=1
            continue
        zb=rand()/<double>RAND_MAX*h-h/2
        d1=sqrt((xa-xb)**2+(ya-yb)**2+(za-zb)**2)
        if (d1<myd[0]) or (d1>myd[lend-1]):
            i-=1
            continue
        if d1>0.5*(myd[lend-2]+myd[lend-1]):
            myresult[lend-1]+=1
        else:
            for j from 0<=j<lend:
                if d1<0.5*(myd[j]+myd[j+1]):
                    myresult[j]+=1
                    break
    #now normalize by the bin width and the number of MC steps, then multiply by the square of the volume
    result[0]=myresult[0]/<double>NMC/(0.5*(myd[1]+myd[0])-myd[0])*(R*R*M_PI*h)**2
    for i from 1<=i<lend-1:
        result[i]=myresult[i]/<double>NMC/(0.5*(myd[i]+myd[i+1])-0.5*(myd[i]+myd[i-1]))*(R*R*M_PI*h)**2
    result[lend-1]=myresult[lend-1]/<double>NMC/(myd[lend-1]-0.5*(myd[lend-1]+myd[lend-2]))*(R*R*M_PI*h)**2
    free(myd)
    free(myresult)
    return result

def ddistsphere(double R,np.ndarray[np.double_t,ndim=1] d not None,Py_ssize_t NMC):
    """Calculate the distance distribution function p(r) for a cylinder.
    
    Inputs:
        R: radius
        d: vector of the values for r
        NMC: number of Monte-Carlo steps
    
    Outputs:
        a vector, of the same size as d. Normalized that its integral with respect
            to d is the square of the volume of the particle
    """
    cdef Py_ssize_t i,j,lend
    cdef double xa,ya,za,xb,yb,zb
    cdef double d1
    cdef double *myd
    cdef double *myresult
    cdef np.ndarray[np.double_t, ndim=1] result
    

    lend=len(d)
    result=np.zeros(lend,dtype=np.double)
    myd=<double*>malloc(sizeof(double)*lend)
    myresult=<double*>malloc(sizeof(double)*lend)
    
    for i from 0<=i<lend:
        myd[i]=d[i]
        myresult[i]=0
    
    for i from 0<=i<NMC:
        xa=rand()/<double>RAND_MAX*2*R-R
        ya=rand()/<double>RAND_MAX*2*R-R
        za=rand()/<double>RAND_MAX*2*R-R
        if (xa*xa+ya*ya+za*za)>R*R:
            i-=1
            continue
        xb=rand()/<double>RAND_MAX*2*R-R
        yb=rand()/<double>RAND_MAX*2*R-R
        zb=rand()/<double>RAND_MAX*2*R-R
        if (xb*xb+yb*yb+zb*zb)>R*R:
            i-=1
            continue
        d1=sqrt((xa-xb)**2+(ya-yb)**2+(za-zb)**2)
        if (d1<myd[0]) or (d1>myd[lend-1]):
            i-=1
            continue
        if d1>0.5*(myd[lend-2]+myd[lend-1]):
            myresult[lend-1]+=1
        else:
            for j from 0<=j<lend:
                if d1<0.5*(myd[j]+myd[j+1]):
                    myresult[j]+=1
                    break
    #now normalize by the bin width and the number of MC steps, then multiply by the square of the volume
    result[0]=myresult[0]/<double>NMC/(0.5*(myd[1]+myd[0])-myd[0])*(4*R*R*R*M_PI/3)**2
    for i from 1<=i<lend-1:
        result[i]=myresult[i]/<double>NMC/(0.5*(myd[i]+myd[i+1])-0.5*(myd[i]+myd[i-1]))*(4*R*R*R*M_PI/3)**2
    result[lend-1]=myresult[lend-1]/<double>NMC/(myd[lend-1]-0.5*(myd[lend-1]+myd[lend-2]))*(4*R*R*R*M_PI/3)**2
    free(myd)
    free(myresult)
    return result

def ddistbrick(double a, double b, double c,np.ndarray[np.double_t,ndim=1] d not None,Py_ssize_t NMC):
    """Calculate the distance distribution function p(r) for a cylinder.
    
    Inputs:
        a: length of one side
        b: length of the second side
        c: length of the third side
        d: vector of the values for r
        NMC: number of Monte-Carlo steps
    
    Outputs:
        a vector, of the same size as d. Normalized that its integral with respect
            to d is the square of the volume of the particle
    """
    cdef Py_ssize_t i,j,lend
    cdef double xa,ya,za,xb,yb,zb
    cdef double d1
    cdef double *myd
    cdef double *myresult
    cdef np.ndarray[np.double_t, ndim=1] result
    

    lend=len(d)
    result=np.zeros(lend,dtype=np.double)
    myd=<double*>malloc(sizeof(double)*lend)
    myresult=<double*>malloc(sizeof(double)*lend)
    
    for i from 0<=i<lend:
        myd[i]=d[i]
        myresult[i]=0
    
    for i from 0<=i<NMC:
        xa=rand()/<double>RAND_MAX*2*a-a
        ya=rand()/<double>RAND_MAX*2*b-b
        za=rand()/<double>RAND_MAX*2*c-c
        xb=rand()/<double>RAND_MAX*2*a-a
        yb=rand()/<double>RAND_MAX*2*b-b
        zb=rand()/<double>RAND_MAX*2*c-c
        d1=sqrt((xa-xb)**2+(ya-yb)**2+(za-zb)**2)
        if (d1<myd[0]) or (d1>myd[lend-1]):
            i-=1
            continue
        if d1>0.5*(myd[lend-2]+myd[lend-1]):
            myresult[lend-1]+=1
        else:
            for j from 0<=j<lend:
                if d1<0.5*(myd[j]+myd[j+1]):
                    myresult[j]+=1
                    break
    #now normalize by the bin width and the number of MC steps, then multiply by the square of the volume
    result[0]=myresult[0]/<double>NMC/(0.5*(myd[1]+myd[0])-myd[0])*(a*b*c)**2
    for i from 1<=i<lend-1:
        result[i]=myresult[i]/<double>NMC/(0.5*(myd[i]+myd[i+1])-0.5*(myd[i]+myd[i-1]))*(a*b*c)**2
    result[lend-1]=myresult[lend-1]/<double>NMC/(myd[lend-1]-0.5*(myd[lend-1]+myd[lend-2]))*(a*b*c)**2
    free(myd)
    free(myresult)
    return result
    
def ftddist(np.ndarray[np.double_t,ndim=1] d not None,
            np.ndarray[np.double_t,ndim=1] dist not None,
            np.ndarray[np.double_t,ndim=1] q not None):
    """Calculate the Fourier transform of a distance distribution function
    
    Inputs:
        d: the abscissa of the distance distribution function
        dist: the distance distribution function
        q: the q-values
        
    Outputs:
        a vector of the same size as q, defined as I(q)=int_dmin^dmax(dist(d)*sin(qd)/(qd) dd)
    """
    cdef Py_ssize_t lend
    cdef Py_ssize_t lenq
    cdef np.ndarray[np.double_t,ndim=1] I
    cdef Py_ssize_t i,j
    cdef double qr,qro
    cdef double factor, factoro
    lend=len(d)
    if len(dist)!=lend:
        raise ValueError, "The length of d and dist should be the same!"
    lenq=len(q)
    
    I=np.zeros(lenq,dtype=np.double)
    for i from 0<=i<lenq:
        qr=q[0]*d[0]
        if qr!=0:
            factor=sin(qr)/qr
        else:
            factor=1
        for j from 1<=j<lend:
            qro=qr
            factoro=factor
            qr=q[i]*d[j]
            if qr!=0:
                factor=sin(qr)/qr
            else:
                factor=1
            I[i]+=(dist[j]*factor+dist[j-1]*factoro)*0.5*(d[j]-d[j-1])
    return I    

