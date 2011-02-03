import numpy as np
cimport numpy as np
import utils

cdef extern from "stdlib.h":
    Py_ssize_t RAND_MAX
    Py_ssize_t rand()

cdef extern from "math.h":
    double log(double)
    double exp(double)

cdef inline double randn():
    """Standard normal distribution
    """
    cdef double x
    cdef double y
    cdef int notready
    notready=1
    while(notready):
        x=-log(rand()/<double>RAND_MAX)
        y=exp(-0.5*(x-1)*(x-1))
        if (rand()/<double>RAND_MAX <y):
            notready=0
            if (rand()/<double>RAND_MAX<0.5):
                x=-x
    return x


def asaxsbasicfunctions(np.ndarray[np.double_t, ndim=2] I not None,
                        np.ndarray[np.double_t, ndim=2] Errors not None,
                        np.ndarray[np.double_t, ndim=1] f1 not None,
                        np.ndarray[np.double_t, ndim=1] f2 not None,
                        np.ndarray[np.double_t, ndim=1] df1=None,
                        np.ndarray[np.double_t, ndim=1] df2=None,
                        unsigned int element=0,
                        bool quiet=False,
                        Py_ssize_t NMC=0):
    """Calculate the basic functions (nonresonant, mixed, resonant)
    
    Inputs:
        I: a matrix of intensity (scattering cross section) data. The columns
            should contain the intensities for each energy
        Errors: a matrix of absolute errors of the intensity data. Of the same
            shape as I.
        f1: vector of the f' values for the corresponding columns of I.
        f2: vector of the f'' values for the corresponding columns of I.
        element: the atomic number of the resonant atom. If zero (default),
            derive the basic functions according to Stuhrmann. If nonzero, the
            partial structure factors of the nonresonant part (N), and the
            resonant part (R) are returned, along with the cross-term S_{NR}.
        quiet: true if no output is requested. Default: false
        NMC: number of Monte Carlo steps. If 0 (default), standard gaussian
            error propagation is done, but the contribution of df1 and df2 is
            overestimated. If positive, do a Monte Carlo error approximation.
            
    Outputs:
        N: vector of the nonresonant term
        M: vector of the mixed term
        R: vector of the pure resonant term
        DN: error vector of the nonresonant term
        DM: error vector of the mixed term
        DR: error vector of the resonant term
    """
    cdef Py_ssize_t Nenergies
    cdef Py_ssize_t Ilen
    cdef np.ndarray[np.double_t, ndim=1] N
    cdef np.ndarray[np.double_t, ndim=1] M
    cdef np.ndarray[np.double_t, ndim=1] R
    cdef np.ndarray[np.double_t, ndim=1] DN
    cdef np.ndarray[np.double_t, ndim=1] DM
    cdef np.ndarray[np.double_t, ndim=1] DR
    cdef np.ndarray[np.double_t, ndim=2] A
    cdef np.ndarray[np.double_t, ndim=2] B
    cdef np.ndarray[np.double_t, ndim=2] I1
    cdef double r1,r2,r3
    cdef Py_ssize_t i    
    
    Nenergies=I.shape[1];
    Ilen=I.shape[0];
    if len(f1) != Nenergies:
        raise ValueError("length of the f' vector should match the number of rows in I.")
    if len(f2) != Nenergies:
        raise ValueError("length of the f'' vector should match the number of rows in I.")
    N=np.zeros(Ilen);
    M=np.zeros(Ilen);
    R=np.zeros(Ilen);
    DN=np.zeros(Ilen);
    DM=np.zeros(Ilen);
    DR=np.zeros(Ilen);
    if NMC==0: #no Monte Carlo, just simple error propagation
        #construct matrix A, i.e. dot(A,[N(q),M(q),R(q)]^T)=[I_1(q),I_2(q),...I_N(q)]^T for each q.        
        A=np.ones((Nenergies,3)); 
        A[:,1]=2*(element+f1);
        A[:,2]=(element+f1)**2+f2**2;
        #construct the error matrix of A.        
        DA=np.zeros((A.shape[0],A.shape[1]))
        if df1 is not None:
            DA[:,1]=2*df1;
            DA[:,2]=np.sqrt(4*(element+f1)**2*df1**2+4*f2**2*df2**2)
        #B is the least-squares solver matrix for A, i.e. ((A^T.A)^-1).A^T, "." is the dot product
        B=np.dot(np.linalg.linalg.inv(np.dot(A.T,A)),A.T);
        # calculate the error propagation of B. Note that this is not correct, as the dependence
        # of the elements of A is not taken into account!
        ATA=np.dot(A.T,A)
        ATAerr=utils.dot_error(A.T,A,DA.T,DA)
        invATA=np.linalg.linalg.inv(ATA)
        invATAerr=utils.inv_error(ATA,ATAerr)
        Berror=utils.dot_error(invATA,A.T,invATAerr,DA.T)
        # print the error of B if desired.
        if not quiet:
            print Berror
            print "Condition number of inv(A'*A)*A' is ",np.linalg.linalg.cond(B)
        #for each q, calculate N,M,R and their errors.
        for j in range(0,Ilen):
            tmp=np.dot(B,I[j,:])
            N[j]=tmp[0];
            M[j]=tmp[1];
            R[j]=tmp[2];
            tmpe=utils.dot_error(B,I[j,:],Berror,Errors[j,:])
            DN[j]=tmpe[0];
            DM[j]=tmpe[1];
            DR[j]=tmpe[2];
    else: # Monte Carlo error propagation
        # first, calculate the true value of N,M,R in a similar way as in the
        # non-MC case above.
        A=np.ones((Nenergies,3))
        A[:,1]=2*(element+f1);
        A[:,2]=(element+f1)**2+f2**2;
        B=np.dot(np.linalg.linalg.inv(np.dot(A.T,A)),A.T);
        for j in range(0,Ilen):
            tmp=np.dot(B,I[j,:])
            N[j]=tmp[0];
            M[j]=tmp[1];
            R[j]=tmp[2];
        # Monte Carlo routine: manipulate each quantity (Intensities, anomalous
        # coefficients) by a random gaussian.
        for i from 0<=i<NMC:
            if (i%100)==0:
                print "Monte Carlo iteration #%d"%i
            # manipulate the intensity
            I1=np.random.randn(Errors.shape[0],Errors.shape[1])*Errors+I
            #construct the new "A" matrix. As we already have one, we only
            # update its elements where needed.
            for j from 0<=j<Nenergies:
                r1=randn()*df1[j]
                r2=randn()*df2[j]
                A[j,1]=2*(element+f1[j]+r1)
                A[j,2]=(element+f1[j]+r1)**2+((f2[j]+r2)**2)
            #find the LSQ matrix.
            B=np.dot(np.linalg.linalg.inv(np.dot(A.T,A)),A.T);
            # for each q, find the squared difference of the expected value
            # and the just calculated value of each basic function. Summarize
            # these over the MC loop.
            for j from 0<=j<Ilen:
                tmp=np.dot(B,I1[j,:])
                DN[j]+=(N[j]-tmp[0])**2
                DM[j]+=(M[j]-tmp[1])**2
                DR[j]+=(R[j]-tmp[2])**2
        #Just after the Monte Carlo loop, the D? vectors hold the sums of
        # squared differences. Divide them by (NMC-1) and take the square
        # root to get the standard deviations.
        DN=np.sqrt(DN/(NMC-1))
        DR=np.sqrt(DM/(NMC-1))
        DR=np.sqrt(DR/(NMC-1))
    # return with the results (both MC and non-MC).
    return N,M,R,DN,DM,DR
