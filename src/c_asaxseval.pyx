import numpy as np
cimport numpy as np
import utils
from libc.stdlib cimport *
from libc.math cimport *

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
                        int quiet=False,
                        Py_ssize_t NMC=0):
    """Calculate the basic functions (nonresonant, mixed, resonant)
    
    Inputs:
        I: a matrix of intensity (scattering cross section) data. The columns
            should contain the intensities for each energy
        Errors: a matrix of absolute errors of the intensity data. Of the same
            shape as I.
        f1: vector of the f' values for the corresponding columns of I.
        f2: vector of the f'' values for the corresponding columns of I.
        df1: error vector of the f' values for the corresponding columns of I.
        df2: error vector of the f'' values for the corresponding columns of I.
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
    cdef np.ndarray[np.double_t, ndim=2] ATA
    cdef np.ndarray[np.double_t, ndim=2] B
    cdef np.ndarray[np.double_t, ndim=2] I1
    cdef np.ndarray[np.double_t,ndim=1] tmp
    cdef np.ndarray[np.double_t,ndim=1] tmpe
    cdef double *DN1
    cdef double *DM1
    cdef double *DR1
    cdef np.ndarray[np.double_t, ndim=1] N1
    cdef np.ndarray[np.double_t, ndim=1] M1
    cdef np.ndarray[np.double_t, ndim=1] R1
    cdef double r1,r2,r3
    cdef Py_ssize_t i,j
    
    Nenergies=I.shape[1];
    Ilen=I.shape[0];
    if df1 is None:
        df1=np.zeros(Nenergies,np.double)
    if df2 is None:
        df2=np.zeros(Nenergies,np.double)
    if len(f1) != Nenergies:
        raise ValueError("length of the f' vector should match the number of rows in I.")
    if len(f2) != Nenergies:
        raise ValueError("length of the f'' vector should match the number of rows in I.")
    if len(df1) != Nenergies:
        raise ValueError("length of the f' error vector should match the number of rows in I.")
    if len(df2) != Nenergies:
        raise ValueError("length of the f'' error vector should match the number of rows in I.")
    N=np.zeros(Ilen);
    M=np.zeros(Ilen);
    R=np.zeros(Ilen);
    DN=np.zeros(Ilen);
    DM=np.zeros(Ilen);
    DR=np.zeros(Ilen);
    if NMC==0: #no Monte Carlo, just simple (and incorrect!!!) error propagation
        #construct matrix A, i.e. for which A.p=I where p is the vector of partial
        # structure factors ([Snn,Snr,Srr] where n is nonresonant, r is resonant),
        # and I is the vector of intensities ([I(E1),I(E2),...I(EN)] for energies)
        # In all cases, q is fixed, a loop is performed through q-s.
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
            tmp=np.linalg.linalg.solve(ATA,np.dot(A.T,I[j,:]))
            N[j]=tmp[0];
            M[j]=tmp[1];
            R[j]=tmp[2];
            tmpe=utils.dot_error(B,I[j,:],Berror,Errors[j,:])
            DN[j]=tmpe[0];
            DM[j]=tmpe[1];
            DR[j]=tmpe[2];
    else: # Monte Carlo error propagation
        DN1=<double*>malloc(sizeof(double)*Ilen)
        DM1=<double*>malloc(sizeof(double)*Ilen)
        DR1=<double*>malloc(sizeof(double)*Ilen)
        N1=np.zeros(Ilen,np.double)
        M1=np.zeros(Ilen,np.double)
        R1=np.zeros(Ilen,np.double)
        # first, calculate the true value of N,M,R in a similar way as in the
        # non-MC case above.
        A=np.ones((Nenergies,3))
        A[:,1]=2*(element+f1);
        A[:,2]=(element+f1)**2+f2**2;
        for j from 0<=j<Ilen:
            tmp=np.linalg.linalg.solve(np.dot(A.T,A),np.dot(A.T,I[j,:]))
            N[j]=tmp[0];
            M[j]=tmp[1];
            R[j]=tmp[2];
            #zero the D?1 arrays if we are looping
            DN1[j]=0
            DM1[j]=0
            DR1[j]=0
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
            # for each q, find the squared difference of the expected value
            # and the just calculated value of each basic function. Summarize
            # these over the MC loop.
            for j from 0<=j<Ilen:
                tmp=np.linalg.linalg.solve(np.dot(A.T,A),np.dot(A.T,I1[j,:]))
                N1[j]+=tmp[0]
                M1[j]+=tmp[1]
                R1[j]+=tmp[2]
                DN1[j]+=(N[j]-tmp[0])**2
                DM1[j]+=(M[j]-tmp[1])**2
                DR1[j]+=(R[j]-tmp[2])**2
        #Just after the Monte Carlo loop, the D? vectors hold the sums of
        # squared differences. Divide them by (NMC-1) and take the square
        # root to get the standard deviations.
        for j from 0<=j<Ilen:
            DN[j]=sqrt(DN1[j]/(NMC-1))
            DM[j]=sqrt(DM1[j]/(NMC-1))
            DR[j]=sqrt(DR1[j]/(NMC-1))
            N1[j]=N1[j]/NMC
            M1[j]=M1[j]/NMC
            R1[j]=R1[j]/NMC
        free(DN1)
        free(DM1)
        free(DR1)
        return N,M,R,DN,DM,DR,N1,M1,R1
    # return with the results (both MC and non-MC).
    return N,M,R,DN,DM,DR

def reconstructfrompsfs(np.ndarray[np.double_t, ndim=1] N not None,
                        np.ndarray[np.double_t, ndim=1] M not None,
                        np.ndarray[np.double_t, ndim=1] R not None,
                        np.ndarray[np.double_t, ndim=1] f1 not None,
                        np.ndarray[np.double_t, ndim=1] f2 not None,
                        np.ndarray[np.double_t, ndim=1] DN=None,
                        np.ndarray[np.double_t, ndim=1] DM=None,
                        np.ndarray[np.double_t, ndim=1] DR=None,
                        np.ndarray[np.double_t, ndim=1] df1=None,
                        np.ndarray[np.double_t, ndim=1] df2=None,
                        unsigned int element=0):
    """Reconstruct the scattering intensity from the partial structure factors
    and the anomalous scattering factors.
    
    Inputs:
        N: nonresonant part or Snn
        M: mixed part or Snr
        R: resonant part or Srr
        f1: real part of the anomalous scattering factors
        f2: imaginary part of the anomalous scattering factors
        DN [optional]: error of the nonresonant part or Snn
        DM [optional]: error of the mixed part or Snr
        DR [optional]: error of the resonant part or Srr
        df1 [optional]: error of the real part of the anomalous scattering factors
        df2 [optional]: error of the imaginary part of the anomalous scattering factors
        element: atomic number. If 0, input arguments are treated as the basic
            functions. If it is defined, arguments are supposed to be the PSFs.
            
        All input arguments except element are 1D numpy arrays (dype: double)
            
    Outputs: Ints, [Errs]
        Ints: intensities in a 2D array.
        Errs [only if errors are given]: error matrix
    """
    cdef Py_ssize_t Nenergies
    cdef Py_ssize_t Ilen
    cdef np.ndarray[np.double_t, ndim=2] A
    cdef np.ndarray[np.double_t, ndim=2] Ints
    cdef np.ndarray[np.double_t, ndim=2] Errs
    cdef Py_ssize_t i    
    cdef int errorneeded
    
    errorneeded=True    
    Nenergies=len(f1);
    Ilen=len(N);
    if DN is None:
        errorneeded=False
        DN=np.zeros(Ilen,dtype=np.double)
    if DM is None:
        errorneeded=False
        DM=np.zeros(Ilen,dtype=np.double)
    if DR is None:
        errorneeded=False
        DR=np.zeros(Ilen,dtype=np.double)
    if df1 is None:
        errorneeded=False
        df1=np.zeros(Nenergies,dtype=np.double)
    if df2 is None:
        errorneeded=False
        df2=np.zeros(Nenergies,dtype=np.double)
    if len(f2) != Nenergies:
        raise ValueError("length of the f'' vector should match the number of energies.")
    if len(df2) != Nenergies:
        raise ValueError("length of the df'' vector should match the number of energies.")
    if len(df1) != Nenergies:
        raise ValueError("length of the df' vector should match the number of energies.")
    if (len(DN) != Ilen) or (len(DM) != Ilen) or (len(DR) != Ilen) or \
         (len(M) != Ilen) or (len(R) != Ilen):
        raise ValueError("Lengths of vectors N, M, R, DN, DM, DN should match!")
    Ints=np.zeros((Ilen,Nenergies),dtype=np.double)
    Errs=np.zeros((Ilen,Nenergies),dtype=np.double)
    A=np.ones((Nenergies,3)); 
    A[:,1]=2*(element+f1);
    A[:,2]=(element+f1)**2+f2**2;
    #construct the error matrix of A.        
    DA=np.zeros((A.shape[0],A.shape[1]))
    if df1 is not None:
        DA[:,1]=2*df1;
        DA[:,2]=np.sqrt(4*(element+f1)**2*df1**2+4*f2**2*df2**2)
    for i in range(0,Ilen):
        Ints[i,:]=np.dot(A,np.array([N[i],M[i],R[i]]))
        Errs[i,:]=utils.dot_error(A,np.array([N[i],M[i],R[i]]),DA,np.array([DN[i],DM[i],DR[i]]))        
    if errorneeded:
        return Ints,Errs
    else:
        return Ints
