import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt, floor, atan, tan, cos, abs

cdef int relaxedGaussSeidel(double *A, double *b, double *x, Py_ssize_t N, double relaxpar, Py_ssize_t Niter,double tolerance=0.0001):
    """A matrix: A[i,j]=i-th row, j-th column. Column-first ordering: A[i,j]=A[j*N+i]
    """
    cdef Py_ssize_t i,k,it
    cdef double d
    cdef double *x2
    x2=<double*>malloc(sizeof(double)*N)
    for it from 0<=it<Niter:
        for i from 0<=i<N:
            x2[i]=x[i]
        for i from 0<=i<N:
            d=b[i]
            for k from 0<=k<N:
                d-=A[i+k*N]*x[k]
            x[i]+=relaxpar*d/A[i+i*N]
        d=0
        for i from 0<=i<N:
            d+=(x2[i]-x[i])**2
        if d<tolerance:
            break
    free(x2)
    if d<tolerance:
        return 1
    else:
        return 0

def GaussSeidel(np.ndarray [np.double_t, ndim=2] A, np.ndarray[np.double_t, ndim=1] b,
                np.ndarray [np.double_t, ndim=1] x, double relaxpar, Py_ssize_t Niter):
    """def GaussSeidel(np.ndarray [np.double_t, ndim=2] A, np.ndarray[np.double_t, ndim=1] b,
                np.ndarray [np.double_t, ndim=1] x, double relaxpar, Py_ssize_t Niter):

    Solve a linear equation according to the relaxed Gauss-Seidel iterative method.
    
    Inputs:
        A: N-by-N double matrix
        b: right-hand side
        x: first estimate for the solution
        relaxpar: parameter for the stabilization, between 0 and 2
        Niter: number of iterations
    
    Outputs:
        the result. The original "x" vector is intact.
    """
    cdef double *A1
    cdef double *b1
    cdef double *x1
    cdef Py_ssize_t N
    cdef Py_ssize_t i
    cdef np.ndarray[np.double_t, ndim=1] out
    
    N=A.shape[0]
    if A.shape[1]!=N:
        raise ValueError('Matrix A should be square.')
    if len(b)!=N:
        raise ValueError('Vector b should be compatible in size with matrix A.')
    if len(x)!=N:
        raise ValueError('Vector x should be compatible in size with matrix A.')
    if relaxpar<0 or relaxpar>2:
        raise ValueError('relaxpar should be between 0 and 2 to converge.')
    A1=<double*>malloc(sizeof(double)*N*N)
    b1=<double*>malloc(sizeof(double)*N)
    x1=<double*>malloc(sizeof(double)*N)
    for i from 0<=i<N:
        b1[i]=b[i]
        x1[i]=x[i]
    for i from 0<=i<N*N:
        A1[i]=A[i%N,i/N]
    relaxedGaussSeidel(A1,b1,x1,N,relaxpar,Niter)
    out=np.zeros((x.shape[0]))
    for i from 0<=i<N:
        out[i]=x1[i]
    free(A1)
    free(b1)
    free(x1)
    return out

cdef double *ctrapezoidshapefunction(double lengthbase,double lengthtop, double *x, Py_ssize_t lenx):
    cdef double * T
    cdef Py_ssize_t i
    cdef double lb2mlt2
    if x==NULL:
        return NULL
    lb2mlt2=lengthbase**2-lengthtop**2
    if lenx<2:
        lenx=1
    T=<double *>malloc(lenx*sizeof(double))
    if lenx==1:
        T[0]=1
        return T
    for i from 0<=i<lenx:
        if x[i]<=-lengthtop/2.0:
            T[i]=(4.0*x[i]+lengthbase*2.0)/lb2mlt2
        elif x[i]<lengthtop/2.0:
            T[i]=2.0/(lengthbase+lengthtop)
        else:
            T[i]=(-4.0*x[i]+lengthbase*2.0)/lb2mlt2
    for i from 0<=i<lenx:
        if (x[i]<=-lengthbase/2) or (x[i]>=lengthbase/2):
            T[i]=0
    return T

cdef inline double mymax(double a, double b):
    if a>b:
        return a
    else:
        return b 

cdef inline double mymin(double a, double b):
    if a>b:
        return b
    else:
        return a 
    
    
def smearingmatrix(Py_ssize_t pixelmin, Py_ssize_t pixelmax, double beamcenter,
                    double pixelsize, double lengthbaseh, double lengthtoph,
                    double lengthbasev=0, double lengthtopv=0,
                    Py_ssize_t beamnumh=1024,Py_ssize_t beamnumv=1):
    """Calculate the smearing matrix for the given geometry.
    
    Inputs: (pixels and pixel coordinates are counted from 0. The direction
        of the detector is assumed to be vertical.)
        pixelmin: the smallest pixel to take into account
        pixelmax: the largest pixel to take into account
        beamcenter: pixel coordinate of the primary beam
        pixelsize: the size of pixels, in micrometers
        lengthbaseh: the length of the base of the horizontal beam profile
        lengthtoph: the length of the top of the horizontal beam profile
        lengthbasev: the length of the base of the vertical beam profile
        lengthtopv: the length of the top of the vertical beam profile
        beamnumh: the number of elementary points of the horizontal beam
            profile
        beamnumv: the number of elementary points of the vertical beam profile
            Give 1 if you only want to take the length of the slit into
            account.
    
    Output:
        The smearing matrix. This is an upper triangular matrix. Desmearing
        of a column vector of the measured intensities can be accomplished by
        multiplying by the inverse of this matrix.
    """
    cdef Py_ssize_t npixels
    cdef double *x
    cdef double * yb
    cdef double * xb    
    cdef double deltah,deltav,centerh,centerv
    cdef double *H
    cdef double *V
    cdef double center
    cdef Py_ssize_t i,j,ix,iy
    cdef np.ndarray[np.double_t,ndim=2] A
    cdef double tmp
    cdef long ind1
    cdef double prop
    cdef double p
    cdef double left
    # coordinates of the pixels
    npixels=pixelmax-pixelmin+1
    x=<double*>malloc(npixels*sizeof(double))
    for i from 0<=i<npixels:
        x[i]=pixelmin+i
    # horizontal and vertical coordinates of the beam-profile in mm.
    if beamnumv<1:
        beamnumv=1
    if beamnumh<1:
        beamnumh=1
    yb=<double *>malloc(beamnumh*sizeof(double))
    yb[0]=0
    xb=<double *>malloc(beamnumv*sizeof(double))
    xb[0]=0
    
    if beamnumh>1:
        left=-mymax(lengthbaseh,lengthtoph)/2.0
        deltah=-2*left/(beamnumh-1)
        for iy from 0<=iy<beamnumh:
            yb[iy]=left+iy*deltah
        centerh=2.0/(lengthbaseh+lengthtoph)
    else:
        deltah=1
        centerh=1
    if beamnumv>1:
        left=-mymax(lengthbasev,lengthtopv)/2.0
        deltav=-2*left/(beamnumv-1)
        for ix from 0<=ix<beamnumv:
            xb[ix]=left+ix*deltav
        centerv=2.0/(lengthbasev+lengthtopv)
    else:
        deltav=1
        centerv=1
    #beam profile vector (trapezoid centered at the origin. Only a half of it
    # is taken into account)
    H=ctrapezoidshapefunction(lengthbaseh,lengthtoph,yb,beamnumh)
    V=ctrapezoidshapefunction(lengthbasev,lengthtopv,xb,beamnumv)
    center=centerh*centerv
    # scale y to detector pixel units
    A=np.zeros((npixels,npixels),dtype=np.double)
    for i from 0<=i<npixels:
        A[i,i]+=center
        for ix from 0<=ix<beamnumv:
            for iy from 0<=iy<beamnumh:
                tmp=sqrt((<double>i-xb[ix]/pixelsize*1e3)*(<double>i-xb[ix]/pixelsize*1e3)+(yb[iy]/pixelsize*1e3)*(yb[iy]/pixelsize*1e3))
                ind1=<long>floor(tmp)
                prop=tmp-ind1
                if ind1>=npixels:
                    continue
                p=H[iy]*V[ix]
                A[i,ind1]+=p*(1-prop)
                if ind1<npixels-1:
                    A[i,ind1+1]+=p*prop
    A=A*deltah*deltav
    free(x)
    free(xb)
    free(yb)
    free(H)
    free(V)
    return A
    

def trapezoidshapefunction(double lengthbase,double lengthtop,x):
    """def trapezoidshapefunction(lengthbase,lengthtop,x):
        
    Return a trapezoid centered at zero
    
    Inputs:
        lengthbase: the length of the base
        lengthtop: the length of the top (normally smaller than lengthbase)
        x: the coordinates
    
    Output:
        the shape function in a numpy array.
    """
    x=np.array(x)
    if len(x)<2:
        return np.array(1)
    T=np.zeros(x.shape)
    indofflimits=(x<=-lengthbase/2.0)|(x>=lengthbase/2.0)
    indslopeleft=(x<=-lengthtop/2.0)
    indsloperight=(x>=lengthtop/2.0)
    indtop=(x<=lengthtop/2.0)&(x>=-lengthtop/2.0)
    T[indsloperight]=-4.0/(lengthbase**2-lengthtop**2)*x[indsloperight]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
    T[indtop]=2.0/(lengthbase+lengthtop)
    T[indslopeleft]=4.0/(lengthbase**2-lengthtop**2)*x[indslopeleft]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
    T[indofflimits]=0
    return T

def smearingmatrixgonio(double tthmin, double tthmax, Py_ssize_t Ntth,
                      np.ndarray[np.double_t, ndim=2] p,
                      np.ndarray[np.double_t,ndim=1] x,
                      np.ndarray[np.double_t, ndim=1] y, double L0):
    """def smearingmatrixgonio(tthmin,tthmax,Ntth,p,x,y,L0):

    Construct a smearing matrix for line focus, goniometer.
    
    Inputs:
        tthmin, tthmax, Ntth: two-theta scale. Ends are included.
        p: beam profile matrix (length: along rows. Height: along columns)
        x: coordinate vector of the beam length
        y: coordinate vector of the beam height
        L0: sample-to-detector distance (detector at 0 angles)
    
    Output:
        the smearing matrix of size Ntth x Ntth
        
    Notes: you usually would want to add a longer two-theta scale and later trim
        the matrix to avoid edge effects.
    """
    cdef np.ndarray[np.double_t, ndim=2] mat
    cdef double X,Y,P,L,TTH,prop,tthnew
    cdef double tmp
    cdef Py_ssize_t idxprev,ix,iy,itth,Nx,Ny
    
    mat=np.zeros((Ntth,Ntth),dtype=np.double)
    Nx=len(x)
    Ny=len(y)
    for itth from 0<=itth<Ntth:
        #column index in mat is itth
        TTH=tthmin+(tthmax-tthmin)/(Ntth-1)*itth
        L=L0*cos(TTH)
        for ix from 0<=ix<Nx:
            X=x[ix]/L
            for iy from 0<=iy<Ny:
                Y=y[iy]/L
                P=p[iy,ix]
                tthnew=atan(sqrt((tan(TTH)-Y)**2+X**2))
                tmp=(tthnew-tthmin)/(tthmax-tthmin)*(Ntth-1)
                idxprev=int(floor(tmp))
                prop=(tmp-idxprev)
                if idxprev>=0 and idxprev<Ntth:
                    mat[itth,idxprev]+=P*(1-prop)
                if idxprev+1>=0 and idxprev+1<Ntth:
                    mat[itth,idxprev+1]+=P*prop
    return mat

def smearingmatrixflat(double pixmin, Py_ssize_t Npix, double pixsize,
                      np.ndarray[np.double_t, ndim=2] p,
                      np.ndarray[np.double_t,ndim=1] x,
                      np.ndarray[np.double_t, ndim=1] y, double L0,callback=None):
    """def smearingmatrixflat(pixmin,Npix,pixsize,p,x,y,L0):

    Construct a smearing matrix for line focus, flat detector.
    
    Inputs:
        pixmin: pixel coordinate of the first point
        Npix: number of pixels.
        pixsize: the width of a pixel (mm)
        p: beam profile matrix (length: along rows. Height: along columns. I.e.
            in normal case, the matrix has more columns than rows).
        x: coordinate vector of the beam width (number of elements= number of rows in the matrix)
        y: coordinate vector of the beam length (number of elements= number of columns in the matrix)
        L0: sample-to-detector distance (detector at 0 angles)
        callback: callback function (will be called (pixmax-pixmin+1) times)
            during the calculation, if not None. Intended for eg. progress bars.
            
    
    Output:
        the smearing matrix of size Ntth x Ntth
        
    Notes:
        you usually would want to add a longer two-theta scale and later trim
            the matrix to avoid edge effects.
        pixel 0 corresponds to the primary beam.
        y is parallel with the length of the beam, x with its width. z points at
            the detector (which is parallel with x)
        
    """
    cdef np.ndarray[np.double_t, ndim=2] mat
    cdef double X,Y,P,prop,pixnew
    cdef double tmp,pix
    cdef double pixmax
    cdef Py_ssize_t idxprev,ix,iy,ipix,Nx,Ny
    #number of pixels
    pixmax=pixmin+Npix-1
    #create an empty matrix
    mat=np.zeros((Npix,Npix),dtype=np.double)
    Nx=len(x)
    Ny=len(y)
    for ipix from 0<=ipix<Npix: # for each pixel (defines the two-theta) in the original (unsmeared) curve:
        #calculate the pixel value for the ith pixel
        pix=pixmin+ipix*(pixmax-pixmin)/(Npix-1)
        if callback is not None: # do the callback function.
            callback.__call__()
        for ix from 0<=ix<Nx: #loop through the beam width (parallel to the detector, the smallest)
            #X=x[ix]/L0 #re-scale coordinate by the s-d distance
            X=x[ix]
            for iy from 0<=iy<Ny: # loop through the beam length (orthogonal to the detector)
                #Y=y[iy]/L0 #re-scale coordinate)
                Y=y[iy]
                P=p[ix,iy] #get the current element of the primary beam matrix
                #calculate the pixel coordinate into which the scattering from this beam point falls under two-theta (defined by pix and ipix)
                #pixnew=L0*atan(sqrt((pix*pixsize/L0-Y)**2+X**2))/pixsize
                pixnew=(sqrt(pixsize*pixsize*pix*pix-Y*Y)+X)/pixsize
                #calculate the index of pixnew -> tmp
                tmp=(pixnew-pixmin)/(pixmax-pixmin)*(Npix-1)
                idxprev=int(floor(tmp)) # index of the previous pixel
                prop=(tmp-idxprev) #difference in pixel coordinate from the previous pixel
                #interpolate linearly
                if idxprev>=0 and idxprev<Npix:
                    mat[idxprev,ipix]+=P*(1-prop)
                if idxprev+1>=0 and idxprev+1<Npix:
                    mat[idxprev+1,ipix]+=P*prop
    return mat

cdef void fastcubicbspline(double * x, double *y, double center, double stretch, Py_ssize_t N):
    cdef Py_ssize_t i
    cdef double curx
    for i from 0<=i<N:
        curx=(x[i]-center)/stretch
        if abs(curx)>2:
            y[i]=0
        elif abs(curx)>1:
            y[i]=1/6.*(2-curx)*(2-curx)*(2-curx)
        else:
            y[i]=2/3.-0.5*curx*curx*(2-curx)
    return
    
cdef void fastdotproduct(double *A, double *x, double *y, Py_ssize_t Nrows, Py_ssize_t Ncols):
    """A should be in rows first format, i.e. A<row><col>=A[row+col*Nrows]
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i from 0<=i<Nrows:
        y[i]=0
        for j from 0<=j<Ncols:
            y[i]+=A[i+j*Nrows]*x[j]
    return
    
#def indirectdesmearflat(np.ndarray [np.double_t, ndim=1] pix not None,
#                        np.ndarray[np.double_t, ndim=1] Intensity not None,
#                        np.ndarray[np.double_t, ndim=1] Error not None,
#                        Py_ssize_t Nknots, double stabparam,
#                        np.ndarray[np.double_t, ndim=2] mat not None,
#                        double Dmax, Py_ssize_t NMC=0, MCcallback=None):
#    """Do an indirect desmear (Glatter) on a scattering curve recorded
#    with a flat detector.
#    
#    Inputs:
#        pix: pixel coordinates of the intensity. Should be equally spaced. Pixel
#            zero corresponds to the primary beam position.
#        Intensity: intensity curve corresponding to pix
#        Error: error curve
#        Nknots: number of spline knots
#        stabparam: stabilization parameter
#        mat: smearing matrix
#        NMC: number of Monte-Carlo iterations for error propagation.
#        MCcallback: call-back routine for the Monte Carlo procedure
#        
#    Outputs: Idesm, [Edesm], mat
#        Idesm: desmeared intensity
#        Edesm: error of the desmeared intensity (only if NMC>=2)
#        mat: smearing matrix
#    """
#    cdef double minpix,maxpix
#    cdef double stretch_spline
#    cdef Py_ssize_t Npix
#    cdef double knot
#    cdef np.ndarray[np.double_t, ndim=2] splines
#    cdef np.ndarray[np.double_t, ndim=2] transsplines
#    cdef double *splines
#    cdef double *transsplines
#    cdef double *B
#    cdef double *d
#    cdef double *c
#    cdef double *pix1
#    cdef double *mat1
#    cdef double *Int1
#    cdef double *Err1
#    cdef double *K
#    
#    minpix=pix.min()
#    maxpix=pix.max()
#    Npix=len(pix)
#    
#    # each knot will have a spline in it. The abscissa of the splines will be pix.
#    splines=<double*>malloc(Npix*Nknots*sizeof(double))
#    transsplines=<double*>malloc(Npix*Nknots*sizeof(double))
#    pix1=<double*>malloc(Npix*sizeof(double))
#    Int1=<double*>malloc(Npix*sizeof(double))
#    Err1=<double*>malloc(Npix*sizeof(double))
#    mat1=<double*>malloc(Npix*Npix*sizeof(double))
#    for i from 0<=i<Npix:
#        pix1[i]=pix[i]
#        Int1[i]=Intensity[i]
#        Err1[i]=Error[i]
#        for j from 0<=j<Npix:
#            #i: columns. j: rows
#            mat1[j+i*Npix]=mat[j,i]
#    print "Calculating splines..."
#    #one knot is assigned a length of len(pixels/(Nknots-1)). We must stretch
#    # the spline function horizontally to overlap with its four neighbours.
#    stretch_spline=Npix/float(Nknots-1)
#    for i from 0<=i<Nknots:
#        fastcubicbspline(pix,splines+i*Npix,minpix+(maxpix-minpix)*i/(Nknots-1),stretch_spline,Npix)
#        fastdotproduct(mat1,splines+i*Npix,transsplines+i*Npix,Npix,Npix)
#    
#    print "Calculating matrices..."
#    d=<double*>malloc(Nknots*sizeof(double))
#    B=<double*>malloc(Nknots*Nknots*sizeof(double))
#    K=<double*>malloc(Nknots*Nknots*sizeof(double))
#    for i from 0<=i<Nknots:
#        d[i]=0
#        for j from 0<=j<Npix:
#            d[i]+=Int1[j]*transsplines[j+i*Npix]/(Err1[j]*Err1[j])
#        K[i+i*Nknots]=2
#        if i>0:
#            K[i+(i-1)*Nknots]=-1
#            K[i-1+i*Nknots]=-1
#        for j from i<=j<Nknots:
#            B[i+j*Nknots]=0
#            for k from 0<=k<Npix:
#                B[i+j*Nknots]+=transsplines[k+i*Npix]*transsplines[k+j*Npix]/(Err1[k]*Err1[k])
#            B[j+i*Nknots]=B[i+j*Nknots]
#    K[0]=1
#    K[Nknots*Nknots-1]=1
#    print "Solving..."
#    raise NotImplementedError
#    cs=[]
#    idesms=[]
#    mdps=[]
#    ncprimes=[]
#    for j in range(len(stabparams)):
#        relaxedGaussSeidel()
#        c=np.linalg.linalg.solve(B+stabparams[j]*K,d)
#        idesm=np.zeros(pix.shape)
#        for i in range(Nknots):
#            idesm+=c[i]*splines[:,i]
#        Ncprime=0
#        for i in range(Nknots-1):
#            Ncprime+=(c[i+1]-c[i])**2
#        
#        mdp=np.sum((Intensity-idesm)**2/Error**2)/len(Intensity)
#        cs.append(c)
#        idesms.append(idesm)
#        mdps.append(mdp)
#        ncprimes.append(Ncprime)
#    if np.isscalar(stabparam):
#        return idesms[0],np.sqrt(idesms[0]),mdps[0]
#    else:
#        return idesms,cs,mdps,B,d,K,splines,transsplines,ncprimes
#   
