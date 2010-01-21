#utils2d.py

import pylab
import numpy as np
import scipy.optimize
import types

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

def polartransform(data,r,phi,origx,origy):
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
    pdata=np.zeros((len(phi),len(r)))
    for i in range(len(phi)):
        for j in range(len(r)):
            x=origx-1+r[j]*np.cos(phi[i]);
            y=origy-1+r[j]*np.sin(phi[i]);
            if (x>=0) and (y>=0) and (x<data.shape[0]) and (y<data.shape[1]):
                pdata[i,j]=data[x,y];
    return pdata
def findbeam_gravity(data,mask):
    # for each row and column find the center of gravity
    data1=data.copy() # take a copy, because elements will be tampered
                      # with
    data1[mask==0]=0 # set masked elements to zero

    pylab.imshow(data1) # show the matrix
    pylab.gcf().show() #
    # vector of x (row) coordinates
    x=np.arange(data1.shape[0])
    # vector of y (column) coordinates
    y=np.arange(data1.shape[1])

    # two column vectors, both containing ones. The length of onex and
    # oney corresponds to length of x and y, respectively.
    onex=np.ones((len(x),1))
    oney=np.ones((len(y),1))
    # Multiply the matrix with x. Each element of the resulting column
    # vector will contain the center of gravity of the corresponding row
    # in the matrix, multiplied by the "weight". Thus: nix_i=sum_j( A_ij
    # * x_j). If we divide this by spamx_i=sum_j(A_ij), then we get the
    # center of gravity. The length of this column vector is len(y).
    nix=np.dot(data1,x).flatten()
    spamx=np.dot(data1,onex).flatten()
    # indices where both nix and spamx is nonzero.
    goodx=((nix!=0) & (spamx!=0))
    # trim y, nix and spamx by goodx, eliminate invalid points.
    y1=y[goodx]
    nix=nix[goodx]
    spamx=spamx[goodx]

    # now do the same for the column direction.
    niy=np.dot(data1.T,y).flatten()
    spamy=np.dot(data1.T,oney).flatten()
    goody=((niy!=0) & (spamy!=0))
    x1=x[goody]
    niy=niy[goody]
    spamy=spamy[goody]
    # column coordinate of the center in each row will be contained in
    # ycent, the row coordinate of the center in each column will be
    # in xcent.
    ycent=nix/spamx
    xcent=niy/spamy
    pylab.figure()
    pylab.plot(x1,xcent,'.',label='xcent')
    pylab.plot(y1,ycent,'.',label='ycent')
    pylab.gcf().show()
    # return the mean values as the centers.
    return xcent.mean(),ycent.mean()
def findbeam_slices(data,orig_initial,mask=None,maxiter=0):
    """Find beam center
    
    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.fmin
    Output:
        a vector of length 2 with the x and y coordinates of the origin.
    """
    print "Finding beam, please be patient..."
    orig=np.array(orig_initial)
    if mask is None:
        mask=np.ones(data.shape)
    def targetfunc(orig,data,mask):
        #integrate four sectors
        print "integrating... (for finding beam)"
        c1,nc1=imageint(data,orig,mask,35,20)
        c2,nc2=imageint(data,orig,mask,35+90,20)
        c3,nc3=imageint(data,orig,mask,35+180,20)
        c4,nc4=imageint(data,orig,mask,35+270,20)
        # the common length is the lowest of the lengths
        last=min(len(c1),len(c2),len(c3),len(c4))
        # first will be the first common point: the largest of the first
        # nonzero points of the integrated data
        first=np.array([pylab.find(nc1!=0).min(),
                           pylab.find(nc2!=0).min(),
                           pylab.find(nc3!=0).min(),
                           pylab.find(nc4!=0).min()]).max()
        ret= np.array(((c1[first:last]-c3[first:last])**2+(c2[first:last]-c4[first:last])**2)/(last-first))
        print "orig:",orig[0],orig[1]
        print "last-first:",last-first
        print "sum(ret):",ret.sum()
        return ret
    orig=scipy.optimize.leastsq(targetfunc,np.array(orig_initial),args=(data,1-mask),maxfev=maxiter,epsfcn=0.0001)
    return orig[0]
def findbeam_azimuthal(data,orig_initial,mask=None,maxiter=100,Ntheta=50,dmin=0,dmax=np.inf):
    """Find beam center using azimuthal integration
    
    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column)
            coordinates of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it
            should be of the same size as data. Nonzero means non-masked.
        maxiter: maximum number of iterations for scipy.optimize.fmin
        Ntheta: the number of theta points for the azimuthal integration
        dmin: pixels nearer to the origin than this will be excluded from
            the azimuthal integration
        dmax: pixels farther from the origin than this will be excluded from
            the azimuthal integration
    Output:
        a vector of length 2 with the x and y coordinates of the origin.
    """
    print "Finding beam, please be patient..."
    orig=np.array(orig_initial)
    if mask is None:
        mask=np.ones(data.shape)
    def targetfunc(orig,data,mask):
        def sinfun(p,x,y):
            return (y-np.sin(x+p[1])*p[0]-p[2])/np.sqrt(len(x))
        t,I,a=azimintpix(data,None,orig,mask,Ntheta,dmin,dmax)
        if len(a)>(a>0).sum():
            raise ValueError,'findbeam_azimuthal: non-complete azimuthal average, please consider changing dmin, dmax and/or orig_initial!'
        p=((I.max()-I.min())/2.0,t[I==I.max()][0],I.mean())
        p=scipy.optimize.leastsq(sinfun,p,(t,I))[0]
        print "findbeam_azimuthal: orig=",orig,"amplitude=",abs(p[0])
        return abs(p[0])
    orig1=scipy.optimize.fmin(targetfunc,np.array(orig_initial),args=(data,1-mask),maxiter=maxiter)
    return orig1
def findbeam_semitransparent(data,pri):
    """Find beam with 2D weighting of semitransparent beamstop area

    Inputs:
        data: scattering matrix
        pri: list of four: [x0,y0,x1,y1] for the corners of the beam
            area under the semitransparent beamstop

    Outputs: bcx,bcy
        the x and y coordinates of the primary beam
    """
    Y,X=np.meshgrid(np.arange(data.shape[1]),
                       np.arange(data.shape[0]))
    indices=((X<=pri[2]) & (X>=pri[0]) & (Y<=pri[3]) & (Y>=pri[1]))
    d=data[indices]
    x=X[indices]
    y=Y[indices]
    bcx=np.sum(d*x)/np.sum(d)
    bcy=np.sum(d*y)/np.sum(d)
    return bcx,bcy
def azimintpix(data,error,orig,mask,Ntheta=100,dmin=0,dmax=np.inf):
    """Perform azimuthal integration of image.

    Inputs:
        data: matrix to average
        error: error matrix. If not applicable, set it to None
        orig: vector of beam center coordinates, starting from 1.
        mask: mask matrix; 1 means masked, 0 means non-masked
        Ntheta: number of desired points on the abscissa

    Outputs: theta,I,[E],A
        theta: theta-range, in radians
        I: intensity points
        E: error values (returned only if the "error" argument was not None)
        A: effective area points
    """
    # create the distance matrix: the distances of the pixels from the origin,
    # expressed in pixel units.
    Y,X=np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))
    X=X-orig[0]+1
    Y=Y-orig[1]+1
    D=np.sqrt(X**2+Y**2)
    Phi=np.arctan2(Y,X) # the angle matrix

    # remove invalid pixels (masked or falling outside [dmin,dmax])
    valid=((mask==0)&(D<=dmax)&(D>=dmin))
    d=D[valid]
    dat=data[valid]
    if error is not None:
        err=error[valid]
    phi=Phi[valid]

    theta=np.linspace(0,2*np.pi,Ntheta) # the abscissa of the results
    I=np.zeros(theta.shape) # vector of intensities
    A=np.zeros(theta.shape) # vector of effective areas
    if error is not None:
        E=np.zeros(theta.shape)
        for i in range(len(dat)):
            if (np.isfinite(err[i])) and (err[i]>0):
                index=np.floor(phi[i]/(2*np.pi)*Ntheta)
                I[index]+=dat[i]/(err[i]**2)
                E[index]+=1/(err[i]**2)
                A[index]+=1
        I[A>0]=I[A>0]/E[A>0]
        E=np.sqrt(1/E)
        return theta,I,E,A
    else: # if error is None
        for i in range(len(dat)):
            index=np.floor(phi[i]/(2*np.pi)*Ntheta)
            I[index]+=dat[i]
            A[index]+=1
        I[A>0]=I[A>0]/A[A>0]
        return theta,I,A
def imageint(data,orig,mask,fi=None,dfi=None):
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
    Y,X=np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]))
    X=X-orig[0]+1
    Y=Y-orig[1]+1
    D=np.sqrt(X**2+Y**2)
    d=D[mask==0]
    dat=data[mask==0]
    if (fi is not None) and (dfi is not None):
        x=X[mask==0]
        y=Y[mask==0]
        phi=np.fmod(np.arctan2(y,x)-np.pi/180.0*fi+10*np.pi,2*np.pi)
        d=d[phi<=dfi*np.pi/180.0]
        dat=dat[phi<=dfi*np.pi/180.0]
    C=np.zeros(np.ceil(d.max()))
    NC=np.zeros(np.ceil(d.max()))
    for i in range(len(dat)):
        C[np.floor(d[i])]+=dat[i]
        NC[np.floor(d[i])]+=1
    C[NC>0]=C[NC>0]/NC[NC>0];
    return C,NC
def sectint(data,fi,orig,mask):
    """Calculate sector-delimited radial average
    
    Inputs:
        data: matrix to average
        fi: a vector of length 2 of starting and ending angles, in degree.
        orig: the origin. Coordinates are counted from 1.
        mask: mask matrix

    Outputs:
        vector of integrated values
        vector of effective areas
        
    Note: based on the work of Mika Torkkeli
    """
    fi=np.array(fi)
    return imageint(data,orig,mask,fi=fi.min(),dfi=fi.max()-fi.min())
def radint(data,dataerr,energy,distance,res,bcx,bcy,mask,q=None,a=None,shutup=True):
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
        q: the q points at which the integration is requested. If None, the
            q-range is automatically guessed from the energy, the distance and
            the mask. It should be defined in 1/Angstroems.
        a: limiting angles. If None (default), simple radial integration
            is performed. If it is a list of length=2
        shutup: if True, work quietly (do not print messages).
        
    Outputs: four ndarrays.
        the q vector
        the intensity vector
        the error vector
        the area vector
    """
    if type(res)!=types.ListType:
        res=[res,res];
    if len(res)==1:
        res=[res[0], res[0]]
    if len(res)>2:
        raise ValueError('res should be a scalar or a nonempty vector of length<=2')
    if data.shape!=dataerr.shape or data.shape!=mask.shape:
        raise ValueError('data, dataerr and mask should be of the same shape')
    M=data.shape[0] # number of rows
    N=data.shape[1] # number of columns
    
    if not shutup:
        print "Creating D matrix...",
    # Creating D matrix which is the distance of the sub-pixels from the origin.
    Y,X=np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]));
    D=np.sqrt((res[0]*(X-bcx))**2+
                 (res[1]*(Y-bcy))**2)
    if not shutup:
        print "done"
        print "Calculating q-matrix...",
    # Q-matrix is calculated from the D matrix
    q1=4*np.pi*np.sin(0.5*np.arctan(D/float(distance)))*energy/float(HC)
    if not shutup:
        print "done"
        print "Masking...",
    # eliminating masked pixels
    data=data[mask==0]
    dataerr=dataerr[mask==0]
    q1=q1[mask==0]
    #del datalin
    #del dataerrlin
    #del masklin
    #del qlin
    if not shutup:
        print "done"
    # if the q-scale was not supplied, create one.
    if q is None:
        if not shutup:
            print "Creating q-scale...",
        qmin=min(q1) # the lowest non-masked q-value
        qmax=max(q1) # the highest non-masked q-value
        #qstep=(qmax-qmin)/10
        qstep=(qmax-qmin)/(np.sqrt(M*M+N*N))
        q=np.arange(qmin,qmax,qstep)
        if not shutup:
            print "done"
    else:
        q=np.array(q)
    # initialize the output vectors
    Intensity=np.zeros(q.size)
    Error=np.zeros(q.size)
    Area=np.zeros(q.size)
    # square the error
    dataerr=dataerr**2
    if not shutup:
        print "Integrating..."
    # set the bounds of the q-bins in qmin and qmax
    qmin=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmin.insert(0,q[0])
    qmin=np.array(qmin)
    qmax=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmax.append(q[-1])
    qmax=np.array(qmax)
    # go through every q-bin
    for l in range(len(q)):
        indices=((q1<=qmax[l])&(q1>qmin[l]) & (np.isfinite(dataerr)) & (dataerr>0)) # the indices of the pixels which belong to this q-bin
        Intensity[l]=np.sum((data[indices])/(dataerr[indices]**2)) # sum the intensities weighted by 1/sigma**2
        Error[l]=1/np.sum(1/(dataerr[indices]**2)) # error of the weighted average
#        Intensity[l]=np.sum(data[indices])
#        Error[l]=np.sum(dataerr[indices]**2)
        Area[l]=np.sum(indices) # collect the area
        # normalization by the area
        Intensity[l]=Intensity[l]*Error[l] # Error[l] is 1/sum_i(1/sigma^2_i)
        Error[l]=np.sqrt(Error[l])
#        Intensity[l]=Intensity[l]/Area[l]
#        Error[l]=np.sqrt(Error[l])/Area[l]
    if not shutup:
        print "done"
    
    return q,Intensity,Error,Area # return
    
def calculateDmatrix(mask,res,bcx,bcy):
    """Calculate distances of pixels from the origin
    
    Inputs:
        mask: mask matrix (only its shape is used)
        res: pixel size in mm-s. Can be a vector of length 2 or a scalar
        bcx: Beam center in pixels, in the row direction
        bcy: Beam center in pixels, in the column direction
        
    Output:
        A matrix of the shape of <mask>. Each element contains the distance
        of the centers of the pixels from the origin (bcx,bcy), expressed in
        mm-s.
    """
    if type(res)!=types.ListType:
        res=[res]
    if len(res)<2:
        res=res*2
    Y,X=np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]));
    D=np.sqrt((res[0]*(X-bcx))**2+
                 (res[1]*(Y-bcy))**2)
    return D
