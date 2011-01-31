import numpy as np
cimport numpy as np
from stdlib cimport *
import warnings
from scipy.integrate import quadrature

HC=12398.419

ctypedef struct Coordtype:
    double x
    double y
    double z

cdef extern from "stdlib.h":
    Py_ssize_t RAND_MAX
    Py_ssize_t rand()

cdef extern from "math.h":
    double sin(double)
    double cos(double)
    double sqrt(double)
    double fabs(double)
    double M_PI
    double exp(double)
    double log(double)

cdef inline double randn():
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
cdef inline double fellipsoid(double qx, double qy, double qz, double a, double b, double c):
    """Scattering factor of an ellipsoid
    
    Inputs:
        qx, qy, qz: x,y,z components of the scattering vector
        a,b,c: half-axes of the ellipsoid (in x, y and z direction, respectively)
    
    Output:
        the value of the scattering factor
    """
    return fsphere(sqrt((qx*a)**2+(qy*b)**2+(qz*c)**2),1)*a*b*c

cdef inline double fcylinder(double qx, double qy, double qz, double vx, double vy, double vz, double L, double R):
    """Scattering factor of a cylinder
    
    Inputs:
        qx, qy, qz: x,y and z component of the q vector
        vx, vy, vz: components of the director of the cylinder
        L: length of the cylinder
        R: radius of the cylinder
    
    Output:
        the value of the scattering factor
    """
    cdef double qax # axial component of the q vector
    cdef double qrad # radial component of q (perpendicular to the axial
    cdef double h
    cdef double term1,term2
    qax=fabs(qx*vx+qy*vy+qz*vz)
    qrad=sqrt(qx*qx+qy*qy+qz*qz-qax**2)
    h=L*0.5
    if (qax*h>0):
        term1=h*sin(qax*h)/(qax*h)
    else:
        term1=h
    if (R*qrad>0):
        term2=R/qrad*bessj1(R*qrad)
    else:
        term2=R*0.5
    return 4*M_PI*term1*term2
    
cdef inline double fsphere(double q, double R):
    """Scattering factor of a sphere
    
    Inputs:
        q: q value(s) (scalar or an array of arbitrary size and shape)
        R: radius (scalar)
        
    Output:
        the values of the scattering factor in an array of the same shape as q
    """
    if q==0:
        return 4*M_PI*R**3/3
    else:
        return 4*M_PI/q**3*(sin(q*R)-q*R*cos(q*R))

cdef inline double fsphere_normV(double q, double R):
    """Scattering factor of a sphere, normalized to volume
    
    Inputs:
        q: q value(s) (scalar or an array of arbitrary size and shape)
        R: radius (scalar)
        
    Output:
        the values of the scattering factor in an array of the same shape as q
    """
    if q==0:
        return 1
    else:
        return 3/(q*R)**3*(sin(q*R)-q*R*cos(q*R))
        
        
def theorsaxs2D(np.ndarray[np.double_t, ndim=2] data not None, double dist,
              double wavelength, Py_ssize_t Nx, Py_ssize_t Ny,
              double pixelsizex, double pixelsizey, double dx=0, double dy=0,
              bint headerout=False, Py_ssize_t FSN=0, Title='',verbose=True):
    """theorsaxs2D(np.ndarray[np.double_t, ndim=2] data not None,
                   double dist, double wavelength, Py_ssize_t Nx,
                   Py_ssize_t Ny, double pixelsizex, double pixelsizey,
                   double dx=0,double dy=0,bint headerout=False,
                   Py_ssize_t FSN=0,Title='',verbose=True):
    
    Calculate theoretical scattering of a sphere/cylinder composite structure
    in a virtual transmission SAXS setup.
    
    Inputs:
        data: numpy array representing the sphere structure data. It should have
            at least 5 columns, which are interpreted as x, y, z, R, rho, [L,
            vx, vy, vz]. These denote the center of the sphere/cylinder, its
            radius and average electron density. L is the length of the cylinder
            (if it is zero, the row is treated as a sphere) and vx, vy, vz
            describe the director of the cylinder. Superfluous columns are
            disregarded. x is horizontal, y is vertical. z points towards the
            detector.
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
        verbose: True if you want to have messages printed out after each column

    Output:
        a 2D numpy array containing the scattering image. The row coordinate is
        the vertical coordinate (rows are horizontal).

    """
    cdef double *x,*y,*z,*R,*rho # sphere data
    cdef double *cx,*cy,*cz,*cR,*crho,*cL,*cvx,*cvy,*cvz # cylinder data
    cdef np.ndarray[np.double_t, ndim=2] output
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t numspheres
    cdef Py_ssize_t numcylinders
    cdef double tmp,I
    cdef double qx,qy,qz,q
    cdef double sx,sy,sz
    
    if data.shape[1]<5:
        raise ValueError('the number of columns in the input matrix should be at least 5.')
    
    output=np.zeros((Ny,Nx),dtype=np.double)
    numcylinders=0
    if data.shape[1]==5:
        numspheres=data.shape[0]
    else:
        numspheres=0
        numcylinders=0
        for i from 0<=i<data.shape[0]:
            if data[i,5]>0:
                numcylinders+=1
            else:
                numspheres+=1
    x=<double*>malloc(sizeof(double)*numspheres)
    y=<double*>malloc(sizeof(double)*numspheres)
    z=<double*>malloc(sizeof(double)*numspheres)
    R=<double*>malloc(sizeof(double)*numspheres)
    rho=<double*>malloc(sizeof(double)*numspheres)
    cx=<double*>malloc(sizeof(double)*numcylinders)
    cy=<double*>malloc(sizeof(double)*numcylinders)
    cz=<double*>malloc(sizeof(double)*numcylinders)
    cR=<double*>malloc(sizeof(double)*numcylinders)
    crho=<double*>malloc(sizeof(double)*numcylinders)
    cL=<double*>malloc(sizeof(double)*numcylinders)
    cvx=<double*>malloc(sizeof(double)*numcylinders)
    cvy=<double*>malloc(sizeof(double)*numcylinders)
    cvz=<double*>malloc(sizeof(double)*numcylinders)
    j=0
    k=0
    if numcylinders==0:
        for i from 0<=i<data.shape[0]:
            x[i]=data[i,0]
            y[i]=data[i,1]
            z[i]=data[i,2]
            R[i]=data[i,3]
            rho[i]=data[i,4]
    else:
        for i from 0<=i<data.shape[0]:
            if data[i,5]>0:
                cx[k]=data[i,0]
                cy[k]=data[i,1]
                cz[k]=data[i,2]
                cR[k]=data[i,3]
                crho[k]=data[i,4]
                cL[k]=data[i,5]
                cvx[k]=data[i,6]
                cvy[k]=data[i,7]
                cvz[k]=data[i,8]
                k=k+1
            else:
                x[j]=data[i,0]
                y[j]=data[i,1]
                z[j]=data[i,2]
                R[j]=data[i,3]
                rho[j]=data[i,4]
                j=j+1
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
                for l from 0<=l<numcylinders:
                    tmp=(x[k]-cx[l])*qx+(y[k]-cy[l])*qy+(z[k]-cz[l])*qz
                    I+=fsphere(q,R[k])*fcylinder(qx,qy,qz,cvx[l],cvy[l],cvz[l],cL[l],cR[l])*2*rho[k]*crho[l]*cos(tmp)
            for l from 0<=l<numcylinders:
                I+=fcylinder(qx,qy,qz,cvx[l],cvy[l],cvz[l],cL[l],cR[l])**2*(crho[l]**2)
            output[j,i]=I
        if verbose:
            print "column",i,"/",Nx,"done"
    free(rho); free(R); free(z); free(y); free(x);
    if numcylinders>0:
        free(crho); free(cR); free(cz); free(cy); free(cx); free(cvx); free(cvy); free(cvz); free(cL);
    if headerout:
        return output,{'BeamPosX':(0.5*Ny+1)-dy/pixelsizey,\
                       'BeamPosY':(0.5*Nx+1)-dx/pixelsizex,'Dist':dist,\
                       'EnergyCalibrated':HC/wavelength,'PixelSizeX':pixelsizex,\
                       'PixelSizeY':pixelsizey,\
                       'PixelSize':sqrt(pixelsizex*pixelsizex+pixelsizey*pixelsizey),\
                       'FSN':FSN,'Title':Title}
    else:
        return output

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
    warnings.warn(DeprecationWarning("The use of Ctheorsphere2D is deprecated. Please use theorsaxs2D instead. In further versions, this function may be removed."))
    return theorsaxs2D(data,dist,wavelength,Nx,Ny,pixelsizex,pixelsizey,dx,dy,
                       headerout,FSN,Title)
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
                  np.ndarray[np.double_t, ndim=1] rho not None,
                  azimuthangle=None):
    """def Ctheorspheres(np.ndarray[np.double_t, ndim=1] qrange not None,
                  np.ndarray[np.double_t, ndim=1] x not None,
                  np.ndarray[np.double_t, ndim=1] y not None,
                  np.ndarray[np.double_t, ndim=1] z not None,
                  np.ndarray[np.double_t, ndim=1] R not None,
                  np.ndarray[np.double_t, ndim=1] rho not None,
                  azimuthangle=None):
    
    Calculate the theoretical scattering intensity of the sphere structure
    
    Inputs:
        qrange: np.ndarray of q values
        x, y, z, R, rho: one-dimensional np.ndarrays of the same
            lengths, containing x, y, z coordinates, radii and electron-
            densities of the spheres, respectively.
        azimuthangle: azimuth angle for q, in radian. The detector plane is the
            x-z plane. This angle starts from 0=x. Leave it None, if you want the
            intensity to be averaged over all orientations of q.
            
    Output:
        a vector of the same size that of qrange. It contains the scattering
        intensities.
    """
    cdef double I
    cdef Py_ssize_t i,j,k
    cdef Py_ssize_t lenq,lensphere
    cdef double q
    cdef double d
    cdef double factor1,factor
    cdef double cosalpha, sinalpha
    cdef int fixazimuth
    cdef double *myx
    cdef double *myy
    cdef double *myz
    cdef double *myR
    cdef double *myrho
    
    if azimuthangle is not None:
        cosalpha=cos(azimuthangle)
        sinalpha=sin(azimuthangle)
        fixazimuth=1
    else:
        cosalpha=0
        sinalpha=0
        fixazimuth=0
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
    
    myx=<double*>malloc(lensphere*sizeof(double))
    myy=<double*>malloc(lensphere*sizeof(double))
    myz=<double*>malloc(lensphere*sizeof(double))
    myR=<double*>malloc(lensphere*sizeof(double))
    myrho=<double*>malloc(lensphere*sizeof(double))
    
    for i from 0<=i<lensphere:
        myx[i]=x[i]
        myy[i]=y[i]
        myz[i]=z[i]
        myR[i]=R[i]
        myrho[i]=rho[i]
    
    Intensity=np.zeros(lenq,dtype=np.double)
    if fixazimuth:
        for i from 0<=i<lenq:
            I=0
            q=qrange[i]
            for j from 0<=j<lensphere:
                factor1=myrho[j]*fsphere(q,myR[j])
                I+=factor1**2
                for k from j<k<lensphere:
                    d=(myx[j]-myx[k])*cosalpha+(myz[j]-myz[k])*sinalpha
                    if (d*q==0):
                        factor=1
                    else:
                        factor=cos(d*q)
                    I+=myrho[k]*fsphere(q,myR[k])*factor1*factor*2
            Intensity[i]=I
    else:
        for i from 0<=i<lenq:
            q=qrange[i]
            I=0
            for j from 0<=j<lensphere:
                factor1=myrho[j]*fsphere(q,myR[j])
                I+=factor1**2
                for k from j<k<lensphere:
                    d=sqrt((myx[j]-myx[k])**2+(myy[j]-myy[k])**2+(myz[j]-myz[k])**2)
                    if (d*q==0):
                        factor=1
                    else:
                        factor=sin(d*q)/(d*q)
                    I+=myrho[k]*fsphere(q,myR[k])*factor1*factor*2
            Intensity[i]=I
    free(myrho)
    free(myx)
    free(myy)
    free(myz)
    free(myR)
    return Intensity            


cdef inline float bessj0(double x):
    """Returns the Bessel function J0 (x) for any real x.
    
    Taken from Numerical Recipes
    """
    cdef float ax
    cdef float z
    cdef double xx,y,ans,ans1,ans2
    ax=fabs(x)
    if (ax < 8.0):
        y=x*x;
        ans1=57568490574.0+y*(-13362590354.0+y*(651619640.7
            +y*(-11214424.18+y*(77392.33017+y*(-184.9052456)))));
        ans2=57568490411.0+y*(1029532985.0+y*(9494680.718
            +y*(59272.64853+y*(267.8532712+y*1.0))));
        ans=ans1/ans2;
    else:
        z=8.0/ax;
        y=z*z;
        xx=ax-0.785398164;
        ans1=1.0+y*(-0.1098628627e-2+y*(0.2734510407e-4
            +y*(-0.2073370639e-5+y*0.2093887211e-6)));
        ans2 = -0.1562499995e-1+y*(0.1430488765e-3
            +y*(-0.6911147651e-5+y*(0.7621095161e-6
            -y*0.934945152e-7)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
    return ans;

cdef inline float bessj1(double x):
    """Returns the Bessel function J1 (x) for any real x.
    
    Taken from Numerical Recipes
    """

    cdef float ax,z
    cdef double xx,y,ans,ans1,ans2
    ax=fabs(x)
    if (ax < 8.0):
        y=x*x;
        ans1=x*(72362614232.0+y*(-7895059235.0+y*(242396853.1
            +y*(-2972611.439+y*(15704.48260+y*(-30.16036606))))));
        ans2=144725228442.0+y*(2300535178.0+y*(18583304.74
            +y*(99447.43394+y*(376.9991397+y*1.0))));
        ans=ans1/ans2;
    else:
        z=8.0/ax;
        y=z*z;
        xx=ax-2.356194491;
        ans1=1.0+y*(0.183105e-2+y*(-0.3516396496e-4
            +y*(0.2457520174e-5+y*(-0.240337019e-6))));
        ans2=0.04687499995+y*(-0.2002690873e-3
            +y*(0.8449199096e-5+y*(-0.88228987e-6
            +y*0.105787412e-6)));
        ans=sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2);
        if (x < 0.0):
            ans = -ans;
    return ans
    
cdef inline double distspheres(double x1, double y1, double z1, double R1,
                               double x2, double y2, double z2, double R2):
    cdef double tmp
    tmp=sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)-R1-R2
    if tmp>0:
        return tmp
    else:
        return 0
        
def maxdistance(np.ndarray[np.double_t, ndim=2] data not None):
    cdef Py_ssize_t i,j
    cdef double maxdistance2
    cdef Py_ssize_t max1, max2
    cdef double dist2
    maxdistance2=0
    max1=0
    max2=0
    for i from 0<=i<data.shape[0]:
        for j from i<j<data.shape[0]:
            dist2=(data[i,0]-data[j,0])**2+(data[i,1]-data[j,1])**2+\
                  (data[i,2]-data[j,2])**2
            if dist2>maxdistance2:
                maxdistance2=dist2
                max1=i
                max2=j
    return sqrt(maxdistance2)

cdef inline Coordtype unidirC():
    cdef Coordtype ret
    cdef double phi
    cdef double rho
    phi=rand()/<double>RAND_MAX*2*M_PI
    ret.z=rand()/<double>RAND_MAX*2-1
    rho=sqrt(1-(ret.z)**2)
    ret.x=rho*cos(phi)
    ret.y=rho*sin(phi)
    return ret

def unidir(double len=1):
    cdef double x
    cdef double y
    cdef double z
    cdef Coordtype ret
    ret=unidirC()
    return (ret.x*len,ret.y*len,ret.z*len)

def grf_saxs2D(np.ndarray[np.double_t, ndim=2] data not None, double sigma,
               double dist, double wavelength, Py_ssize_t Nx, Py_ssize_t Ny,
               double pixelsizex, double pixelsizey, double dx=0, double dy=0,
               bint headerout=False, Py_ssize_t FSN=0, Title='', verbose=True):
    """grf_saxs2D(np.ndarray[np.double_t, ndim=2] data not None, double sigma,
                  double dist, double wavelength, Py_ssize_t Nx,
                  Py_ssize_t Ny, double pixelsizex, double pixelsizey,
                  double dx=0,double dy=0,bint headerout=False,
                  Py_ssize_t FSN=0,Title='',verbose=True):
    
    Calculate theoretical scattering of a Gaussian Random Field in a virtual
        transmission SAXS setup.
    
    Inputs:
        data: numpy array representing the Gaussian Random Field. Each row
            corresponds to a wave. The columns are interpreted as amplitude,
            x, y, z components of the wavenumber-vector, and phase. Superfluous
            columns are disregarded quietly. x is horizontal, y is vertical.
            z points towards the detector.
        sigma: correlation length
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
        verbose: True if you want to print messages after each column.
    Output:
        a 2D numpy array containing the scattering image. The row coordinate is
        the vertical coordinate (rows are horizontal).

    """
    cdef np.ndarray[np.double_t, ndim=2] output
    cdef Py_ssize_t i,j,k,l
    cdef Py_ssize_t Nwaves
    cdef double tmp
    cdef double qx,qy,qz,q
    cdef double sx,sy,sz
    cdef double I1,I2
    cdef double kq,sinphi,cosphi,sigma2
    cdef double qpluskexponent
    
    if data.shape[1]<5:
        raise ValueError('the number of columns in the input matrix should be at least 5.')
    Nwaves=data.shape[0]
    output=np.zeros((Ny,Nx),np.double)
    
    sigma2=sigma*sigma
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
            #q=sqrt(qx**2+qy**2+qz**2)
            I1=0
            I2=0
            for k from 0<=k<Nwaves:
                sinphi=sin(data[k,4])
                cosphi=cos(data[k,4])
                kq=data[k,1]*qx+data[k,2]*qy+data[k,3]*qz
                qpluskexponent=exp(-sigma2/2*((qx+data[k,1])**2+(qy+data[k,2])**2+(qz+data[k,3])**2))
                I1+=data[k,0]*sinphi*(exp(2*kq*sigma2)+1)*qpluskexponent
                I2+=data[k,0]*cosphi*(exp(2*kq*sigma2)-1)*qpluskexponent
            output[j,i]=I1**2+I2**2
        if verbose:
            print "column",i,"/",Nx,"done"
    if headerout:
        return output,{'BeamPosX':(0.5*Ny+1)-dy/pixelsizey,\
                       'BeamPosY':(0.5*Nx+1)-dx/pixelsizex,'Dist':dist,\
                       'EnergyCalibrated':HC/wavelength,'PixelSizeX':pixelsizex,\
                       'PixelSizeY':pixelsizey,\
                       'PixelSize':sqrt(pixelsizex*pixelsizex+pixelsizey*pixelsizey),\
                       'FSN':FSN,'Title':Title}
    else:
        return output
def grf_realize(np.ndarray[np.double_t, ndim=2] grf not None,
                double x0, double y0, double z0,
                double x1, double y1, double z1,
                Py_ssize_t Nx, Py_ssize_t Ny, Py_ssize_t Nz):
    """def grf_realize(np.ndarray[np.double_t, ndim=2] grf not None,
                double x0, double y0, double z0,
                double x1, double y1, double z1,
                Py_ssize_t Nx, Py_ssize_t Ny, Py_ssize_t Nz):
    
    Evaluate a Gaussian Random Field on a 3D lattice
    
    Inputs:
        grf: grf array. Each row corresponds to a wave. The columns are
            interpreted as amplitude, wavevector_x, wavevector_y, wavevector_z,
            phi.
        x0, y0, z0: coordinates of a corner of the box
        x1, y1, z1: coordinates of the opposite corner of the box
        Nx, Ny, Nz: number of voxels in each direction
        
    Output:
        the GRF matrix
    """
    cdef Py_ssize_t i,j,k,l,Nwaves
    cdef np.ndarray[np.double_t, ndim=3] output
    cdef double xstep, ystep, zstep
    cdef double *kx, *ky, *kz, *A, *phi
    cdef double tmp
    output=np.zeros((Nx,Ny,Nz),np.double)
    Nwaves=grf.shape[0]
    kx=<double*>malloc(Nwaves*sizeof(double))
    ky=<double*>malloc(Nwaves*sizeof(double))
    kz=<double*>malloc(Nwaves*sizeof(double))
    A=<double*>malloc(Nwaves*sizeof(double))
    phi=<double*>malloc(Nwaves*sizeof(double))
    
    for l from 0<=l<Nwaves:
        kx[l]=grf[l,1]
        ky[l]=grf[l,2]
        kz[l]=grf[l,3]
        A[l]=grf[l,0]
        phi[l]=grf[l,4]
    
    xstep=(x1-x0)/Nx
    ystep=(y1-y0)/Ny
    zstep=(z1-z0)/Nz
    for i from 0<=i<Nx:
        for j from 0<=j<Ny:
            for k from 0<=k<Nz:
                tmp=0
                for l from 0<=l<Nwaves:
                    tmp+=A[l]*sin(kx[l]*(x0+xstep*i)+ky[l]*(y0+ystep*j)+kz[l]*(z0+zstep*k)+phi[l])
                output[i,j,k]=tmp
    free(A); free(kx); free(ky); free(kz); free(phi)
    return output

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
    """Calculate the distance distribution function p(r) for a sphere.
    
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

def ddistellipsoid(double a, double b, double c,np.ndarray[np.double_t,ndim=1] d not None,Py_ssize_t NMC):
    """Calculate the distance distribution function p(r) for an ellipsoid.
    
    Inputs:
        a,b,c: radii
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
        if (xa*xa/a**2+ya*ya/b**2+za*za/c**2)>1:
            i-=1
            continue
        xb=rand()/<double>RAND_MAX*2*a-a
        yb=rand()/<double>RAND_MAX*2*b-b
        zb=rand()/<double>RAND_MAX*2*c-c
        if (xb*xb/a**2+yb*yb/b**2+zb*zb/c**2)>1:
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
    result[0]=myresult[0]/<double>NMC/(0.5*(myd[1]+myd[0])-myd[0])*(4*a*b*c*M_PI/3)**2
    for i from 1<=i<lend-1:
        result[i]=myresult[i]/<double>NMC/(0.5*(myd[i]+myd[i+1])-0.5*(myd[i]+myd[i-1]))*(4*a*b*c*M_PI/3)**2
    result[lend-1]=myresult[lend-1]/<double>NMC/(myd[lend-1]-0.5*(myd[lend-1]+myd[lend-2]))*(4*a*b*c*M_PI/3)**2
    free(myd)
    free(myresult)
    return result

    
    
def ddistbrick(double a, double b, double c,np.ndarray[np.double_t,ndim=1] d not None,Py_ssize_t NMC):
    """Calculate the distance distribution function p(r) for a rectangular brick.
    
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

def ddistgrf(double a, double b, double c, np.ndarray[np.double_t, ndim=2] grf not None, clip, np.ndarray[np.double_t,ndim=1] d not None,Py_ssize_t NMC,Py_ssize_t printevery=10000):
    """Calculate the distance distribution function p(r) for a Gaussian Random Field confined in a rectangular box.
    
    Inputs:
        a: length of one side
        b: length of the second side
        c: length of the third side
        grf: grf array. Each row corresponds to a wave. The columns are
            interpreted as amplitude, wavevector_x, wavevector_y, wavevector_z,
            phi.
        clip: Clipping threshold. GRF values below this are considered as "solid",
            above this "vacuum". If None, no clipping is performed.
        d: vector of the values for r
        NMC: number of Monte-Carlo steps
        printevery: print a message after every <printevery>-th run
    
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
    cdef double *myA
    cdef double *mykx
    cdef double *myky
    cdef double *mykz
    cdef double *myphi
    cdef Py_ssize_t Nwaves
    cdef double grfvala,grfvalb
    cdef int doclip
    cdef double myclip
    cdef Py_ssize_t counter_firstvacuum
    cdef Py_ssize_t counter_secondvacuum
    cdef Py_ssize_t counter_alltries
    cdef Py_ssize_t counter_outsided
    if clip is None:
        doclip=False
        myclip=0
    else:
        doclip=True
        myclip=<double>clip
    
    lend=len(d)
    Nwaves=grf.shape[0]
    result=np.zeros(lend,dtype=np.double)
    myd=<double*>malloc(sizeof(double)*lend)
    myresult=<double*>malloc(sizeof(double)*lend)
    myA=<double*>malloc(sizeof(double)*Nwaves)
    mykx=<double*>malloc(sizeof(double)*Nwaves)
    myky=<double*>malloc(sizeof(double)*Nwaves)
    mykz=<double*>malloc(sizeof(double)*Nwaves)
    myphi=<double*>malloc(sizeof(double)*Nwaves)
    
    for i from 0<=i<lend:
        myd[i]=d[i]
        myresult[i]=0

    for i from 0<=i<Nwaves:
        myA[i]=grf[i,0]
        mykx[i]=grf[i,1]
        myky[i]=grf[i,2]
        mykz[i]=grf[i,3]
        myphi[i]=grf[i,4]
    
    counter_firstvacuum=0
    counter_secondvacuum=0
    counter_alltries=0
    counter_outsided=0
    for i from 0<=i<NMC:
        if counter_alltries%printevery==0:
            print "Starting the %d. Monte Carlo step" % (counter_alltries+1)
        counter_alltries+=1
        xa=rand()/<double>RAND_MAX*2*a-a
        ya=rand()/<double>RAND_MAX*2*b-b
        za=rand()/<double>RAND_MAX*2*c-c
        grfvala=0
        for j from 0<=j<Nwaves:
            grfvala+=myA[j]*sin(mykx[j]*xa+myky[j]*ya+mykz[j]*za+myphi[j])
        if doclip:
            grfvala=(grfvala<=myclip) # zero if above, one if below.
            if grfvala<=0:
                counter_firstvacuum+=1
                continue
        xb=rand()/<double>RAND_MAX*2*a-a
        yb=rand()/<double>RAND_MAX*2*b-b
        zb=rand()/<double>RAND_MAX*2*c-c
        grfvalb=0
        for j from 0<=j<Nwaves:
            grfvalb+=myA[j]*sin(mykx[j]*xb+myky[j]*yb+mykz[j]*zb+myphi[j])
        if doclip:
            grfvalb=(grfvalb<=myclip) # zero if above, one if below.
            if grfvalb<=0:
                counter_secondvacuum+=1
                continue
        d1=sqrt((xa-xb)**2+(ya-yb)**2+(za-zb)**2)
        if (d1<myd[0]) or (d1>myd[lend-1]):
            i-=1
            counter_outsided+=1
            continue
        if d1>0.5*(myd[lend-2]+myd[lend-1]):
            myresult[lend-1]+=grfvala*grfvalb
        else:
            for j from 0<=j<lend:
                if d1<0.5*(myd[j]+myd[j+1]):
                    myresult[j]+=grfvala*grfvalb
                    break
    #now normalize by the bin width and the number of MC steps, then multiply by the square of the volume
    result[0]=myresult[0]/<double>NMC/(0.5*(myd[1]+myd[0])-myd[0])*(a*b*c)**2
    for i from 1<=i<lend-1:
        result[i]=myresult[i]/<double>NMC/(0.5*(myd[i]+myd[i+1])-0.5*(myd[i]+myd[i-1]))*(a*b*c)**2
    result[lend-1]=myresult[lend-1]/<double>NMC/(myd[lend-1]-0.5*(myd[lend-1]+myd[lend-2]))*(a*b*c)**2
    free(myd)
    free(myresult)
    free(myA)
    free(mykx)
    free(myky)
    free(mykz)
    free(myphi)
    print "Number of MC steps: ",NMC
    print "Clipping: ",myclip
    print "Number of done MC steps: ",counter_alltries,",",counter_alltries/<double>NMC*100-100,"% more."
    print "Number of firstvacuum steps: ",counter_firstvacuum,",",counter_firstvacuum/<double>counter_alltries*100,"%."
    print "Number of secondvacuum steps: ",counter_secondvacuum,",",counter_secondvacuum/<double>counter_alltries*100,"%."
    print "Number of outside steps: ",counter_outsided
    return result

def charfuncgrf(double R0, np.ndarray[np.double_t, ndim=2] grf not None,
                clip, np.ndarray[np.double_t,ndim=1] r not None,
                Py_ssize_t NMC,Py_ssize_t printevery=10000):
    """Calculate the characteristic function gamma_0(r) for a Gaussian Random Field confined in a rectangular box.
    
    Inputs:
        R0: radius of the sphere inside which the first point will be sampled
        grf: grf array. Each row corresponds to a wave. The columns are
            interpreted as amplitude, wavevector_x, wavevector_y, wavevector_z,
            phi.
        clip: Clipping threshold. GRF values below this are considered as "solid",
            above this "vacuum". If None, no clipping is performed.
        r: vector of the values for r
        NMC: number of Monte-Carlo steps
        printevery: print a message after every <printevery>-th run
    
    Outputs:
        a vector, of the same size as d. Normalized that its integral with respect
            to d is the square of the volume of the particle
    """
    cdef Py_ssize_t i,j,lend,k
    cdef double xa,ya,za,xb,yb,zb
    cdef double d1
    cdef double *myd
    cdef double *myresult
    cdef np.ndarray[np.double_t, ndim=1] result
    cdef double *myA
    cdef double *mykx
    cdef double *myky
    cdef double *mykz
    cdef double *myphi
    cdef Py_ssize_t Nwaves
    cdef double grfvala,grfvalb
    cdef int doclip
    cdef double myclip
    cdef Py_ssize_t counter_firstvacuum
    cdef Py_ssize_t counter_alltries
    cdef Coordtype coord
    if clip is None:
        doclip=False
        myclip=0
    else:
        doclip=True
        myclip=<double>clip
    
    lend=len(r)
    Nwaves=grf.shape[0]
    result=np.zeros(lend,dtype=np.double)
    myd=<double*>malloc(sizeof(double)*lend)
    myresult=<double*>malloc(sizeof(double)*lend)
    myA=<double*>malloc(sizeof(double)*Nwaves)
    mykx=<double*>malloc(sizeof(double)*Nwaves)
    myky=<double*>malloc(sizeof(double)*Nwaves)
    mykz=<double*>malloc(sizeof(double)*Nwaves)
    myphi=<double*>malloc(sizeof(double)*Nwaves)
    
    for i from 0<=i<lend:
        myd[i]=r[i]
        myresult[i]=0

    for i from 0<=i<Nwaves:
        myA[i]=grf[i,0]
        mykx[i]=grf[i,1]
        myky[i]=grf[i,2]
        mykz[i]=grf[i,3]
        myphi[i]=grf[i,4]
    
    for j from 0<=j<lend:
        counter_firstvacuum=0
        counter_alltries=0
        for i from 0<=i<NMC:
            if counter_alltries%printevery==0:
                print "Starting the %d. Monte Carlo step, r=%f (%d/%d)" % (counter_alltries+1,myd[j],j+1,lend)
            counter_alltries+=1
            coord=unidirC()
            d1=rand()/<double>RAND_MAX*R0
            xa=coord.x*d1
            ya=coord.y*d1
            za=coord.z*d1
            grfvala=0
            for k from 0<=k<Nwaves:
                grfvala+=myA[k]*sin(mykx[k]*xa+myky[k]*ya+mykz[k]*za+myphi[k])
            if doclip:
                grfvala=(grfvala<=myclip) # zero if above, one if below.
                if grfvala<=0:
                    counter_firstvacuum+=1
                    i-=1
                    continue
            coord=unidirC()
            xb=myd[j]*coord.x
            yb=myd[j]*coord.y
            zb=myd[j]*coord.z
            grfvalb=0
            for k from 0<=k<Nwaves:
                grfvalb+=myA[k]*sin(mykx[k]*xb+myky[k]*yb+mykz[k]*zb+myphi[k])
            if doclip:
                grfvalb=(grfvalb<=myclip) # zero if above, one if below.
            myresult[j]+=grfvala*grfvalb
        print "Number of MC steps: ",NMC
        print "Clipping: ",myclip
        print "Number of done MC steps: ",counter_alltries,",",counter_alltries/<double>NMC*100-100,"% more."
        print "Number of firstvacuum steps: ",counter_firstvacuum,",",counter_firstvacuum/<double>counter_alltries*100,"%."
    #now normalize by the bin width and the number of MC steps, then multiply by the square of the volume
    for i from 0<=i<lend:
        result[i]=myresult[i]
    free(myd)
    free(myresult)
    free(myA)
    free(mykx)
    free(myky)
    free(mykz)
    free(myphi)
    return result
def packspheres(Py_ssize_t N,double R,double dR):
    cdef double *x
    cdef double *y
    cdef double *z
    cdef double vx
    cdef double vy
    cdef double vz
    cdef Coordtype v
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef double *r
    cdef double l
    cdef double h
    cdef double d
    cdef double d1
    cdef np.ndarray[np.double_t,ndim=2] result
    
    x=<double*>malloc(sizeof(double)*N)
    y=<double*>malloc(sizeof(double)*N)
    z=<double*>malloc(sizeof(double)*N)
    r=<double*>malloc(sizeof(double)*N)
    for i from 0<=i<N:
        r[i]=-1;
        while (r[i]<=0):
           r[i]=dR*randn()+R
        if i==0:
            x[i]=0
            y[i]=0
            z[i]=0
            continue
        elif i==1:
            x[i]=r[i-1]+r[i]
            y[i]=0
            z[i]=0
        else:
            # sample an isotropic direction of the new sphere.
            v=unidirC()
            #put the sphere into the origin
            d=0 
            for j from 0<=j<i:
                l=v.x*x[j]+v.y*y[j]+v.z*z[j]
                if l<0:
                    continue
                h=sqrt(x[j]*x[j]+y[j]*y[j]+z[j]*z[j]-l*l)
                if h>r[j]+r[i]:
                    continue
                d1=l+sqrt((r[j]+r[i])**2-h**2)
                if d1>d:
                    d=d1
            x[i]=d*v.x
            y[i]=d*v.y
            z[i]=d*v.z
        
        # we now have the coordinates and radii of i+1 spheres. Move the whole 
        # cluster that its center of gravity (COG) coincides with the origin.
        # To accomplish this, a few remarks:
        # 1) we assume, that the COG of the original structure (without the
        #    newest sphere) is the origin.
        # 2) The vector to the new origin is: v/(i+1), where v is (x[i],y[i],z[i])
        # 3) all spheres should be moved by -v/(i+1) to move the whole thing
        # into the origin.
        vx=x[i]/(i+1)
        vy=y[i]/(i+1)
        vz=z[i]/(i+1)
        for j from 0<=j<=i:
            x[j]-=vx
            y[j]-=vy
            z[j]-=vz
    result=np.zeros((N,4))
    for i from 0<=i<N:
        result[i,0]=x[i]
        result[i,1]=y[i]
        result[i,2]=z[i]
        result[i,3]=r[i]
    free(x)
    free(y)
    free(z)
    free(r)
    return result

def structurefactor(np.ndarray[np.double_t,ndim=2] points,np.ndarray[np.double_t,ndim=1] qrange):
    cdef double *x
    cdef double *y
    cdef double *z
    cdef double q
    cdef double qd
    cdef Py_ssize_t N
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef double S
    cdef np.ndarray[np.double_t, ndim=1] output
    
    N=points.shape[0]
    x=<double*>malloc(sizeof(double)*N)
    y=<double*>malloc(sizeof(double)*N)
    z=<double*>malloc(sizeof(double)*N)
    output=np.zeros(len(qrange))
    
    for i from 0<=i<N:
        x[i]=points[i,0]
        y[i]=points[i,1]
        z[i]=points[i,2]
    for k from 0<=k<qrange.shape[0]:
        q=qrange[k]
        S=0
        for j from 0<=j<N:
            S+=1
            for i from 0<=i<j:
                qd=q*sqrt((x[i]-x[j])**2+(y[i]-y[j])**2+(z[i]-z[j])**2)
                if qd==0:
                    S+=2
                else:
                    S+=2*sin(qd)/qd
        output[k]=S/N
    return output
    
def _intellipsoid(double theta,double q,double R,double nu):
    return sin(theta)*fsphere_normV(q,R*sqrt(nu*nu*cos(theta)**2+sin(theta)**2))**2
    
def ellipsoid_scatter(np.ndarray[np.double_t, ndim=1] q,double R,double nu,double drho=1):
    """Calculate the isotropic scattering of an ellipsoid of revolution
    
    Inputs:
        q: q-range
        R: half length of axis of revolution
        nu: ratio of the length of transverse axes and longitudinal axis.
        drho: scattering contrast between the particle and its environment
        
    Outputs:
        the intensity curve
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N
    cdef double scaling
    
    scaling=(4*M_PI*R**3/3.0*drho)**2
    
    N=len(q)
    output=np.zeros(len(q),dtype=np.double)
    
    for i from 0<=i<N:
        output[i]=scaling*quadrature(_intellipsoid,0,M_PI/2,(q[i],R,nu),vec_func=False)[0]
    return output