import numpy as np
cimport numpy as np
from stdlib cimport *
import warnings

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
        return R**3/3
    else:
        return 1/q**3*(sin(q*R)-q*R*cos(q*R))

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
    