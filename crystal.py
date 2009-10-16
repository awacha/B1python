from B1python import *

R0=1
dR=R0*0
d0=30
dd=d0*0
Nx=7
Ny=7
Nz=7
shake=0


qrange=pylab.linspace(0.01,1,500)
spheres=pylab.zeros((Nx*Ny*Nz,6))
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            xcent=Nx*i+pylab.randn()*shake
            ycent=Ny*j+pylab.randn()*shake
            zcent=Nz*k+pylab.randn()*shake
            spheres[i*Nx+j*Ny+k,0]=xcent
            spheres[i*Nx+j*Ny+k,1]=ycent
            spheres[i*Nx+j*Ny+k,2]=zcent
            spheres[i*Nx+j*Ny+k,3]=R0+dR*pylab.randn()
            spheres[i*Nx+j*Ny+k,4]=1
            spheres[i*Nx+j*Ny+k,5]=0;
savespheres(spheres,'crystal.txt')
Intensity=theorspheres(qrange,spheres)/(300**2/200**2)
pylab.semilogy(qrange,Intensity)
pylab.show()