from B1python import *

R0=1
dR=R0*0
d0=30
dd=d0*0
Nx=10
Ny=10
shake=0

dist=0

qrange=pylab.linspace(0.01,1,500)
spheres=pylab.zeros((Nx*Ny,6))
for i in range(Nx):
    for j in range(Ny):
        xcent=Nx*i+pylab.randn()*shake
        ycent=Ny*j+pylab.randn()*shake
        zcent=pylab.randn()*shake
        spheres[i*Nx+j,0]=xcent
        spheres[i*Nx+j,1]=ycent
        spheres[i*Nx+j,2]=zcent
        spheres[i*Nx+j,3]=R0+dR*pylab.randn()
        spheres[i*Nx+j,4]=1
        spheres[i*Nx+j,5]=0;
savespheres(spheres,'planarcrystal.txt')
Intensity=theorspheres(qrange,spheres)/(300**2/200**2)
pylab.semilogy(qrange,Intensity)
pylab.show()