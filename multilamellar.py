from B1python import *

coreradius=1000
layerdistance=60
layerthickness=5
qrange=pylab.linspace(0.01,0.5,500)
nlayers=2
corefluctuation=0
radiusfluctuation=0
spheres=pylab.zeros((2*nlayers,6))
for i in range(nlayers):
    xcent=pylab.randn()*corefluctuation
    ycent=pylab.randn()*corefluctuation
    zcent=pylab.randn()*corefluctuation
    radfluct=radiusfluctuation*pylab.randn()
    spheres[2*i,0]=xcent
    spheres[2*i,1]=ycent
    spheres[2*i,2]=zcent
    spheres[2*i,3]=coreradius+i*layerdistance+radfluct
    spheres[2*i,4]=-1
    spheres[2*i,5]=0;
    spheres[2*i+1,0]=xcent
    spheres[2*i+1,1]=ycent
    spheres[2*i+1,2]=zcent
    spheres[2*i+1,3]=coreradius+i*layerdistance+layerthickness+radfluct
    spheres[2*i+1,4]=1
    spheres[2*i+1,5]=0;
Intensity=theorspheres(qrange,spheres)/(300**2/200**2)
pylab.semilogy(qrange,Intensity)
pylab.show()