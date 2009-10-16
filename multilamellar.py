from B1python import *

Nrepetitions=60

coreradius0=10000000
layerdistance=60
layerthickness=10
qrange=pylab.linspace(0.01,0.5,500)
nlayers=30
corefluctuation=coreradius0*0.5
radiusfluctuation=1
spheres=pylab.zeros((2*nlayers,6))
Ints=pylab.zeros(qrange.shape)
for j in range(Nrepetitions):
    coreradius=coreradius0+pylab.randn()*corefluctuation
    for i in range(nlayers):
        xcent=pylab.randn()
        ycent=pylab.randn()
        zcent=pylab.randn()
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
    savespheres(spheres,'multilamellar_%03d.txt'%j)
    Intensity=theorspheres(qrange,spheres)/(300**2/200**2)
    Ints+=Intensity
Ints/=Nrepetitions
pylab.semilogy(qrange,Ints)
pylab.semilogy(qrange,Intensity)
pylab.show()