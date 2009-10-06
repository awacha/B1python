from B1python import *

coreradius=1000
layerdistance=60
qrange=pylab.linspace(0.01,0.5,500)
nspheres=1000
corefluctuation=10
radiusfluctuation=2
spheres=pylab.zeros((nspheres,6))
for i in range(nspheres):
    spheres[i,0]=pylab.randn()*corefluctuation
    spheres[i,1]=pylab.randn()*corefluctuation
    spheres[i,2]=pylab.randn()*corefluctuation
    spheres[i,3]=coreradius+i*layerdistance+radiusfluctuation*pylab.randn()
    spheres[i,4]=pow(-1,(nspheres-i+1))
    spheres[i,5]=0;
Intensity=theorspheres(qrange,spheres)
pylab.semilogy(qrange,Intensity)
pylab.show()