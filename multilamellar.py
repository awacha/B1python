from B1python import *

coreradius=1000
layerdistance=60
qrange=pylab.linspace(0.05,0.5,1000)
nspheres=100
corefluctuation=2
radiusfluctuation=1
spheres=pylab.zeros((nspheres,4))
for i in range(nspheres):
    spheres[i,0]=pylab.randn()*corefluctuation
    spheres[i,1]=pylab.randn()*corefluctuation
    spheres[i,2]=pylab.randn()*corefluctuation
    spheres[i,3]=coreradius+i*layerdistance+radiusfluctuation*pylab.randn()
Intensity=theorspheres(qrange,spheres)
pylab.loglog(qrange,Intensity)
pylab.show()