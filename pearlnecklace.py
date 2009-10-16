from B1python import *

R0pearls=10
dRpearls=R0pearls*0
d0pearls=20
ddpearls=d0pearls*0
Npearls=100
shakepearls=3
dist=0

qrange=pylab.linspace(0.01,1,3000)
spheres=pylab.zeros((Npearls,6))
for i in range(Npearls):
    xcent=dist+pylab.randn()*shakepearls
    ycent=pylab.randn()*shakepearls
    zcent=pylab.randn()*shakepearls
    dist=dist+d0pearls+ddpearls*pylab.randn()
    spheres[i,0]=xcent
    spheres[i,1]=ycent
    spheres[i,2]=zcent
    spheres[i,3]=R0pearls+dRpearls*pylab.randn()
    spheres[i,4]=1
    spheres[i,5]=0;
for i in range(Npearls):
    spheres[i,0]=spheres[i,0]-dist/2.0
savespheres(spheres,'pearlnecklace.txt')
Intensity=theorspheres(qrange,spheres)/(300**2/200**2)
pylab.semilogy(qrange,Intensity)
pylab.show()