from B1python import *

modegenerate=False
modeplot=True

Nrepetitions=60

coreradius0=10000000
layerdistance=48.68
layerthickness0=5
qrange=pylab.linspace(0.01,0.5,500)
nlayersmin=10
nlayersmax=100
corefluctuation=coreradius0*0.1
radiusfluctuation=3
thicknessfluctuation=0
shakecenter=1

if modegenerate:
    Ints=pylab.zeros(qrange.shape)
    for j in range(Nrepetitions):
        print "------------\nIteration #",j,"\n------------"
        coreradius=-1
        while coreradius<=0:
            coreradius=coreradius0+pylab.randn()*corefluctuation
        nlayers=nlayersmin+(nlayersmax-nlayersmin)*pylab.random()
        spheres=pylab.zeros((2*nlayers,6))
        for i in range(nlayers):
            layerthickness=-1
            while layerthickness<=0:
                layerthickness=layerthickness0+pylab.randn()*thicknessfluctuation
            xcent=pylab.randn()*shakecenter
            ycent=pylab.randn()*shakecenter
            zcent=pylab.randn()*shakecenter
            radfluct=radiusfluctuation*pylab.randn()
            spheres[2*i,0]=xcent
            spheres[2*i,1]=ycent
            spheres[2*i,2]=zcent
            spheres[2*i,3]=coreradius+i*layerdistance-layerthickness/2.0+radfluct
            spheres[2*i,4]=-1
            spheres[2*i,5]=0;
            spheres[2*i+1,0]=xcent
            spheres[2*i+1,1]=ycent
            spheres[2*i+1,2]=zcent
            spheres[2*i+1,3]=coreradius+i*layerdistance+0.5*layerthickness+radfluct
            spheres[2*i+1,4]=1
            spheres[2*i+1,5]=0;
        savespheres(spheres,'multilamellar_%03d.txt'%j)
        os.system('saxs multilamellar_%03d.txt saxs_lab1.imp multilamellar_%03d.calc.txt 2'% (j,j))
if modeplot:
    Ints=None
    for j in range(Nrepetitions):
        A=pylab.loadtxt('multilamellar_%03d.calc.txt' % j)
        if Ints is None:
            Ints=A[:,1]
            qs=2*pylab.pi*A[:,0]
        else:
            Ints+=A[:,1]
    print qs.shape
    print Ints.shape
    pylab.semilogy(qs,Ints)
    pylab.show()