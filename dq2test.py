import pylab
q=pylab.linspace(0,1,1000)
energy=8333 # eV
dist=935e7 # A
HC=12398.419 #eV*A
qdQ2=pylab.zeros(q.shape)
for l in range(len(q)):
    tg2theta=pylab.tan(2*pylab.arcsin(q[l]*HC/(4*pylab.pi*energy)))
    qdQ2[l]=q[l]*(2*pylab.pi*energy/(HC*dist))**2*(2+tg2theta**2+2*pylab.sqrt(1+tg2theta**2))/((1+tg2theta**2+pylab.sqrt(1+tg2theta**2))**2*pylab.sqrt(1+tg2theta**2))
pylab.plot(q,qdQ2)
pylab.show()    
