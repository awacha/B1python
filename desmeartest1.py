from B1python import *
import time

print 'Calculating desmear matrix according to the new method'
t0=time.time()
B=smearingmatrix(200,900,207,52,22,16,0,0,200,0)
t1=time.time()
print 'It took ',t1-t0,'seconds'
print 'Calculating desmear matrix according to the old method'
t2=time.time()
A=smearingmatrix(200,900,207,52,22,16,1,0,200,20)
t3=time.time()
print 'It took ',t3-t2,'seconds'
print 'Done'
pylab.imshow(A)
pylab.title('A')
pylab.gcf().show()
pylab.figure()
pylab.imshow(B)
pylab.title('B')
pylab.gcf().show()