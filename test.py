from B1python import *
import os
os.chdir('testdata')
data,dataerr,param=read2dintfile(range(100,115))
#testorigin(data[0],(param[0]['BeamPosX'],param[0]['BeamPosY']))
for p in range(len(param)):
    print p,param[p]['FSN'], param[p]['Title']

#mask=makemask(pylab.zeros(data[6].shape),data[6],'masktest.mat')
mask1=scipy.io.loadmat('masktest.mat')['mask']
#testorigin(data[0],[100,100])
a=radint2(data[6],dataerr[6],param[6]['EnergyCalibrated'],param[6]['Dist'],
                param[6]['PixelSize'],param[6]['BeamPosX'],param[6]['BeamPosY'],mask1)
print dir(a.Area)
print type(a.Area)
#pylab.show()