from B1python import *
import os
os.chdir('/home/andris/kutatas/jusifa/Projekte/2009/0921Bota')

data,dataerr,param=read2dintfile(54)
#testorigin(data[0],(param[0]['BeamPosX'],param[0]['BeamPosY']))
for p in range(len(param)):
    print p,param[p]['FSN'], param[p]['Title']
DATAIDX=0
print data[DATAIDX].max()
#mask=makemask(pylab.zeros(data[DATAIDX].shape),data[DATAIDX],'masktest.mat')
mask1=scipy.io.loadmat('processing/mask.mat')['mask4']
#testorigin(data[0],[100,100])
[q,Intensity,Error,Area]=radint(data[DATAIDX],dataerr[DATAIDX],param[DATAIDX]['EnergyCalibrated'],param[DATAIDX]['Dist'],
                param[DATAIDX]['PixelSize'],param[DATAIDX]['BeamPosX'],param[DATAIDX]['BeamPosY'],1-mask1)
hist=radhist(data[DATAIDX],param[DATAIDX]['EnergyCalibrated'],param[DATAIDX]['Dist'],
             param[DATAIDX]['PixelSize'],param[DATAIDX]['BeamPosX'],param[DATAIDX]['BeamPosY'],1-mask1,q,pylab.linspace(0,10,len(q)))
pylab.imshow(pylab.log(hist+1),origin='lower')#,extent=[q.min(),q.max(),Intensity.min(),Intensity.max()])
pylab.figure()
pylab.plot(q,Intensity)
pylab.show()
quit()

print Intensity
print Area
print Error
print Weight
print type(Area)
#pylab.show()