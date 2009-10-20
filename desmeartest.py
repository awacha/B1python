from B1python import *

datatrip=pylab.loadtxt('/home/andris/labor/waxs/Szilvi/TRIPWJ15.P00')[1:]
dataagst=pylab.loadtxt('/home/andris/labor/waxs/Szilvi/AGSTWJ15.P00')[1:]
pixels=pylab.arange(len(datatrip))
#p1=findpeak(pixels,datatrip)
#p2=findpeak(pixels,datatrip)
#p3=findpeak(pixels,datatrip)
#pylab.close('all')
#p4=findpeak(pixels,dataagst)
#p5=findpeak(pixels,dataagst)

print waxscalib([149.172,311.142,351.325,148.092,302.530],
          [2*pylab.pi*0.21739,2*pylab.pi*0.25641,2*pylab.pi*0.27027,
           2*pylab.pi*10/48.68,2*pylab.pi*11/48.68])

#x,A,s,M=directdesmear(data,smoothing=1e6,pixelmin=0,pixelmax=1023,
#                      beamcenter=206.55,pixelsize=51.9641,lengthbaseh=22,
#                      lengthtoph=17,lengthbasev=10e-3,lengthtopv=0,
#                      gui=False,smoothlow=1e3,smoothhi=1e9,smoothingmode='log',
#                      beamnumv=None)
#pylab.gcf().show()
#pylab.figure()
#pylab.plot(data)
#ax=pylab.axis()
#pylab.plot(x,A)
#pylab.axis(ax)
#pylab.gcf().show()
#pylab.figure()
#qrange=agstcalib(x,A,[1,2,3,4,5],'Lorentz')
#pylab.figure()
#pylab.semilogy(qrange,A)
#pylab.gcf().show()