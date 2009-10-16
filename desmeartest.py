from B1python import *

data=pylab.loadtxt('/home/andris/labor/saxs/Andris/AS2J15.P00')[1:]

x,A,s=directdesmear(data,1e6,214,900,207,52,22,17,gui=True,smoothlow=1e3,smoothhi=1e9,smoothingmode='log')
pylab.gcf().show()
pylab.figure()
pylab.plot(data)
ax=pylab.axis()
pylab.plot(x,A)
pylab.axis(ax)
pylab.gcf().show()