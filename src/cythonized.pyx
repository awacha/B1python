import pylab

def trapezoidshapefunction(lengthbase,lengthtop,x):
    x=pylab.array(x)
    if len(x)<2:
        return pylab.array(1)
    T=pylab.zeros(x.shape)
    indslopeleft=(x<=-lengthtop/2.0)
    indsloperight=(x>=lengthtop/2.0)
    indtop=(x<=lengthtop/2.0)&(x>=-lengthtop/2.0)
    T[indsloperight]=-4.0/(lengthbase**2-lengthtop**2)*x[indsloperight]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
    T[indtop]=2.0/(lengthbase+lengthtop)
    T[indslopeleft]=4.0/(lengthbase**2-lengthtop**2)*x[indslopeleft]+lengthbase*2.0/(lengthbase**2-lengthtop**2)
    return T

def test(a):
    print "a: ",a
    return "Hello world!"

def smearingmatrix(pixelmin,pixelmax,beamcenter,pixelsize,lengthbaseh,
                   lengthtoph,lengthbasev=0,lengthtopv=0,beamnumh=1024,
                   beamnumv=1):
    """Calculate the smearing matrix for the given geometry.
    
    Inputs: (pixels and pixel coordinates are counted from 0. The direction
        of the detector is assumed to be vertical.)
        pixelmin: the smallest pixel to take into account
        pixelmax: the largest pixel to take into account
        beamcenter: pixel coordinate of the primary beam
        pixelsize: the size of pixels, in micrometers
        lengthbaseh: the length of the base of the horizontal beam profile
        lengthtoph: the length of the top of the horizontal beam profile
        lengthbasev: the length of the base of the vertical beam profile
        lengthtopv: the length of the top of the vertical beam profile
        beamnumh: the number of elementary points of the horizontal beam
            profile
        beamnumv: the number of elementary points of the vertical beam profile
            Give 1 if you only want to take the length of the slit into
            account.
    
    Output:
        The smearing matrix. This is an upper triangular matrix. Desmearing
        of a column vector of the measured intensities can be accomplished by
        multiplying by the inverse of this matrix.
    """
    # coordinates of the pixels
    pixels=pylab.arange(pixelmin,pixelmax+1)
    # distance of each pixel from the beam in pixel units
    x=pylab.absolute(pixels-beamcenter);
    # horizontal and vertical coordinates of the beam-profile in mm.
    if beamnumh>1:
        yb=pylab.linspace(-max(lengthbaseh,lengthtoph)/2.0,max(lengthbaseh,lengthtoph)/2.0,beamnumh)
        deltah=(yb[-1]-yb[0])*1.0/beamnumh
        centerh=2.0/(lengthbaseh+lengthtoph)
    else:
        yb=pylab.array([0])
        deltah=1
        centerh=1
    if beamnumv>1:
        xb=pylab.linspace(-max(lengthbasev,lengthtopv)/2.0,max(lengthbasev,lengthtopv)/2.0,beamnumv)
        deltav=(xb[-1]-xb[0])*1.0/beamnumv
        centerv=2.0/(lengthbasev+lengthtopv)
    else:
        xb=pylab.array([0])
        deltav=1
        centerv=1
    Xb,Yb=pylab.meshgrid(xb,yb)
    #beam profile vector (trapezoid centered at the origin. Only a half of it
    # is taken into account)
    H=trapezoidshapefunction(lengthbaseh,lengthtoph,yb)
    V=trapezoidshapefunction(lengthbasev,lengthtopv,xb)
    P=pylab.kron(H,V)
    center=centerh*centerv
    # scale y to detector pixel units
    Yb=Yb/pixelsize*1e3
    Xb=Xb/pixelsize*1e3
    A=pylab.zeros((len(x),len(x)))
    for i in range(len(x)):
        A[i,i]+=center
        tmp=pylab.sqrt((i-Xb)**2+Yb**2)
        ind1=pylab.floor(tmp).astype('int').flatten()
        prop=tmp.flatten()-ind1
        indices=(ind1<len(pixels))        
        ind1=ind1[indices].flatten()
        prop=prop[indices].flatten()
        p=P[indices].flatten()
        for j in range(len(ind1)):
            A[i,ind1[j]]+=p[j]*(1-prop[j])
            if ind1[j]<len(pixels)-1:
                A[i,ind1[j]+1]+=p[j]*prop[j]
    A=A*deltah*deltav
#    pylab.imshow(A)
#    pylab.colorbar()
#    pylab.gcf().show()
    return A
