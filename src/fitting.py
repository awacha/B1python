#fitting.py
import numpy as np
import pylab
import scipy.interpolate
import matplotlib.widgets
import utils
import types

def smoothcurve(x,y,param,mode='spline',extrapolate='reflect'):
    if mode.upper()!='SPLINE':
        param=int(param)
        if param<2:
            return y
    if mode.upper()=='SPLINE':
        tck=scipy.interpolate.splrep(x,y,s=param)
        return scipy.interpolate.splev(x,tck)
    else:
        if extrapolate.upper()=='REFLECT':
            s=np.r_[2*y[0]-y[param:1:-1],y,2*y[-1]-y[-1:-param:-1]]
        else:
            y1=-np.arange(param-1,0,-1)*(y[1]-y[0])+y[0]
            y2=np.arange(len(y),len(y)+param-1)*(y[-1]-y[-2])-(y[-1]-y[-2])*(len(y)-1)+y[-1]
            s=np.r_[y1,y,y2]
        if mode.upper()=='FLAT':
            w=np.ones(param,'d')
            return np.convolve(w/w.sum(),s,mode='same')[param-1:-param+1]
        elif mode.upper()=='HAMMING':
            w=np.hamming(param)
            return np.convolve(w/w.sum(),s,mode='same')[param-1:-param+1]
        elif mode.upper()=='HANNING':
            w=np.hanning(param)
            return np.convolve(w/w.sum(),s,mode='same')[param-1:-param+1]
        elif mode.upper()=='BARTLETT':
            w=np.bartlett(param)
            return np.convolve(w/w.sum(),s,mode='same')[param-1:-param+1]
        elif mode.upper()=='BLACKMAN':
            w=np.blackman(param)
            return np.convolve(w/w.sum(),s,mode='same')[param-1:-param+1]
        else:
            raise ValueError, "invalid window type!"
def fsphere(q,R):
    """Scattering factor of a sphere
    
    Inputs:
        q: q value(s) (scalar or an array of arbitrary size and shape)
        R: radius (scalar)
        
    Output:
        the values of the scattering factor in an array of the same shape as q
    """
    return 1/q**3*(np.sin(q*R)-q*R*np.cos(q*R))
def linfit(xdata,ydata,errdata=None):
    """Fit line to dataset.
    
    Inputs:
        xdata: vector of abscissa
        ydata: vector of ordinate
        errdata [optional]: y error
        
    Outputs:
        a: the slope of the line
        b: the intersection of the line and the y axis
        aerr: the error of "a", calculated by error propagation (sqrt(variance))
        berr: the error of "b"
    """
    xdata=np.array(xdata);
    ydata=np.array(ydata);
    if xdata.size != ydata.size:
        print "The sizes of xdata and ydata should be the same."
        return
    if errdata is not None:
        if ydata.size !=errdata.size:
            print "The sizes of ydata and errdata should be the same."
            return
        errdata=np.array(errdata);
        S=np.sum(1.0/(errdata**2))
        Sx=np.sum(xdata/(errdata**2))
        Sy=np.sum(ydata/(errdata**2))
        Sxx=np.sum(xdata*xdata/(errdata**2))
        Sxy=np.sum(xdata*ydata/(errdata**2))
    else:
        S=xdata.size
        Sx=np.sum(xdata)
        Sy=np.sum(ydata)
        Sxx=np.sum(xdata*xdata)
        Sxy=np.sum(xdata*ydata)
    Delta=S*Sxx-Sx*Sx;
    a=(S*Sxy-Sx*Sy)/Delta;
    b=(Sxx*Sy-Sx*Sxy)/Delta;
    aerr=np.sqrt(S/Delta);
    berr=np.sqrt(Sxx/Delta);
    return a,b,aerr,berr
def unifiedscattering(q,B,G,Rg,P=4):
    """Evaluate the unified equation from G. Beaucage 
        (J. Appl. Cryst. (1995) 28, pp717-728)
    
    Inputs:
        q: vector of momentum transfer values
        B: prefactor for the power-law.
        G: exponential (Guinier) prefactor
        Rg: radius of gyration
        P: exponent for power-law (4 for Porod)
        
    Output:
        a vector of the same size as of q.
    """
    return G*np.exp(-q**2*Rg**2/3.0)+B*pow(pow(scipy.special.erf(q*Rg/np.sqrt(6)),3)/q,P)
def intintensity(data,alpha,alphaerr,qmin=-np.inf,qmax=np.inf,m=0):
    """Calculate integral of the intensity.
    
    Inputs:
        data: 1D small-angle scattering dict
        alpha: (negative) exponent of the last power-law decay of the
            curve
        alphaerr: absolute error of alpha
        qmin: minimal q to take into account, default is -infinity
        qmax: maximal q to take into account, default is +infinity
        m: a positive number. The m-th modulus will be calculated.
        
    Outputs: Q,dQ
        Q: the m-th modulus of the curve
        dQ: its absolute error
    
    Notes:
        The measured parts of the curve are integrated according to the
            trapezoid formula (function trapz()).
        The final slope is assumed to be a power-law decay with exponent
            alpha.
        The low-angle part is assumed to be a rectangle, its height
            being the first intensity value.
            
    """
    alpha=-alpha # it is easier to handle it as a positive number :-)
    if alpha<1:
        raise ValueError('m+alpha should be larger than 1. alpha: %f m: %f (m+alpha): %f',(-alpha,m,m-alpha))
    alpha=alpha-m
    data1=trimq(data,qmin,qmax)
    q2=data1['q'].max()
    q1=data['q'].min()
    
    ret1=q2**(1.0-alpha)/(alpha-1.0)
    dret1=q2**(1.0-alpha)/(alpha-1.0)**2+q2**(-alpha)
    ret2=np.trapz(data['Intensity']*(data['q']**(-alpha)),data['q'])
    dret2=utils.errtrapz(data['q'],data['Error']*(data['q']**(-alpha)))
    ret3=q1*data['Intensity'][data['q']==q1][0]*q1**(-alpha)
    dret3=q1*data['Error'][data['q']==q1][0]*q1**(-alpha)
    
    #print ret1, "+/-",dret1
    #print ret2, "+/-",dret2
    #print ret3, "+/-",dret3
    
    return ret1+ret2+ret3,np.sqrt(dret1**2+dret2**2+dret3**2)
def trimq(data,qmin=-np.inf,qmax=np.inf):
    """Trim the 1D scattering data to a given q-range
    
    Inputs:
        data: scattering data
        qmin: lowest q-value to include
        qmax: highest q-value to include
    
    Output:
        an 1D scattering data dictionary, with points whose q-values were not
            smaller than qmin and not larger than qmax.
    """
    indices=(data['q']<=qmax) & (data['q']>=qmin)
    data1={}
    for k in data.keys():
        data1[k]=data[k][indices]
    return data1
def subconstbg(data,bg,bgerror):
    """Subtract a constant background from the 1D dataset.
    
    Inputs:
        data: 1D data dictionary
        bg: background value
        bgerror: error of bg
    
    Output:
        the background-corrected 1D data.
    """
    return {'q':data['q'].copy(),
           'Intensity':data['Intensity']-bg,
           'Error':np.sqrt(data['Error']**2+bgerror**2)};
def shullroess(data,qmin=-np.inf,qmax=np.inf,gui=False):
    """Do a Shull-Roess fitting on the scattering data dictionary.
    
    Inputs:
        data: scattering data dictionary
        qmin, qmax (optional): borders of ROI
        gui: if true, allows the user to interactively choose the ROI
        
    Output:
        r0: the fitted value of r0
        n: the fitted value of n
        
    Note: This first searches for r0, which best linearizes the
            log(Intensity) vs. log(q**2+3/r0**2) relation.
            After this is found, the parameters of the fitted line give the
            parameters of a Maxwellian-like particle size distribution function.
            After it a proper least squares fitting is carried out, using the
            obtained values as initial parameters.
    """
    leftborder=0.1
    topborder=0.075
    rightborder=0.05
    bottomborder=0.1
    hdistance=0.12
    vdistance=0.12
    if gui:
        bottomborder=0.25
    width=(1-leftborder-rightborder-hdistance)/2.0
    height=(1-topborder-bottomborder-vdistance)/2.0
    ax1=pylab.axes((leftborder,1-topborder-height,width,height))
    ax2=pylab.axes(((1-rightborder-width),1-topborder-height,width,height))
    ax3=pylab.axes((leftborder,bottomborder,width,height))
    ax4=pylab.axes(((1-rightborder-width),bottomborder,width,height))
    
    data1=trimq(data,qmin,qmax)

    if gui:
        axq1=pylab.axes((leftborder,bottomborder/7.0,1-rightborder-leftborder-0.1,bottomborder/7.0))
        axq2=pylab.axes((leftborder,3*bottomborder/7.0,1-rightborder-leftborder-0.1,bottomborder/7.0))
        qsl2=matplotlib.widgets.Slider(axq1,'q_max',data1['q'].min(),data1['q'].max(),data1['q'].max(),valfmt='%1.4f')
        qsl1=matplotlib.widgets.Slider(axq2,'q_min',data1['q'].min(),data1['q'].max(),data1['q'].min(),valfmt='%1.4f')
    
    def dofitting(data,qmin,qmax):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        data1=trimq(data,qmin,qmax)
        Iexp=np.array(data1['Intensity'])
        qexp=np.array(data1['q'])
        errexp=np.array(data1['Error'])
        print "---Shull-Roess-fitting-with-qmin:-%lf-and-qmax:-%lf----" % (qexp.min(),qexp.max())
        logIexp=np.log(Iexp)
        errlogIexp=errexp/Iexp
    
        r0s=np.linspace(1,2*np.pi/qexp.min(),200)
        chi2=np.zeros(r0s.shape)
        for i in range(len(r0s)): # calculate the quality of the line for each r0.
            xdata=np.log(qexp**2+3/r0s[i]**2)
            a,b,aerr,berr=linfit(xdata,logIexp,errlogIexp)
            chi2[i]=np.sum(((xdata*a+b)-logIexp)**2)
        # display the results
        pylab.axes(ax1)
        pylab.title('Quality of linear fit vs. r0')
        pylab.xlabel('r0 (%c)' % 197)
        pylab.ylabel('Quality')
        pylab.plot(r0s,chi2)
        # this is the index of the best fit.
        print chi2.min()
        tmp=pylab.find(chi2==chi2.min())
        print tmp
        bestindex=tmp[0]
    
        xdata=np.log(qexp**2+3/r0s[bestindex]**2)
        a,b,aerr,berr=linfit(xdata,logIexp,errlogIexp)
        n=-(a*2.0+4.0)
        #display the measured and the fitted curves
        pylab.axes(ax2)
        pylab.title('First approximation')
        pylab.xlabel('q (1/%c)' %197)
        pylab.ylabel('Intensity')
        pylab.plot(xdata,logIexp,'.',label='Measured')
        pylab.plot(xdata,a*xdata+b,label='Fitted')
        pylab.legend()
        #display the maxwellian.
        pylab.axes(ax3)
        pylab.title('Maxwellian size distributions')
        pylab.xlabel('r (%c)' % 197)
        pylab.ylabel('prob. dens.')
        pylab.plot(r0s,utils.maxwellian(n,r0s[bestindex],r0s))
        print "First approximation:"
        print "r0: ",r0s[bestindex]
        print "n: ",n
        print "K: ",b
        # do a proper least squares fitting
        def fitfun(p,x,y,err): # p: K,n,r0
            return (y-np.exp(p[0])*(x**2+3/p[2]**2)**(-(p[1]+4)/2.0))/err
        res=scipy.optimize.leastsq(fitfun,np.array([b,n,r0s[bestindex]]), 
                                    args=(qexp,Iexp,errexp),maxfev=1000,full_output=1)
        K,n,R0=res[0]
        print "After lsq fit:"
        print "r0: ",R0
        print "n: ",n
        print "K: ",K
        print "Covariance matrix:",res[1]
        dR0=np.sqrt(res[1][2][2])
        dn=np.sqrt(res[1][1][1])
        # plot the measured and the fitted curves
        pylab.axes(ax4)
        pylab.title('After LSQ fit')
        pylab.xlabel('q (1/%c)'%197)
        pylab.ylabel('Intensity')
        pylab.plot(np.log(qexp**2+3/R0**2),-(n+4)/2.0*np.log(qexp**2+3/R0**2)+K,label='Fitted')
        pylab.plot(np.log(qexp**2+3/R0**2),logIexp,'.',label='Measured')
        pylab.legend()
        # plot the new maxwellian
        pylab.axes(ax3)
        pylab.plot(r0s,utils.maxwellian(n,R0,r0s))
        print "R0:",R0,"+/-",dR0
        print "n:",n,"+/-",dn
        return R0,n,dR0,dn
    if not gui:
        return dofitting(data,qmin,qmax)
    else:
        dofitting(data,qmin,qmax)
        def callbackfun(A,data=data,sl1=qsl1,sl2=qsl2):
            pylab.gcf().show()
            dofitting(data,sl1.val,sl2.val)
            pylab.gcf().show()
        qsl1.on_changed(callbackfun)
        qsl2.on_changed(callbackfun)
def guiniercrosssectionfit(data,qmin=-np.inf,qmax=np.inf,testimage=False,smearingmatrix=None):
    """Do a cross-section Guinier fit on the dataset.
    
    Inputs:
        data: 1D scattering data dictionary
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
        smearingmatrix (not yet working): a matrix for slit-smearing. Must fit the length of the
            scattering curve given in <data>. Leave it None to disable smearing.
            
    Outputs:
        the Guinier radius (radius of gyration) of the cross-section
        the prefactor
        the calculated error of Rg
        the calculated error of the prefactor
    """
    data1=trimq(data,qmin,qmax)
    x1=data1['q']**2;
    err1=np.absolute(data1['Error']/data1['Intensity']*data1['q'])
    y1=np.log(data1['Intensity'])*data1['q']
    Rgcs,Gcs,dRgcs,dGcs=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data1['q']**2,np.log(data1['Intensity'])*data1['q'],'.')
        pylab.plot(data1['q']**2,Rgcs*data1['q']**2+Gcs,'-',color='red');
        pylab.xlabel('$q^2$ (1/%c$^2$)' % 197)
        pylab.ylabel('$q\ln I$')
    return np.sqrt(-Rgcs*2),Gcs,1/np.sqrt(-Rgcs)*dRgcs,dGcs
def guinierthicknessfit(data,qmin=-np.inf,qmax=np.inf,testimage=False):
    """Do a thickness Guinier fit on the dataset.
    
    Inputs:
        data: 1D scattering data dictionary
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
    
    Outputs:
        the Guinier radius (radius of gyration) of the thickness
        the prefactor
        the calculated error of Rgt
        the calculated error of the prefactor
    """
    data1=trimq(data,qmin,qmax)
    x1=data1['q']**2;
    err1=np.absolute(data1['Error']/data1['Intensity']*data1['q']**2)
    y1=np.log(data1['Intensity'])*data1['q']**2
    Rgt,Gt,dRgt,dGt=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data1['q']**2,np.log(data1['Intensity'])*data1['q']**2,'.')
        pylab.plot(data1['q']**2,Rgt*data1['q']**2+Gt,'-',color='red');
        pylab.xlabel('$q^2$ (1/%c$^2$)' % 197)
        pylab.ylabel('$q^2\ln I$')
    return np.sqrt(-Rgt),Gt,0.5/np.sqrt(-Rgt)*dRgt,dGt
def guinierfit(data,qmin=-np.inf,qmax=np.inf,testimage=False):
    """Do a Guinier fit on the dataset.
    
    Inputs:
        data: 1D scattering data dictionary
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
    
    Outputs:
        the Guinier radius (radius of gyration)
        the prefactor
        the calculated error of Rg
        the calculated error of the prefactor
    """
    data1=trimq(data,qmin,qmax)
    x1=data1['q']**2;
    err1=np.absolute(data1['Error']/data1['Intensity']);
    y1=np.log(data1['Intensity']);
    Rg,G,dRg,dG=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data1['q']**2,np.log(data1['Intensity']),'.');
        pylab.plot(data1['q']**2,Rg*data1['q']**2+G,'-',color='red');
        pylab.xlabel('$q^2$ (1/%c$^2$)' % 197)
        pylab.ylabel('ln I');
    return np.sqrt(-Rg*3),G,1.5/np.sqrt(-Rg*3)*dRg,dG
def porodfit(data,qmin=-np.inf,qmax=np.inf,testimage=False):
    """Do a Porod fit on the dataset.
    
    Inputs:
        data: 1D scattering data dictionary
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
    
    Outputs:
        the constant background
        the Porod coefficient
        the calculated error of the constant background
        the calculated error of the Porod coefficient
    """
    data1=trimq(data,qmin,qmax)
    x1=data1['q']**4;
    err1=data1['Error']*x1;
    y1=data1['Intensity']*x1;
    a,b,aerr,berr=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data1['q']**4,data1['Intensity']*data1['q']**4,'.');
        pylab.plot(data1['q']**4,a*data1['q']**4+b,'-',color='red');
        pylab.xlabel('$q^4$ (1/%c$^4$)' % 197)
        pylab.ylabel('I$q^4$');
    return a,b,aerr,berr
def powerfit(data,qmin=-np.inf,qmax=np.inf,testimage=False):
    """Fit a power-law on the dataset (I=e^b*q^a)
    
    Inputs:
        data: 1D scattering data dictionary
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
    
    Outputs:
        the exponent
        ln(prefactor)
        the calculated error of the exponent
        the calculated error of the logarithm of the prefactor
    """
    data1=trimq(data,qmin,qmax)
    x1=np.log(data1['q']);
    err1=np.absolute(data1['Error']/data1['Intensity']);
    y1=np.log(data1['Intensity']);
    a,b,aerr,berr=linfit(x1,y1)
    if testimage:
        pylab.loglog(data1['q'],data1['Intensity'],'.');
        pylab.loglog(data1['q'],np.exp(b)*pow(data1['q'],a),'-',color='red');
        pylab.xlabel('$q$ (1/%c)' % 197)
        pylab.ylabel('I');
    return a,b,aerr,berr
def powerfitwithbackground(data,qmin=-np.inf,qmax=np.inf,testimage=False):
    """Fit a power-law on the dataset (I=B*q^A+C)
    
    Inputs:
        data: 1D scattering data dictionary
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
    
    Outputs:
        the exponent
        the prefactor
        the constant background
        the calculated error of the exponent
        the calculated error of the prefactor
        the calculated error of the constant background
    """
    data1=trimq(data,qmin,qmax)
    x1=data1['q'];
    err1=data1['Error'];
    y1=data1['Intensity'];
    def costfunc(p,x,y,err):
        res= (y-x**p[0]*p[1]-p[2])/err
        return res
    Cinit=0
    Ainit=-4
    Binit=1#(y1[0]-Cinit)/x1[0]**Ainit
    res=scipy.optimize.leastsq(costfunc,np.array([Ainit,Binit,Cinit]),args=(x1,y1,err1),full_output=1)
    if testimage:
        pylab.loglog(data1['q'],data1['Intensity'],'.');
        pylab.loglog(data1['q'],res[0][1]*pow(data1['q'],res[0][0])+res[0][2],'-',color='red');
        pylab.xlabel('$q$ (1/%c)' % 197)
        pylab.ylabel('I');
    return res[0][0],res[0][1],res[0][2],np.sqrt(res[1][0][0]),np.sqrt(res[1][1][1]),np.sqrt(res[1][2][2])    
def unifiedfit(data,B,G,Rg,P,qmin=-np.inf,qmax=np.inf,maxiter=1000):
    """Do a unified fit on the dataset, in the sense of G. Beaucage
    (J. Appl. Cryst. (1995) 28, 717-728)
    
    Inputs:
        data: 1D scattering data dictionary
        B: the initial value for the porod prefactor
        G: the initial value for the Guinier prefactor
        Rg: the initial value for the Guinier radius
        P: the initial value for the exponent of the power-law
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
        maxiter: maximum number of Levenberg-Marquardt iterations
            (scipy.optimize.leastsq)
        
    Outputs:
        the Porod prefactor
        the Guinier prefactor
        the radius of gyration
        the error of the Porod prefactor
        the error of the Guinier prefactor
        the error of the radius of gyration
    """
    data=trimq(data,qmin,qmax)
    def fitfun(data,x,y,err):
        G=data[0]
        B=data[1]
        Rg=data[2]
        P=data[3]
        return (unifiedscattering(x,B,G,Rg,P)-y)/err
    res=scipy.optimize.leastsq(fitfun,np.array([B,G,Rg,P]),args=(data['q'],data['Intensity'],data['Error']),full_output=1)
    return res[0][0],res[0][1],res[0][2],res[0][3],np.sqrt(res[1][0][0]),np.sqrt(res[1][1][1]),np.sqrt(res[1][2][2]),np.sqrt(res[1][3][3])
def fitspheredistribution(data,distfun,R,params,qmin=-np.inf,qmax=np.inf,testimage=False):
    """Fit the scattering data with a sphere distribution function
    
    Inputs:
        data: 1D scattering dictionary
        distfun: distribution function. Should have the following form:
            fun(R,param1,param2,...paramN) where N is the length of params
            (see below). R is a numpy array of radii in Angstroem
        R: numpy array of radii
        params: list of initial parameters for the distribution. 
        qmin: minimum q-value
        qmax: maximum q-value
        testimage: if a test image (visual check of the quality of the fit)
            is desired. Default is False.
    
    Outputs:
        paramsfitted: list of the fitted parameters plus a scaling factor
            added as the last.
    """
    data1=trimq(data,qmin,qmax)
    q=data1['q']
    Int=data1['Intensity']
    Err=data1['Error']
    params=list(params)
    params.append(1)
    params1=tuple(params)
    tsI=np.zeros((len(q),len(R)))
    for i in range(len(R)):
        tsI[:,i]=fsphere(q,R[i])
    R.reshape((len(R),1))
    def fitfun(params,R,q,I,Err,dist=distfun,tsI=tsI):
        return (params[-1]*np.dot(tsI,dist(R,*(params[:-1])))-I)/Err
    res=scipy.optimize.leastsq(fitfun,params1,args=(R,q,Int,Err),full_output=1)
    print "Fitted values:",res[0]
    print "Covariance matrix:",res[1]
    if testimage:
        pylab.semilogy(data['q'],data['Intensity'],'.');
        tsIfull=np.zeros((len(data['q']),len(R)))
        for i in range(len(R)):
            tsIfull[:,i]=fsphere(data['q'],R[i])
        print data['q'].shape
        print np.dot(tsIfull,distfun(R,*(res[0][:-1]))).shape
        pylab.semilogy(data['q'],res[0][-1]*np.dot(tsIfull,distfun(R,*(res[0][:-1]))),'-',color='red');
        pylab.xlabel('$q$ (1/%c)' % 197)
        pylab.ylabel('I');
    return res[0]
def theorspheres(qrange, spheres):
    """Calculate the theoretical scattering intensity of the sphere structure
    
    Inputs:
        qrange: vector of q values or just a single value
        spheres: sphere structure. Could be:
            1) a single number: it is then assumed that the scattering of a single
                sphere with this radius is the scatterer
            2) a python list of numbers or a numpy array with one columns: a 
                sphere population with these radii is assumed
            3) a numpy array: the columns correspond to x, y, z, R respectively.
                If supplied, the 5th column is the real, the 6th is the imaginary
                part of the scattering length density.
            
    Output:
        a vector of the same size that of qrange. It contains the scattering
        intensities.
    """
    
    if (type(qrange)!=types.ListType) and (type(qrange)!=np.ndarray):
        qrange=[qrange]
    if (type(spheres)!=types.ListType) and (type(spheres)!=np.ndarray):
        spheres=[spheres]
    Intensity=np.zeros(qrange.size)
    if (type(spheres)==types.ListType):
        for i in range(len(spheres)):
            Intensity=Intensity+fsphere(qrange,spheres[i])**2
    if (type(spheres)==np.ndarray):
        if spheres.ndim==1:
            for i in range(len(spheres)):
                Intensity=Intensity+fsphere(qrange,spheres[i])**2
            return Intensity
        elif spheres.shape[1]<4:
            raise ValueError("Not enough columns in spheres structure")
        elif spheres.shape[1]<5:
            s1=np.zeros((spheres.shape[0],6))
            s1[:,0:4]=spheres
            s1[:,4]=1;
            s1[:,5]=0;
            spheres=s1;
        for i in range(spheres.shape[0]):
            f1=fsphere(qrange,spheres[i,3])
            Intensity+=(spheres[i,4]**2+spheres[i,5]**2)*f1**2;
            for j in range(i+1,spheres.shape[0]):
                f2=fsphere(qrange,spheres[j,3])
                dist=np.sqrt((spheres[i,0]-spheres[j,0])**2+(spheres[i,1]-spheres[j,1])**2+(spheres[i,2]-spheres[j,2])**2)
                if dist!=0:
                    fact=np.sin(qrange*dist)/(qrange*dist)
                else:
                    fact=1;
                Intensity+=2*(spheres[i,4]*spheres[j,4]+spheres[i,5]*spheres[j,5])*f1*f2*fact;
    return Intensity            
                

