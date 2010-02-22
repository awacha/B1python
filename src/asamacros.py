#-----------------------------------------------------------------------------
# Name:        asamacros.py
# Purpose:     Macros for processing 1D SAXS data (line focus, Hecus-ASA)
#
# Author:      Andras Wacha
#
# Created:     2010/02/22
# RCS-ID:      $Id: asamacros.py $
# Copyright:   (c) 2010
# Licence:     GPLv2
#-----------------------------------------------------------------------------

import numpy as np
import pylab
import fitting
import matplotlib.widgets
import guitools
import time
from c_asamacros import smearingmatrix, trapezoidshapefunction

def directdesmear(data,smoothing,params,title=''):
    """Desmear the scattering data according to the direct desmearing
    algorithm by Singh, Ghosh and Shannon
    
    Inputs:
        data: measured intensity vector of arbitrary length (numpy array)
        smoothing: smoothing parameter for smoothcurve(). A scalar
            number. If not exactly known, a dictionary may be supplied with
            the following fields:
                low: lower threshold
                high: upper threshold
                val: initial value
                mode: 'lin' or 'log'
                smoothmode: 'flat', 'hanning', 'hamming', 'bartlett',
                    'blackman' or 'spline', for smoothcurve()
            In this case a GUI will be set up. A slider and an Ok button at
            the bottom of the figure will aid the user to select the optimal
            smoothing parameter.
        params: a dictionary with the following fields:
            pixelmin: left trimming value (default: -infinity)
            pixelmax: right trimming value (default: infinity)
            beamcenter: pixel coordinate of the beam (no default value)
            pixelsize: size of the pixels in micrometers (no default value)
            lengthbaseh: length of the base of the horizontal beam profile
                (millimetres, no default value)
            lengthtoph: length of the top of the horizontal beam profile
                (millimetres, no default value)
            lengthbasev: length of the base of the vertical beam profile
                (millimetres, no default value)
            lengthtopv: length of the top of the vertical beam profile
                (millimetres, no default value)
            beamnumh: the number of elementary points for the horizontal beam
                profile (default: 1024)
            beamnumv: the number of elementary points for the vertical beam
                profile (default: 0)
            matrix: if this is supplied, all but pixelmin and pixelmax are
                disregarded.
        title: display this title over the graph
                
    Outputs: (pixels,desmeared,smoothed,mat,params,smoothing)
        pixels: the pixel coordinates for the resulting curves
        desmeared: the desmeared curve
        smoothed: the smoothed curve
        mat: the desmearing matrix
        params: the desmearing parameters
        smoothing: smoothing parameter
    """
    #default values
    dparams={'pixelmin':-np.inf,'pixelmax':np.inf,
             'beamnumh':1024,'beamnumv':0}
    dparams.update(params)
    params=dparams
    
    # calculate the matrix
    if params.has_key('matrix') and type(params['matrix'])==np.ndarray:
        A=params['matrix']
    else:
        t=time.time()
        A=smearingmatrix(params['pixelmin'],params['pixelmax'],
                         params['beamcenter'],params['pixelsize'],
                         params['lengthbaseh'],params['lengthtoph'],
                         params['lengthbasev'],params['lengthtopv'],
                         params['beamnumh'],params['beamnumv'])
        t1=time.time()
        print "smearingmatrix took %f seconds" %(t1-t)
        params['matrix']=A
    #x coordinates in pixels
    pixels=np.arange(len(data))
    def smooth_and_desmear(pixels,data,params,smoothing,smmode):
        # smoothing the dataset. Errors of the data are sqrt(data), weight will be therefore 1/data
        indices=(pixels<=params['pixelmax']) & (pixels>=params['pixelmin'])
        data=data[indices]
        pixels=pixels[indices]
        data1=fitting.smoothcurve(pixels,data,smoothing,smmode,extrapolate='Linear')
        ret=(pixels,np.linalg.linalg.solve(params['matrix'],data1.reshape(len(data1),1)),
             data1,params['matrix'],params,smoothing)
        return ret
    if type(smoothing)!=type({}):
        res=smooth_and_desmear(pixels,data,params,smoothing,'spline')
        return res
    else:
        f=pylab.figure()
        f.donedesmear=False
        axsl=pylab.axes((0.08,0.02,0.7,0.05))
        axb=pylab.axes((0.85,0.02,0.08,0.05))
        ax=pylab.axes((0.1,0.12,0.8,0.78))
        b=matplotlib.widgets.Button(axb,'Ok')
        def buttclick(a=None,f=f):
            f.donedesmear=True
        b.on_clicked(buttclick)
        if smoothing['mode']=='log':
            sl=matplotlib.widgets.Slider(axsl,'Smoothing',
                                         np.log(smoothing['low']),
                                         np.log(smoothing['high']),
                                         np.log(smoothing['val']))
        elif smoothing['mode']=='lin':
            sl=matplotlib.widgets.Slider(axsl,'Smoothing',
                                         smoothing['low'],
                                         smoothing['high'],
                                         smoothing['val'])
        else:
            raise ValueError('Invalid value for smoothingmode: %s',
                             smoothing['mode'])
        def sliderfun(a=None,sl=sl,ax=ax,mode=smoothing['mode'],x=pixels,
                      y=data,p=params,smmode=smoothing['smoothmode']):
            if mode=='lin':
                sm=sl.val
            else:
                sm=np.exp(sl.val)
            [x1,y1,ysm,A,par,sm]=smooth_and_desmear(x,y,p,sm,smmode)
            a=ax.axis()
            ax.cla()
            ax.semilogy(x,y,'.',label='Original')
            ax.semilogy(x1,ysm,'-',label='Smoothed (%lg)'%sm)
            ax.semilogy(x1,y1,'-',label='Desmeared')
            ax.legend(loc='best')
            ax.axis(a)
            ax.set_title(title)
            pylab.gcf().show()
            pylab.draw()
        sl.on_changed(sliderfun)
        [x1,y1,ysm,A,par,sm]=smooth_and_desmear(pixels,data,params,smoothing['val'],smoothing['smoothmode'])
        ax.semilogy(pixels,data,'.',label='Original')
        ax.semilogy(x1,ysm,'-',label='Smoothed (%lg)'%smoothing['val'])
        ax.semilogy(x1,y1,'-',label='Desmeared')
        ax.legend(loc='best')
        ax.set_title(title)
        pylab.gcf().show()
        pylab.draw()
        while not f.donedesmear:
            pylab.waitforbuttonpress()
        if smoothing['mode']=='lin':
            sm=sl.val
        elif smoothing['mode']=='log':
            sm=np.exp(sl.val)
        else:
            raise ValueError('Invalid value for smoothingmode: %s',
                             smoothing['mode'])
        res=smooth_and_desmear(pixels,data,params,sm,smoothing['smoothmode'])
        return res        
def agstcalib(xdata,ydata,peaks,peakmode='Lorentz',wavelength=1.54,d=48.68,returnq=True):
    """Find q-range from AgSt (or AgBeh) measurements.
    
    Inputs:
        xdata: vector of abscissa values (typically pixel numbers)
        ydata: vector of scattering data (counts)
        peaks: list of the orders of peaks (ie. [1,2,3])
        peakmode: what type of function should be fitted on the peak. Possible
            values: 'Lorentz' and 'Gauss'
        wavelength: wavelength of the X-ray radiation. Default is Cu Kalpha,
            1.54 Angstroems
        d: the periodicity of the sample (default: 48.68 A for silver
            stearate)
        returnq: returns only the q-range if True. If False, returns the
            pixelsize/dist and beamcenter values

    Output:
        If returnq is true then the q-scale in a vector which is of the
            same size as xdata.
        If returnq is false, then a,b,aerr,berr where a is pixelsize/dist,
            b is the beam center coordinate in pixels and aerr and berr
            are their errors, respectively
        
    Notes:
        A graphical window will be popped up len(peaks)-times, each prompting
            the user to zoom on the n-th peak. After the last peak was
            selected, the function returns.
    """
    pcoord=[]
    for p in peaks:
        tmp=guitools.findpeak(xdata,ydata,('Zoom to peak %d and press ENTER' % p),peakmode,scaling='log')
        pcoord.append(tmp)
    pcoord=np.array(pcoord)
    n=np.array(peaks)
    a=(n*wavelength)/(2*d)
    x=2*a*np.sqrt(1-a**2)/(1-2*a**2)
    LperH,xcent,LperHerr,xcenterr=fitting.linfit(x,pcoord)
    print 'pixelsize/dist:',1/LperH,'+/-',LperHerr/LperH**2
    print 'beam position:',xcent,'+/-',xcenterr
    b=(np.array(xdata)-xcent)/LperH
    if returnq:
        return 4*np.pi*np.sqrt(0.5*(b**2+1-np.sqrt(b**2+1))/(b**2+1))/wavelength
    else:
        return 1/LperH,xcent,LperHerr/LperH**2,xcenterr
def tripcalib(xdata,ydata,peakmode='Lorentz',wavelength=1.54,qvals=2*np.pi*np.array([0.21739,0.25641,0.27027]),returnq=True):
    """Find q-range from Tripalmitine measurements.
    
    Inputs:
        xdata: vector of abscissa values (typically pixel numbers)
        ydata: vector of scattering data (counts)
        peakmode: what type of function should be fitted on the peak. Possible
            values: 'Lorentz' and 'Gauss'
        wavelength: wavelength of the X-ray radiation. Default is Cu Kalpha,
            1.54 Angstroems
        qvals: a list of q-values corresponding to peaks. The default values
            are for Tripalmitine
        returnq: True if the q-range is to be returned. False if the fit
            parameters are requested instead of the q-range
    Output:
        The q-scale in a vector which is of the same size as xdata, if 
            returnq was True.
        Otherwise a,b,aerr,berr where q=a*x+b and x is the pixel number
        
    Notes:
        A graphical window will be popped up len(qvals)-times, each prompting
            the user to zoom on the n-th peak. After the last peak was
            selected, the q-range will be returned.
    """
    pcoord=[]
    peaks=range(len(qvals))
    for p in peaks:
        tmp=guitools.findpeak(xdata,ydata,
                     ('Zoom to peak %d (q = %f) and press ENTER' % (p,qvals[p])),
                     peakmode,scaling='lin')
        pcoord.append(tmp)
    pcoord=np.array(pcoord)
    n=np.array(peaks)
    a,b,aerr,berr=fitting.linfit(pcoord,qvals)
    if returnq:
        return a*xdata+b
    else:
        return a,b,aerr,berr
