#-----------------------------------------------------------------------------
# Name:        utils.py
# Purpose:     various utility macros
#
# Author:      Andras Wacha
#
# Created:     2010/02/22
# RCS-ID:      $Id: utils.py $
# Copyright:   (c) 2010
# Licence:     GPLv2
#-----------------------------------------------------------------------------
#utils.py

import pylab
import numpy as np
import time
import types
import scipy.special
_pausemode=True

def combinesasdicts(*args):
    """Combines 1D SAS dictionaries.
    
    Inputs:
        arbitrary number of SAS dictionaries (field 'q' is mandatory)
        
    Output:
        combined 1D SAS dict.
    
    Note:
        the SAS dictionaries should have the 'q' field. The fields in a SAS
        dictionary should be numpy ndarrays of the same length. Combining works
        as follows. Let d1 and d2 be two SAS dicts, nd the resulting dict,
        initialized to a copy of d1. For each element of the fields of d2, it
        is first checked, if the corresponding q-value is present in d1. If
        yes, the corresponding element in nd will be the average of those in
        d1 and d2 (nd['fieldname'][i]=0.5*(d1['fieldname'][i]+d2['fieldname'][i])).
        If no, the bin is simply added as a new bin.
        The only exception is field 'Error', where instead of the arithmetic
        average, the quadratic average is used (0.5*sqrt(x**2+y**2))
    """
    if len(args)==1:
        return args[0] # do nothing
    if len(args)==0:
        return None
    newdict={}
    d1=args[0]
    for i in range(1,len(args)):
        d2=args[i]
        a=d1.keys()
        b=d2.keys()
        a.sort()
        b.sort()
        if a!=b:
            raise ValueError,"Input dictionary #%u does not have the same fields (%s) as the previous ones (%s)!" % (i,b,a)
        for i in d1.keys():
            newdict[i]=d1[i].copy()
        for i in range(len(d2['q'])):
            idx=(newdict['q']==d2['q'][i])
            if idx.sum()==0:
                for j in d2.keys():
                    np.append(newdict[j],d2[j][i])
            for j in d2.keys():
                if j=='Error':
                    newdict['Error'][idx]=np.sqrt(newdict['Error'][idx]**2+d2['Error'][i]**2)*0.5
                elif j=='q':
                    continue
                else:
                    newdict[j][idx]=(newdict[j][idx]+d2[j][i])*0.5
        d1=newdict
    return newdict
def matrixsummary(matrix,name):
    """Returns numerical summary of a matrix as a string

    Inputs:
        matrix: the matrix to be summarized
        name: the name of the matrix
    """
    nonnan=-np.isnan(matrix)
    return "%s: %g +/- %g (min: %g; max: %g); NaNs: %d" % (name,matrix[nonnan].mean(),matrix[nonnan].std(),np.nanmin(matrix),np.nanmax(matrix),(1-nonnan).sum())
    
def pause(setto=None):
    """Pause function
    
    Inputs:
        setto: set pause mode. If None, do "pausing". If not None:
            a) boolean value: True or False or 'on' or 'off'. Turns
                pause on or off. If it is turned off, further calls to
                pause() will return immediately.
            b) positive numeric value: set sleep time. After this,
                whenever pause() is called, it will sleep for this many
                seconds.
        
    Examples:
        1)  >>>pause('off') #turns pausing off
            >>>pause()
            #returns immediately
        2) If a matplotlib window is open:
            >>>pause('on') #turns pausing on
            >>>pause()
            Press any key on the current figure (Figure 1) to continue...
            # you should press a key when the figure window is active
        3) if no matplotlib window is open:
            >>>pause('on')
            >>>pause()
            Press Return to continue...
            # you should press ENTER in the Python interpreter to
            # continue
    """
    global _pausemode
    if type(setto)==type(True):
        _pausemode=setto
        return
    elif type(setto)==type('on'):
        if setto.lower()=='on':
            _pausemode=True
        elif setto.lower()=='off':
            _pausemode=False
        return
    elif setto is not None:
        try:
            f=float(setto)
        except:
            return
        if f>=0:
            _pausemode=f
        return
    if type(_pausemode)==type(True) and _pausemode:
        if len(pylab.get_fignums())==0:
            raw_input("Press Return to continue...")
        else:
            try: #wxwidgets backend
                title=pylab.gcf().canvas.manager.window.Title
            except AttributeError:
                try: #tk backend
                    title=pylab.gcf().canvas.manager.window.title()
                except TypeError: #gtk backend
                    title=pylab.gcf().canvas.manager.window.title
                except AttributeError: 
                    try: # qt backend
                        title=pylab.gcf().canvas.manager.window.caption().utf8().data()[:-1]
                    except AttributeError:
                        try: # qt4 backend
                            title=pylab.gcf().canvas.manager.window.windowTitle().toUtf8().data()
                        except AttributeError:
                            try: #tk backend
                                title=pylab.gcf().canvas.manager.window.title()
                            except:
                                title=''
            print "Press any key on the current figure (%s) to continue..." % title
            
            pylab.draw()
            while pylab.gcf().waitforbuttonpress()==False:
                pylab.gcf().show()
                pass
    else:
        try:
            a=float(_pausemode)
        except:
            return
        if a>0:
            time.sleep(a)
        return
def unique(list,comparefun=(lambda a,b:(a==b))):
    """Return unique elements of list.
    
    Inputs:
        list: the list to be treated
        comparefun [optional]: function to compare two items of the list. It
            should return True if the two elements are identical, False if not.
            
    Output:
        the sorted list of unique elements.
        
    Notes:
        comparefun defaults to (lambda a,b:(a==b))
    """
    list1=list[:]; # copy the list because .sort() sorts _in_place_
    list1.sort()
    def redhelper(a,b): # a helper function for reduce(). 
        if type(a)!=types.ListType:
            a=[a]
        if comparefun(a[-1],b):
            return a
        else:
            return a+[b]
    list1=reduce(redhelper,list1)
    if type(list1)!=types.ListType:  #if list originally consisted of 1 element,
                                     #not a list but a single entry would be returned by reduce().
        list1=[list1];
    return list1
def common(a,b):
    """Return common elements in a and b
    
    Inputs:
        a,b: two lists
        
    Output:
        a list with the elements common for a and b
    """
    c=[x for x in a if a in b]
    return c
def derivative(x,y=None):
    """Approximate the derivative by finite difference
    
    Inputs:
        x: x data
        y: y data. If None, x is differentiated.
        
    Outputs:
        x1, dx/dy or dx
    """
    x=np.array(x);
    if y is None:
        return x[1:]-x[:-1]
    else:
        y=np.array(y)
        return (0.5*(x[1:]+x[:-1])),(y[1:]-y[:-1])
def maxwellian(n,r0,x):
    """Evaluate a Maxwellian-like function
    
    Inputs:
        r0: the center
        n: the order
        x: points in which to evaluate.
        
    Output:
        the values at x.
        
    Note: the function looks like:
        M(x)= 2/(x^(n+1)*gamma((n+1)/2))*x^n*exp(-x^2/r0^2)
    """
    return 2.0/(r0**(n+1.0)*scipy.special.gamma((n+1.0)/2.0))*(x**n)*np.exp(-x**2/r0**2);
def errtrapz(x,yerr):
    """Error of the trapezoid formula
    Inputs:
        x: the abscissa
        yerr: the error of the dependent variable
        
    Outputs:
        the error of the integral
    """
    x=np.array(x);
    yerr=np.array(yerr);
    return 0.5*np.sqrt((x[1]-x[0])**2*yerr[0]**2+np.sum((x[2:]-x[:-2])**2*yerr[1:-1]**2)+
                          (x[-1]-x[-2])**2*yerr[-1]**2)
def multfactor(q,I1,E1,I2,E2):
    """Calculate multiplication factor for I1 and I2
    
    Inputs:
        q: abscissa
        I1 and I2: dependent values
        E1 and E2: errors of the dependent values
    
    Output:
        mult: the multiplication factor (I2*mult=I1)
        errmult: the error of the multiplication factor
        
    Notes:
        Currently the integrals of I1 and I2 are calculated and mult is their
        ratio. However, a total least-squares fit (taking E1 and E2 into account
        as well) would be better, but I need help here.
    """
    S1=np.trapz(I1,q)
    eS1=errtrapz(q,E1)
    S2=np.trapz(I2,q)
    eS2=errtrapz(q,E2)
    mult=S1/S2
    errmult=np.sqrt((eS1/S1)**2+(eS2/S2)**2)*mult
    return mult,errmult
def lorentzian(x0,gamma,x):
    """Evaluate the PDF of the Cauchy-Lorentz distribution at given points
    
    Inputs:
        x0: the location of the peak
        gamma: the scale (half-width at half-maximum, HWHM)
        x: points in which the value is needed
        
    Outputs:
        a vector of the same size as x.
    """
    return gamma/(gamma**2+(x-x0)**2)
def lognormdistrib(x,mu,sigma):
    """Evaluate the PDF of the log-normal distribution
    
    Inputs:
        x: the points in which the values should be evaluated
        mu: parameter mu
        sigma: parameter sigma
    
    Outputs:
        y: 1/(x*sigma*sqrt(2*pi))*exp(-(log(x)-mu)^2/(2*sigma^2))
    """
    return 1/(x*sigma*np.sqrt(2*np.pi))*np.exp(-(np.log(x)-mu)**2/(2*sigma**2))
def flatten1dsasdict(data):
    """Flattens 1D SAXS dictionaries
    
    Input:
        data: 1D SAXS dictionary
    
    Output:
        The same dictionary, but every element ('q', 'Intensity', ...) gets
        flattened into 1D.
    """
    d1={}
    for k in data.keys():
        d1[k]=data[k].flatten()
    return d1
def sanitizeint(data):
    """Remove points with nonpositive intensity from 1D SAXS dataset
    
    Input:
        data: 1D SAXS dictionary
        
    Output:
        a new dictionary of which the points with nonpositive intensities were
            omitted
    """
    indices=(data['Intensity']>0)
    data1={}
    for k in data.keys():
        data1[k]=data[k][indices]
    return data1
def dot_error(A,B,DA,DB):
    """Calculate the error of np.dot(A,B) according to squared error
    propagation.
    
    Inputs:
        A,B: The matrices
        DA,DB: The absolute error matrices corresponding to A and B, respectively
        
    Output:
        The error matrix
    """
    return np.sqrt(np.dot(DA**2,B**2)+np.dot(A**2,DB**2));
def inv_error(A,DA):
    """Calculate the error of np.inv(A) according to squared error
    propagation.
    
    Inputs:
        A: The matrix (square shaped)
        DA: The error of the matrix (same size as A)
    
    Output:
        The error of the inverse matrix
    """
    B=np.inv(A);
    return np.sqrt(np.dot(np.dot(B**2,DA**2),B**2))
    
def energycalibration(energymeas,energycalib,energy1):
    """Do energy calibration.
    
    Inputs:
        energymeas: vector of measured (apparent) energies
        energycalib: vector of theoretical energies corresponding to the measured ones
        energy1: vector or matrix or a scalar of apparent energies to calibrate.
        
    Output:
        the calibrated energy/energies, in the same form as energy1 was supplied
        
    Note:
        to do backward-calibration (theoretical -> apparent), swap energymeas
        and energycalib on the parameter list.
    """
    if type(energymeas)!=type(energycalib):
        raise ValueError,"Parameters energymeas and energycalib should be of the same type!"
    try:
        if len(energymeas)!=len(energycalib):
            raise ValueError, "The same number of apparent and true energy values should be given for energy calibration!"
    except TypeError:
        # len() is not defined: energymeas or energycalib is not a list
        energymeas=[energymeas]
        energycalib=[energycalib]
    except:
        raise
    if len(energymeas)==1: # in this case, only do a shift.
        a=1
        aerr=0
        b=energycalib[0]-energymeas[0]
        berr=0
    else: # if more energy values are given, do a linear fit.
        a,b,aerr,berr=fitting.linfit(energymeas,energycalib)
    if type(energy1)==np.ndarray:
        return a*energy1+b
    elif type(energy1)==type([]):
        return [a*e+b for e in energy1]
    elif type(energy1)==type(()):
        return tuple([a*e+b for e in energy1])
    else:
        return a*energy1+b
