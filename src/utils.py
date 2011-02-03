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
import matplotlib.widgets
import numpy as np
import time
import types
import scipy.special
_pausemode=True
import string
import utils2d
from functools import wraps
import math

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

class FittingError(Exception):
    pass

class ValueAndError(object):
    def __init__(self,value=0,error=0):
        self.value=value
        self.error=error
    def __float__(self):
        return float(self.value)
    def __unicode__(self):
        return u'%s +/- %s'%(unicode(self.value),unicode(self.error))
    __str__=__unicode__
    def __getitem__(self,v):
        if v==0:
            return self.value
        elif v==1:
            return self.error
        else:
            raise IndexError
    def __len__(self):
        return 2
    def __setitem__(self,k,v):
        if k==0:
            self.value=v
        elif k==1:
            self.error=v
        else:
            raise IndexError
    def __add__(self,x):
        if isinstance(x,tuple) or isinstance(x,list):
            x=ValueAndError(*x)
        if isinstance(x,ValueAndError):
            v=self.value+x.value
            e=math.sqrt(self.error**2+x.error**2)
        elif np.isscalar(x):
            v=self.value+x
            e=self.error
        else:
            return NotImplemented
        return ValueAndError(v,e)
    __radd__=__add__
    def __iadd__(self,x):
        v=self+x
        self.value=v.value
        self.error=v.error
    def __rsub__(self,x):
        return x + (-self)
    def __sub__(self,x):
        return - (x-self)
    def __isub__(self,x):
        v=self-x
        self.value=v.value
        self.error=v.error
    def __neg__(self):
        return ValueAndError(-self.value,self.error)
    def __mul__(self,x):
        if isinstance(x,tuple) or isinstance(x,list):
            x=ValueAndError(*x)
        if isinstance(x,ValueAndError):
            v=self.value*x.value
            e=math.sqrt(self.error**2*x.value**2+x.error**2*self.value**2)
        elif np.isscalar(x):
            v=self.value*x
            e=self.error
        else:
            return NotImplemented
        return ValueAndError(v,e)
    __rmul__=__mul__
    def __div__(self,x):
        if isinstance(x,tuple) or isinstance(x,list):
            x=ValueAndError(*x)
        if isinstance(x,ValueAndError):
            v=self.value/x.value
            e=math.sqrt(self.error**2/x.value**2+x.error**2*self.value**2/x.error**4)
        elif np.isscalar(x):
            v=self.value/x
            e=self.error/x
        else:
            return NotImplemented
        return ValueAndError(v,e)
    def __rdiv__(self,x): # x/self
        if isinstance(x,tuple) or isinstance(x,list):
            x=ValueAndError(*x)
        if isinstance(x,ValueAndError):
            v=x.value/self.value
            e=math.sqrt(x.error**2/self.value**2+self.error**2*x.value**2/self.error**4)
        elif np.isscalar(x):
            v=x/self.value
            e=x*self.error/self.value**2
        else:
            return NotImplemented
        return ValueAndError(v,e)
    def __idiv__(self,x):
        v=self/x
        self.value=v.value
        self.error=v.error
    __truediv__=__div__
    __rtruediv__=__rdiv__
    __itruediv__=__div__
    def __pow__(self,x):
        v=pow(self.value,x)
        e=abs(x*pow(self.value,x-1))*self.error
        return ValueAndError(v,e)
    def __abs__(self):
        return ValueAndError(abs(self.value),self.error)
    def __repr__(self):
        return unicode(self)
    def relative(self):
        return abs(self.error6self.value)
        
class SASDict(object):
    """Small Angle Scattering results in a dictionary-like representation.
    
    This is the recommended way to represent small-angle scattering curves.
    Old representation (dictionary) can be transformed as:
    
    newrep=SASDict(**data)
    
    This class retains the functionality of the dictionary representation:
    fields as 'q', 'Intensity', 'Error', 'Area' are accessible as dictionary
    fields. However, they are also accessible as attributes, ie. data.q, etc.
    
    Arithmetic operations are implemented (in-place also, like in data*=2). 
    Currently the other operand can only be either a scalar (or vector of the
    same length as q), or a tuple of two (elements of it being scalars or
    vectors of the same length as q), in which latter case the second element
    of the tuple is treated as absolute error.
    
    Plotting functions are implemented (plot, loglog, semilogx, semilogy, errorbar),
    which call the matplotlib functions of the same name, forwarding all
    optional arguments to them. Thus constructs like these are possible:
    
    data.plot('.-',linewidth=3,markersize=5,label='test plot')
    
    """
    qtolerance=0.001
    __instances=0
    def __init__(self,q,Intensity,Error=None,**kwargs):
        """Initialize a SASDict.
        
        Inputs:
            q: one-dimensional numpy array of the q values.
            Intensity: one-dimensional numpy array of the Intensity,
                same length as q.
            Error: one-dimensional numpy array of the Error values,
                same length as q, or None if not defined.
            other keyword arguments: names and 1D numpy arrays, ie.
                values of functions of q, ie. Area, qerror...
    
        """
        self._dict={'q':None,'Intensity':None,'Error':None}
        self._dict['q']=np.array(q).flatten()
        self._transform=None
        self._plotaxes=None
        if len(Intensity)!=len(self._dict['q']):
            raise ValueError('Intensity should be of the same length as q!')
        self._dict['Intensity']=np.array(Intensity).flatten()
        if Error is not None:
            if len(Error)!=len(self._dict['q']):
                raise ValueError('Error, if defined, should be of the same length as q!')
            self._dict['Error']=np.array(Error).flatten()
        for k in kwargs.keys():
            if k in ['q','Intensity','Error','s']:
                raise ValueError('%s cannot appear as a variable in a SAS dictionary.'%k)
            if len(kwargs[k])!=len(self._dict['q']):
                raise ValueError('Argument %s should be of the same length as q!'%k)
            self._dict[k]=np.array(kwargs[k]).flatten()
        SASDict.__instances+=1
    def __getattr__(self,key):
        """Overloaded function for attribute fetching a la sasdict.<attr>.
        In addition to normal attributes, the following are defined:
            'q', 'Intensity', 'Error' and other vectors, which were added
                either by __init__() or __setattr__.
            's': q/2/pi
            'transform': transformation object
            'x', 'y', 'dy': transformed 'q', 'Intensity' and 'Error'
        """
        selfdict=object.__getattribute__(self,'_dict')
        if key in selfdict.keys():
            return np.array(selfdict[key])
        elif key=='s':
            return np.array(selfdict[key]/(2*np.pi))
        elif key in ['x','y','dy']:
            if key not in selfdict:
                self.do_transform()
            return selfdict[key]
        elif key=='transform':
            return self._transform
        else: # __getattr__ is called only if the attribute has not been found in the usual places.
            raise AttributeError('Attribute %s not found.'%key)
    def __setattr__(self,key,value):
        """Overloaded __setattr__ method.
        """
        if key=='_dict':
            return object.__setattr__(self,key,value)
        elif key in self._dict.keys():
            if key=='q':
                self._setq(value)
            elif key=='s':
                self._setq(value*2*np.pi)
            elif key in ['x','y','dy']:
                raise AttributeError('Attribute %s is read-only!'%key)
            else:
                if len(value)==len(self._dict['q']):
                    self._dict[key]=np.array(value).flatten()
                else:
                    raise ValueError('New value for %s should be of the same length as q!' %key)
        elif key=='transform':
            self._transform=value
            try:
                del self._dict['x']
                del self._dict['y']
                del self._dict['dy']
            except KeyError:
                pass
            self.do_transform()
        else:
            return object.__setattr__(self,key,value)
    def __getitem__(self,key):
        """Same as __getattr__
        """
        return self.__getattr__(key)
    def __setitem__(self,key,value):
        """Same as __setattr__.
        """
        return self.__setattr__(key,value)
    def _setq(self,q1):
        """Setter function of q, usually called by __setattr__.
        """
        q1=np.array(q1).flatten()
        if (len(self._dict['q'])==len(q1)):
            self._dict['q']=q1
        else:
            # new q differs, Intensity, Error and Area have to be reset.
            del self._dict
            self._dict={'q':q1}
    def keys(self):
        """keys() function, a la dict.
        """
        return self._dict.keys()
    def __len__(self):
        """Gives the length of the dict.
        """
        return self._dict.__len__()
    def values(self):
        """values() function, a la dict.
        """
        return self._dict.values()
    def save(self,filename,cols=['q','Intensity','Error']):
        """Saves the SASDict to a text file with comments in the first line.
        
        Inputs:
            filename: name of the file
            cols [optional]: column names to save.
        """
        keys=[k for k in cols if k in self.keys()]
        f=open(filename,'wt')
        f.write('#%s\n'%string.join([str(k) for k in keys()]))
        np.savetxt(f,np.array(self,keys))
        f.close()
    def copy(self):
        """Make a copy"""
        return SASDict(**self)
    def trimq(self,qmin=-np.inf,qmax=np.inf,inplace=False):
        """Trim the 1D scattering data to a given q-range
        
        Inputs:
            qmin: lowest q-value to include (default: ignore)
            qmax: highest q-value to include (default: ignore)
            inplace: True if the data in the current SASDict should
                be manipulated. False (default) if a new SASDict should
                be returned.

        Intensity, Error and Area (if present) will be trimmed.
        """
        newdict={}
        indices=(self._dict['q']<=qmax) & (self._dict['q']>=qmin)
        for k in self._dict.keys():
            if inplace:
                self._dict[k]=self._dict[k][indices]
            else:
                newdict[k]=self._dict[k][indices]
        if inplace:
            return self
        else:
            return SASDict(**newdict)
    def trims(self,smin=-np.inf,smax=np.inf,*args,**kwargs):
        """The same as trimq, but according to s (s=q/(2*pi))
        """
        return self.trimq(qmin=smin*2*np.pi,qmax=smax*2*np.pi,*args,**kwargs)
    def loglog(self,*args,**kwargs):
        """Plot the transformed dataset in log-log plot. Additional
        arguments are forwarded to pylab.loglog().
        """
        self.do_transform()
        pylab.loglog(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def semilogy(self,*args,**kwargs):
        """Plot the transformed dataset in lin-log plot. Additional
        arguments are forwarded to pylab.semilogy().
        """
        self.do_transform()
        pylab.semilogy(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def semilogx(self,*args,**kwargs):
        """Plot the transformed dataset in log-lin plot. Additional
        arguments are forwarded to pylab.semilogx().
        """
        self.do_transform()
        pylab.semilogx(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def plot(self,*args,**kwargs):
        """Plot the transformed dataset in lin-lin plot. Additional
        arguments are forwarded to pylab.plot().
        """
        self.do_transform()
        pylab.plot(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def errorbar(self,*args,**kwargs):
        """Plot the transformed dataset and errors in lin-lin plot. Additional
        arguments are forwarded to pylab.errorbar().
        """
        self.do_transform()
        pylab.errorbar(self.x,self.y,self.dy,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def _check_compat(self,x,die=False):
        """Helper function to check the compatibility of another SASDict against
        the current, by comparing the q-scales. If die is True, raises a
        ValueError if x and self are incompatible. Otherwise a boolean is 
        returned.
        """
        if len(x._dict['q'])!=len(self._dict['q']):
            if die:
                raise ValueError('Incompatible SAS dicts (q-scales have different lengths)!')
            return False
        if (2*np.absolute(x._dict['q']-self._dict['q'])/(x._dict['q']+self._dict['q'])).sum()/len(self._dict['q'])>SASDict.qtolerance:
            if die:
                raise ValueError('Incompatible SAS dicts (q-scales are different)!')
            return False
        return True
    def __imul__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            self._dict['q']=0.5*(self._dict['q']+x._dict['q'])
            self._dict['Intensity']=self._dict['Intensity']*x._dict['Intensity']
            self._dict['Error']=np.sqrt((x._dict['Intensity']*self._dict['Error'])**2+(self._dict['Intensity']*x._dict['Error'])**2)
            #leave other _dict items as they are
            return self
        else:
            val=x
            err=0
        self._dict['Error']=np.sqrt((self._dict['Intensity']*err)**2+(self._dict['Error']*val)**2)
        self._dict['Intensity']=self._dict['Intensity']*val
        return self
    def __idiv__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self*=1/x # use __imul__ and __rdiv__
            return self
        else:
            val=x
            err=0
        self._dict['Error']=np.sqrt((self._dict['Intensity']/(val*val)*err)**2+(self._dict['Error']/val)**2)
        self._dict['Intensity']=self._dict['Intensity']/val
        return self
    def __iadd__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            self._dict['q']=0.5*(self._dict['q']+x._dict['q'])
            self._dict['Intensity']=self._dict['Intensity']+x._dict['Intensity']
            self._dict['Error']=np.sqrt(self._dict['Error']**2+x._dict['Error']**2)
            return self
        else:
            val=x
            err=0
        self._dict['Error']=np.sqrt((err)**2+(self._dict['Error'])**2)
        self._dict['Intensity']=self._dict['Intensity']+val
        return self
    def __isub__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self+=(-x) # use __iadd__ and __neg__
            return self
        else:
            val=x
            err=0
        self._dict['Error']=np.sqrt((err)**2+(self._dict['Error'])**2)
        self._dict['Intensity']=self._dict['Intensity']-val
        return self
    def __mul__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            q1=0.5*(self._dict['q']+x._dict['q'])
            Intensity1=self._dict['Intensity']*x._dict['Intensity']
            Error1=np.sqrt((x._dict['Intensity']*self._dict['Error'])**2+(self._dict['Intensity']*x._dict['Error'])**2)
            return SASDict(q=q1,Intensity=Intensity1,Error=Error1)
        else:
            val=x
            err=0
        err=np.sqrt((self._dict['Intensity']*err)**2+(self._dict['Error']*val)**2)
        val=self._dict['Intensity']*val
        return SASDict(q=self._dict['q'],Intensity=val,Error=err)
    def __div__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            return self*(1/x) # __mul__ and __rdiv__
        else:
            val=x
            err=0
        err=np.sqrt((self._dict['Intensity']/(val*val)*err)**2+(self._dict['Error']/val)**2)
        val=self._dict['Intensity']/val
        return SASDict(q=self._dict['q'],Intensity=val,Error=err)
    __itruediv__=__idiv__
    def __add__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            return SASDict(q=0.5*(self._dict['q']+x._dict['q']),Intensity=self._dict['Intensity']+x._dict['Intensity'],Error=np.sqrt(self._dict['Error']**2+x._dict['Error']**2))
        else:
            val=x
            err=0
        err=np.sqrt((err)**2+(self._dict['Error'])**2)
        val=self._dict['Intensity']+val
        return SASDict(q=self._dict['q'],Intensity=val,Error=err)
    def __sub__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            return self+(-x) # __add__ and __neg__
        else:
            val=x
            err=0
        err=np.sqrt((err)**2+(self._dict['Error'])**2)
        val=self._dict['Intensity']-val
        return SASDict(q=self._dict['q'],Intensity=val,Error=err)
    def __neg__(self):
        return SASDict(q=self._dict['q'],Intensity=-self._dict['Intensity'],Error=self._dict['Error'])
    __itruediv__=__idiv__
    __truediv__=__div__
    __rmul__=__mul__
    def __rdiv__(self,x):
        if (isinstance(x,tuple) and len(x)==2) or isinstance(x,ValueAndError):
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            return x*(1/self) # __mul__ and __rdiv__, but the latter with x being a scalar 1
        else:
            val=x
            err=0
        err=np.sqrt((err/self._dict['Intensity'])**2+(val/(self._dict['Error'])**2)**2)
        val=self._dict['Intensity']/val
        return SASDict(q=self._dict['q'],Intensity=val,Error=err)
    __rtruediv__=__truediv__
    __radd__=__add__
    def __rsub__(self,x):
        return -(self-x)
    def __pow__(self,exponent,modulus=None):
        if modulus is not None:
            return NotImplemented # this is hard to implement for SAS curves.
        if exponent==0:
            return SASDict(q=self._dict['q'],Intensity=np.zeros(self._dict['q'].shape),Error=np.zeros(self._dict['q'].shape),)
        else:
            return SASDict(q=self._dict['q'],Intensity=np.power(self._dict['Intensity'],exponent),
                           Error=self._dict['Error']*np.absolute((exponent)*np.power(self._dict['Intensity'],exponent-1)))
    def __array__(self,keys=None):
        """Make a structured numpy array from the current dataset.
        """
        if keys==None:
            keys=self.keys()
            values=self.values()
        else:
            values=[self[k] for k in keys]
        a=np.array(zip(*values),dtype=zip(keys,[np.double]*len(keys)))
        return a
    def sort(self,order='q'):
        """Sort the current dataset according to 'order' (defaults to 'q').
        """
        a=self.__array__()
        sorted=np.sort(a,order=order)
        for k in self._dict.keys():
            self._dict[k]=sorted[k]
        return self
    def sanitize(self,accordingto='Intensity',thresholdmin=0,thresholdmax=np.inf,function=None):
        """Do a sanitization on this SASDict, i.e. remove invalid elements.
        
        Inputs:
            accordingto: the field, which should be inspected, defaults to
                'Intensity'
            thresholdmin: if the inspected field is smaller than this one,
                the line is disregarded.
            thresholdmax: if the inspected field is larger than this one,
                the line is disregarded.
            function: if this is not None, the validity of the dataline is
                decided from the boolean return value of function(value).
                Should accept a list and return a list of booleans.
        """        
        if hasattr(function,'__call__'):
            indices=function(self._dict[accordingto])
        else:
            indices=(self._dict[accordingto]>thresholdmin) & (self._dict[accordingto]<thresholdmax)
        for k in self._dict.keys():
            self._dict[k]=self._dict[k][indices]
        return self
    def modulus(self,exponent=0,errorrequested=False):
        """Calculate moduli (i.e. integral from 0 to infinity of Intensity
        times q^exponent.
        
        Inputs:
            exponent: the exponent of q in the integration.
            errorrequested: True if error should be returned.
        """
        x=self._dict['q']
        y=self._dict['Intensity']*self._dict['q']**exponent
        err=self._dict['Error']*self._dict['q']**exponent
        m=np.trapz(y,x)
        dm=errtrapz(x,err)
        if errorrequested:
           return (m,dm) 
        else:
            return m
    def integral(self,errorrequested=False):
        """Integrate the scattering curve.
        
        Inputs:
            errorrequested: True if error should be returned.
        
        Note:
            this calculates the 0-th modulus (int_0^inf (q^0*Intensity) dq
        """
        return self.modulus(errorrequested=errorrequested)
    def __del__(self):
        for k in self._dict.keys():
            del self._dict[k]
        del self._dict
        SASDict.__instances-=1
        #print "An instance of SASDict has been disposed of. Remaining instances:",SASDict.__instances
    def do_transform(self):
        """Carry out the transformation (field "transform"), ie. calculate
        fields 'x', 'y' and 'dy'.
        """
        if self._transform is None:
            self._dict['x']=self._dict['q']
            self._dict['y']=self._dict['Intensity']
            self._dict['dy']=self._dict['Error']
        else:
            self._dict.update(self._transform.do_transform(**(self._dict)))
            
    def fit(self,function,params_initial,full_output=False,**kwargs):
        """Do a fit on the current dataset
        
        Inputs:
            function: the fit function. Should return the calculated values for 
                the intensity in a vector of the same length as its first
                argument (q). Will be called as:
                    
                function(q,param1,param2,...).
                
            params_initial: initial values of the parameters, in a list
            full_output: if True, optional output variables are returned.
                Defaults to False.
            
        Keyword arguments will be forwarded to scipy.optimize.leastsq.
        
        Outputs: params_final, errors_final, {fittedcurve, chisquare, dof}
            params_final: fitted parameters
            errors_final: errors of the fitted parameters
            fittedcurve: optional output. The fitted curve (ie. the value of
                function() with the last parameters
            chisquare: chi squared
            dof: degrees of freedom
        Will raise a ValueError if the fitting does not succeed.
        """
        if self._dict['Error'] is None or np.any(self._dict['Error']==0):
            p,cov_x,infodict,mesg,ier=scipy.optimize.leastsq(lambda p:(function(self._dict['q'],*p)-self._dict['Intensity']),params_initial,full_output=1,**kwargs)
        else:
            p,cov_x,infodict,mesg,ier=scipy.optimize.leastsq(lambda p:(function(self._dict['q'],*p)-self._dict['Intensity'])/self._dict['Error'],params_initial,full_output=1,**kwargs)
        chisquare=(infodict['fvec']**2).sum()
        degrees_of_freedom=len(self._dict['q'])-len(p)
        if ier<1 or ier>4:
            raise ValueError('Fitting did not succeed. Reason: %s'%mesg)
            #print "Fitting did not succeed:",mesg
        if cov_x is None:
            errors=[np.inf]*len(p)
            print "Infinite covariance in fitting!"
        else:
            errors=[ np.sqrt(cov_x[i,i]*chisquare/degrees_of_freedom) for i in range(len(p))]
        if full_output:
            return p,errors,function(self._dict['q'],*p),chisquare,degrees_of_freedom
        else:
            return p,errors
    def _fitting_base(self,function,transformdatasettolinear=None, transformparamfromlinear=None,params_initial=None,plotinfo=None,**kwargs):
        """Base fitting function, for internal usage only.
        
         Parameters:
            function is the fitting function to be fitted. Will be forwarded to self.fit().
            transformdatasettolinear: this is a callable, eg. a subclass of SASTransform, which
                will transform q,Intensity to a line, if possible. If such a linearization is
                not viable, should be None
            transformparamfromlinear: a callable, which transforms the parameters of a fitted line
                back to values usable for a proper least-squares fit of the function. Can be
                None, if no such transformation is needed.
            params_initial: first guess for the parameters. Can be None if
                transformdatasettolinear is defined. In other cases, this should either be a list
                which will be forwarded to self.fit(). Or, it can be a callable, accepting
                q and Intensity and returning a list of initial parameters forwardable to self.fit()
            plotinfo: None, a dictionary, or anything.
                a) None: no plotting is desired.
                b) dictionary: fields needed are 'funcname' (description of the function) and
                    'paramnames' (list of names of the parameters). An optional field is 
                    'otherstringforlegend', which can be a string, which will be appended to
                    the legend, or a callable, which provides that string. Then its arguments
                    will be (q,Intensity,Error,paramsfitted,errorofparamsfitted).
        
         Returns: params,errors,curve,chi2,dof
            params: values fitted
            errors: errors of the fitted params
            curve: the fitted curve
            chi2: chi-squared (not reduced)
            dof: degrees of freedom
        """
        #try to linearize the dataset and get the initial values for the params.
        if transformdatasettolinear is not None:
            # do the linearization
            d=transformdatasettolinear(self._dict['q'],self._dict['Intensity'],self._dict['Error'])
            # do the fitting
            a,b=scipy.polyfit(d['x'],d['y'],1)
            # try to transform the parameters back
            if transformparamfromlinear is not None:
                a,b=transformparamfromlinear(a,b)
            # set them as the initial guess
            params_initial=[a,b]
            if not np.isfinite(params_initial).all():
                raise FittingError('Linearization did not succeed')
        # if linearization did not succeed, check if params_initial is supplied
        elif params_initial is None:
            raise ValueError('params_initial should not be None if transformdatasettolinear is None.')
        # if params_initial is callable, call it to get the first guess
        elif hasattr(params_initial,'__call__'): #in this case, it is a guessing function
            params_initial=params_initial(self._dict['q'],self._dict['Intensity'])
        # at this point, we have params_initial. Fitting can be carried out.
        if not linearization_error:
            params,errors,curve,chi2,dof=self.fit(function,params_initial,full_output=True,**kwargs)
        #check if plotting was requested.
        if plotinfo is not None:
            if transformdatasettolinear is not None: # if linearization is possible, linearize it.
                pylab.errorbar(d['x'],d['y'],d['dy'],fmt='b.',label='original dataset')
                d1=transformdatasettolinear(self._dict['q'],curve,self._dict['Error'])
                pylab.plot(d1['x'],d1['y'],'r-',label='fitted curve')
            else: # no linearization
                pylab.errorbar(self._dict['q'],self._dict['Intensity'],self._dict['Error'],fmt='b.',label='original dataset')
                pylab.plot(self._dict['q'],curve,'r-',label='fitted curve')
            if hasattr(plotinfo,'keys'): # construct the legend.
                fittinglog=u'Function: %s\nParameters:\n'%plotinfo['funcname']
                for p,e,l in zip(params,errors,plotinfo['paramnames']):
                    fittinglog=fittinglog+(u'    %s: %g +/- %g\n'%(l,p,e))
                fittinglog=fittinglog+(u'Chi-square: %g\n'%chi2)
                fittinglog=fittinglog+(u'Degrees of freedom: %d\n'%dof)
                fittinglog=fittinglog+(u'RMS of residuals: %g\n'%np.sqrt(chi2/dof))
                fittinglog=fittinglog+(u'q in [%g, %g] 1/\xc5\n'%(self._dict['q'].min(),self._dict['q'].max()))
                if 'otherstringforlegend' in plotinfo.keys():
                    if hasattr(plotinfo['otherstringforlegend'],'__call__'):
                        fittinglog=fittinglog+plotinfo['otherstringforlegend'](self._dict['q'],self._dict['Intensity'],self._dict['Error'],params,errors)
                    else:
                        fittinglog=fittinglog+plotinfo['otherstringforlegend']
            #plot the legend
            pylab.text(0.95,0.95,fittinglog,bbox={'facecolor':'white','alpha':0.6,'edgecolor':'black'},ha='right',va='top',multialignment='left',transform=pylab.gca().transAxes)
        return params,errors,curve,chi2,dof
 
    def guinierfit(self,qpower=0,plot=True):
        """Do a Guinier-fit (I=G*q^qpower*exp(-q^2*Rg^2/3)) on the dataset.
        
        Inputs:
            qpower: modification for the fit function for cross-section and
                thickness fittings (see the guinierthicknessfit and
                guiniercrosssectionfit). Default: 0.
            plot: if a plot is requested.
        
        Outputs: [G,Rg],[dG,dRg]
            the fitted parameters and their errors.
        """
        divisor=3-qpower
        fitfunction=lambda q,G,R:np.power(q,qpower)*G*np.exp(-q**2*R**2/divisor)
        paramtransform=lambda mR2div,lnG:(np.exp(lnG),np.sqrt(-divisor*mR2div))
        if plot:
            def legend_addendum(q,I,E,params,errors):
                return 'q_max*R = %g\n'%(q.max()*params[1])
            plotinfo={'funcname':'G*q^%d*exp(-q^2*R^2/%d)'%(qpower,divisor),
                      'paramnames':['G','R_g'],
                      'otherstringforlegend':legend_addendum}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              SASTransformGuinier(qpower),
                                              paramtransform,
                                              None,
                                              plotinfo)
        return p,e
    def guinierthicknessfit(self,*args,**kwargs):
        """Do a Guinier-thickness-fit (I=G*q^2*exp(-q^2*Rg^2)) on the dataset.
        
        This is a specialization of guinierfit() with qpower==2.
        """
        return self.guinierfit(qpower=2,*args,**kwargs)
    def guiniercrosssectionfit(self,*args,**kwargs):
        """Do a Guinier-cross-section-fit (I=G*q^2*exp(-q^2*Rg^2/2)) on the dataset.
        
        This is a specialization of guinierfit() with qpower==1.
        """
        return self.guinierfit(qpower=1,*args,**kwargs)
    def powerlawfit(self,plot=True):
        """Do a Power-law fit (I=A*q^B) on the dataset.
        
        Inputs:
            plot: if a plot is requested.
        
        Outputs: [A,B],[dA,dB]
            the fitted parameters and their errors.
        """
        fitfunction=lambda q,A,B:np.power(q,B)*A
#        paramtransform=lambda lnA,B:(np.exp(lnA),B)
        if plot:
            plotinfo={'funcname':'A*q^B',
                      'paramnames':['A','B']}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              None,
                                              None,
                                              params_initial=[1.0,-4.0],
                                              plotinfo=plotinfo)
        return p,e
    def powerlawconstantbackgroundfit(self,plot=True):
        """Do a Power-law with constant background fit (I=A*q^B+C) on the dataset.
        
        Inputs:
            plot: if a plot is requested.
        
        Outputs: [A,B,C],[dA,dB,dC]
            the fitted parameters and their errors.
        """
        fitfunction=lambda q,A,B,C:np.power(q,B)*A+C
        if plot:
            plotinfo={'funcname':'A*q^B+C',
                      'paramnames':['A','B','C']}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              None,
                                              None,
                                              params_initial=[1.0,-4.0,0.0],
                                              plotinfo=plotinfo)
        return p,e
    def powerlawlinearbackgroundfit(self,plot=True):
        """Do a Power-law with linear background fit (I=A*q^B+C+D*q) on the dataset.
        
        Inputs:
            plot: if a plot is requested.
        
        Outputs: [A,B,C,D],[dA,dB,dC,dD]
            the fitted parameters and their errors.
        """
        fitfunction=lambda q,A,B,C,D:np.power(q,B)*A+C+D*q
        if plot:
            plotinfo={'funcname':'A*q^B+C+D*q',
                      'paramnames':['A','B','C','D']}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              None,
                                              None,
                                              params_initial=[1.0,-4.0,0.0,1.0],
                                              plotinfo=plotinfo)
        return p,e
    def porodfit(self,porod_exponent=-4,plot=True):
        """Do a Porod fit (I=A*q^porod_exponent + B) on the dataset.
        
        Inputs:
            porod_exponent: the fixed exponent of q. Default is -4 (Porod's law)
            plot: if a plot is requested.
        
        Outputs: [A,B],[dA,dB]
            the fitted parameters and their errors.
        """
        fitfunction=lambda q,A,B:np.power(q,porod_exponent)*A+B
        paramtransform=lambda A,B:(B,A)
        if plot:
            plotinfo={'funcname':'A*q^(%g)+B'%(porod_exponent),
                      'paramnames':['A','B']}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              SASTransformPorod(porod_exponent),
                                              paramtransform,
                                              params_initial=None,
                                              plotinfo=plotinfo)
        return p,e 

    def zimmfit(self,plot=True):
        """Do a Zimm fit (I=I0/(1+xi^2*q^2)) on the dataset.
        
        Inputs:
            plot: if a plot is requested.
        
        Outputs: [I0,xi],[dI0,dxi]
            the fitted parameters and their errors.
        """
        fitfunction=lambda q,I0,xi:I0/(1+xi*xi*q*q)
        paramtransform=lambda A,B:(1/B,np.sqrt(A/B))
        if plot:
            plotinfo={'funcname':'I0/(1+xi^2*q^2)',
                      'paramnames':['I0','xi']}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              SASTransformZimm(),
                                              paramtransform,
                                              params_initial=None,
                                              plotinfo=plotinfo)
        return p,e 

    def guinierandpowerlawfit(self,qpower=0,plot=True,G=1e-3,R=20,A=1,B=-4):
        """Do a simultaneous Guinier and Power-law fit on the dataset
        ( I = A*q^B + G*q^qpower*exp(-q^2*R^2/3) )
        
        Inputs:
            qpower: see guinierfit().
            plot: if a plot is requested.
            G: initial value for parameter G
            R: initial value for parameter R
            A: initial value for parameter A
            B: initial value for parameter B
        
        Outputs: [G,R,A,B],[dG,dR,dA,dB]
            the fitted parameters and their errors.
        """
        divisor=3-qpower
        fitfunction=lambda q,G,R,A,B:np.power(q,qpower)*G*np.exp(-q**2*R**2/divisor)+A*np.power(q,B)
        if plot:
            def legend_addendum(q,I,E,params,errors):
                return 'q_max*R = %g\n'%(q.max()*params[1])
            plotinfo={'funcname':'G*q^%d*exp(-q^2*R^2/%d) + A*q^B'%(qpower,divisor),
                      'paramnames':['G','R_g','A','B'],
                      'otherstringforlegend':legend_addendum}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              None,
                                              None,
                                              [G,R,A,B],
                                              plotinfo)
        return p,e

    def multigaussfit(self,m=[0],sigma=[1],scaling=[1],plot=True):
        """Do a multiple Gauss-peaks fit on the dataset
        
        Inputs:
            m: list of expected values
            sigma: list of variances
            scaling: list of Intensity scaling values
            plot: if a plot is requested
        
        Outputs: [scaling1,m1,sigma1,scaling2,m2,sigma2,...],[dscaling1,dm1,dsigma1,dscaling2,dm2,dsigma2,...]
            lists of the fitted parameters and their errors.
        """
        nargs=min(len(m),len(sigma),len(scaling))
        def fitfunction(q,*args):
            ret=np.zeros(q.shape)
            for i in xrange(len(args)/3):
                ret+=args[3*i+0]/np.sqrt(2*np.pi*args[3*i+2]**2)*np.exp(-(q-args[3*i+1])**2/(2*args[3*i+2]**2))
            return ret
        paramnames=[]
        params=[]
        for i in range(nargs):
            paramnames.extend(['scaling%d'%(i+1),'m%d'%(i+1),'sigma%d'%(i+1)])
            params.extend([scaling[i],m[i],sigma[i]])
        if plot:
            def legend_addendum(q,I,E,params,errors):
                return 'q_max*R = %g\n'%(q.max()*params[1])
            plotinfo={'funcname':'%d Gauss peak(s)'%(nargs),
                      'paramnames':paramnames,
                      'otherstringforlegend':legend_addendum}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              None,
                                              None,
                                              params,
                                              plotinfo)
        return p,e

    def trimzoomed(self,inplace=False):
        """Trim dataset according to the current zoom on the last plot.
        
        Inputs:
            inplace: True if the current dataset is to be trimmed. If False,
                a new SASDict is returned.
                
        Notes:
            This method is useful to reduce the dataset to the currently viewed
                range. I.e. if this SASDict was plotted using one of its plot
                methods (e.g. plot(), loglog(), errorbar(), ...) and the graph
                was zoomed in, then calling this function will eliminate the
                off-graph points.
            It is not detected if the axis is still open or the plot is still
                there. If it is not (e.g. cla() or clf() or close() was issued),
                the behaviour of this function is undefined.
        """
        if self._plotaxes is None:
            raise ValueError('No plot axes corresponds to this SASDict')
        limits=self._plotaxes.axis()
        indices=(self.x>=limits[0])&(self.x<=limits[1])&(self.y>=limits[2])&(self.y<=limits[3])
        newdict={}
        for k in self._dict.keys():
            if inplace:
                self._dict[k]=self._dict[k][indices]
            else:
                newdict[k]=self._dict[k][indices]
        if inplace:
            return self
        else:
            return SASDict(**newdict)

    def basicfittinggui(self,title='',blocking=False):
        """Graphical user interface to carry out basic (Guinier, Porod, etc.)
        fitting to 1D scattering data.
        
        Inputs:
            title: title to display
            blocking: False if the function should return just after drawing the
                fitting gui. True if it should wait for closing the figure window.
        Output:
            If blocking was False then none, this leaves a figure open for further
                user interactions.
            If blocking was True then after the window was destroyed, a list of
                the fits and their parameters are returned.
        """
        listoffits=[]
        
        leftborder=0.05
        topborder=0.9
        bottomborder=0.1
        leftbox_end=0.3
        fig=pylab.figure()
        pylab.clf()
        plots=[{'name':'Guinier','transform':SASTransformGuinier(),
                'plotmethod':'plot'},
               {'name':'Guinier thickness','transform':SASTransformGuinier(2),
                'plotmethod':'plot'},
               {'name':'Guinier cross-section','transform':SASTransformGuinier(1),
                'plotmethod':'plot'},
               {'name':'Porod','transform':SASTransformPorod(2),
                'plotmethod':'plot'},
               {'name':'Double linear','transform':SASTransformLogLog(False,False),
                'plotmethod':'plot'},
               {'name':'Logarithmic y','transform':SASTransformLogLog(False,False),
                'plotmethod':'semilogy'},
               {'name':'Logarithmic x','transform':SASTransformLogLog(False,False),
                'plotmethod':'semilogx'},
               {'name':'Double logarithmic','transform':SASTransformLogLog(False,False),
                'plotmethod':'loglog'},
               {'name':'Zimm','transform':SASTransformZimm(),
                'plotmethod':'plot'},
                ]
        buttons=[{'name':'Guinier','fitmethod':'guinierfit'},
                 {'name':'Guinier thickness','fitmethod':'guinierthicknessfit'},
                 {'name':'Guinier cross-section','fitmethod':'guiniercrosssectionfit'},
                 {'name':'Porod','fitmethod':'porodfit'},
                 {'name':'A * q^B','fitmethod':'powerlawfit'},
                 {'name':'A * q^B + C','fitmethod':'powerlawconstantbackgroundfit'},
                 {'name':'A * q^B + C + D * q','fitmethod':'powerlawlinearbackgroundfit'},
                 {'name':'I0/(1+xi^2*q*2)','fitmethod':'zimmfit'}
                 ]
                
        for i in range(len(buttons)):
            ax=pylab.axes((leftborder,topborder-(i+1)*(0.8)/(len(buttons)+len(plots)),leftbox_end,0.7/(len(buttons)+len(plots))))
            but=matplotlib.widgets.Button(ax,buttons[i]['name'])
            def onclick(A=None,B=None,data=self,type=buttons[i]['name'],fitfun=buttons[i]['fitmethod']):
                data1=data.trimzoomed()
                data1.transform=data.transform
                pylab.figure()
                res=getattr(data1,fitfun).__call__(plot=True)
                listoffits.append({'type':type,'res':res,'time':time.time(),'qmin':data1.q.min(),'qmax':data1.q.max()})
                pylab.gcf().show()
            but.on_clicked(onclick)
        ax=pylab.axes((leftborder,topborder-(len(buttons)+len(plots))*(0.8)/(len(buttons)+len(plots)),leftbox_end,0.7/(len(buttons)+len(plots))*len(plots) ))
        pylab.title('Plot types')
        rb=matplotlib.widgets.RadioButtons(ax,[p['name'] for p in plots],active=7)
        pylab.gcf().blocking=blocking
        if pylab.gcf().blocking: #if blocking mode is desired, put a "Done" button.
            ax=pylab.axes((leftborder,0.03,leftbox_end,bottomborder-0.03))
            b=matplotlib.widgets.Button(ax,"Done")
            fig=pylab.gcf()
            fig.fittingdone=False
            def onclick1(A=None,B=None,fig=fig):
                fig.fittingdone=True
                pylab.gcf().blocking=False
         #       print "blocking should now end"
            b.on_clicked(onclick1)
        pylab.axes((0.45,bottomborder,0.5,0.8))
        def onselectplottype(plottype,q=self['q'],I=self['Intensity'],title=title):
            pylab.cla()
            plotinfo=[d for d in plots if d['name']==plottype][0]
            self.transform=plotinfo['transform']
            getattr(self,plotinfo['plotmethod']).__call__()
            pylab.title(title)
            pylab.gcf().show()
        rb.on_clicked(onselectplottype)
        pylab.title(title)
        self.loglog('.')
        pylab.gcf().show()
        pylab.draw()
        fig=pylab.gcf()
        while blocking:
            fig.waitforbuttonpress()
    #        print "buttonpress"
            if fig.fittingdone:
                blocking=False
                #print "exiting"
        #print "returning"
        return listoffits
    bfg=basicfittinggui;

def returnsSASDict(func):
    """Decorator function for fit functions.
    
    Usage:
    
        @returnsSASDict
        def powerlaw(q,A,alpha):
            return A*np.pow(q,alpha)
        
        will make powerlaw() to return the results in a SASDict.
    """
    @wraps(func)
    def func1(q,*args,**kwargs):
        I=func(q,*args,**kwargs)
        return SASDict(q,I,np.zeros(q.shape))
    return func1

class SASTransform(object):
    def __init__(self):
        pass
    def do_transform(self,q,Intensity,Error,**kwargs):
        raise NotImplementedError('SASTransform is an abstract class. Please derive a class from this, overriding the do_tranform() method!')
    def __call__(self,*args,**kwargs):
        return self.do_transform(*args,**kwargs)
    def xlabel(self,unit=u'1/\xc5'):
        return u'q (%s)' %unit
    def ylabel(self,unit=u'1/cm'):
        return u'Intensity (%s)' % unit

class SASTransformGuinier(SASTransform):
    def __init__(self,qpower=0):
        self._qpower=qpower
    def do_transform(self,q,Intensity,Error,**kwargs):
        d={}
        d['x']=np.power(q,2)
        d['y']=np.log(Intensity*np.power(q,self._qpower))
        d['dy']=np.absolute(Error/Intensity)
        return d
    def xlabel(self, qunit=u'1/\xc5'):
        return u'q^2 (%s^2)' %qunit
    def ylabel(self,Iunit=u'1/cm',qunit=u'1/\xc5'):
        if self._qpower!=0:
            return u'ln (Intensity*q^%d) (ln(%s*%s^%d))' % (self._qpower,Iunit,qunit,self._qpower)
        return u'log Intensity (log %s)' % Iunit

class SASTransformLogLog(SASTransform):
    def __init__(self,xlog=True,ylog=True):
        self._xlog=xlog
        self._ylog=ylog
    def do_transform(self,q,Intensity,Error,**kwargs):
        if self._xlog:
            retq=np.log(q)
        else:
            retq=np.array(q) # make a copy!
        if self._ylog:
            retI=np.log(Intensity)
            retE=np.absolute(Error/Intensity)
        else:
            retI=np.array(Intensity)
            retE=np.array(Error)
        return {'x':retq,'y':retI,'dy':retE}
    def xlabel(self,unit=u'1/\xc5'):
        if self._xlog:
            return u'ln q (ln %s)' %unit
        else:
            return u'q (%s)' %unit
    def ylabel(self,unit=u'1/cm'):
        if self._ylog:
            return u'ln Intensity (ln %s)' % unit
        else:
            return u'Intensity (%s)' % unit

class SASTransformPorod(SASTransform):
    def __init__(self,exponent=4):
        self._exponent=exponent
    def do_transform(self,q,Intensity,Error,**kwargs):
        return {'x':np.power(q,self._exponent),
                'y':np.power(q,self._exponent)*Intensity,
                'dy':np.power(q,self._exponent)*Error}
    def xlabel(self,unit=u'1/\xc5'):
        return u'q^4 (%s^4)' %unit
    def ylabel(self,Iunit=u'1/cm',qunit=u'1/\xc5'):
        return u'q^4*Intensity (%s*%s^4)' % (Iunit,qunit)

class SASTransformShullRoess(SASTransform):
    def __init__(self,r0):
        self._r0=r0
    def do_transform(self,q,Intensity,Error,**kwargs):
        retq=np.log(np.power(q,2)+3/self._r0**2)
        retI=np.log(Intensity)
        retE=np.absolute(Error/Intensity)
        return {'x':retq,'y':retI,'dy':retE}
    def xlabel(self,unit=u'1/\xc5'):
        return u'ln q^2 (ln %s^2)' %unit
    def ylabel(self,Iunit=u'1/cm'):
        return u'ln Intensity (ln %s)' % (Iunit)

class SASTransformZimm(SASTransform):
    def __init__(self):
        pass
    def do_transform(self,q,Intensity,Error,**kwargs):
        retq=np.power(q,2)
        retI=1/Intensity
        retE=Error*retI
        return {'x':retq,'y':retI,'dy':retE}
    def xlabel(self,unit=u'1/\xc5'):
        return u'q^2 (%s^2)' %unit
    def ylabel(self,Iunit=u'1/cm'):
        return u'Reciprocal intensity (1/(%s))' % (Iunit)

        
TransformGuinier=SASTransformGuinier()
TransformGuinierThickness=SASTransformGuinier(2)
TransformGuinierCrosssection=SASTransformGuinier(1)
TransformPorod=SASTransformPorod()
TransformLogLog=SASTransformLogLog(True,True)
TransformSemilogX=SASTransformLogLog(True,False)
TransformSemilogY=SASTransformLogLog(False,True)
TransformLinLin=SASTransformLogLog(False,False)
TransformZimm=SASTransformZimm()

class SASImage(object):
    def __init__(self,A,Aerr=None,param=None,mask=None):
        self._A=A
        self._Aerr=Aerr
        self._param=param
        self._mask=mask
        self._q=None
    def __del__(self):
        object.__delattr__(self,'_A')
        object.__delattr__(self,'_Aerr')
        object.__delattr__(self,'_mask')
        object.__delattr__(self,'_q')
    def __getitem__(self,item):
        try:
            return self._param[item]
        except KeyError:
            raise
    def __getattr__(self,key):
        if key in ['A','Aerr','mask','param']:
            key='_%s'%key
            x=object.__getattribute__(self,key)
            if type(x)==np.ndarray:
                return np.array(x)
            else:
                return x
        if key=='q':
            return object.__getattribute__(self,'_getq').__call__()
        try:
            return object.__getattribute__(self,key)
        except AttributeError:
            if not key.startswith('_'):
                return object.__getattribute__(self,'_%s'%key)
            raise
    def __setattr__(self,key,value):
        if key in ['A','Aerr', 'mask']:
            key='_%s'%key
            value=np.array(value)
            if (value.shape==self.A.shape):
                object.__setattr__(self,key,value)
            elif key=='A':
                object.__setattr__(self,key,value)
                object.__setattr__(self,'Aerr',None)
                object.__setattr__(self,'mask',None)
            else:
                raise ValueError('Cannot broadcast objects to a single shape!')
        else:
            return object.__setattr__(self,key,value)
    def _getq(self):
        if not all([x in self._param for x in ['BeamPosX','BeamPosY','Dist','EnergyCalibrated','PixelSize']]):
            raise ValueError('Cannot calculate q! Some parameters are missing.')
        if self._q is None:
            self._q=utils2d.calculateqmatrix(self.A.astype(np.uint8),self['Dist'],self['EnergyCalibrated'],self['PixelSize'],self['BeamPosX'],self['BeamPosY'])
        return self._q
    def _qscalepossible(self):
        res=False
        if self._param is not None:
            if all([x in self._param.keys() for x in ['BeamPosX','BeamPosY','Dist','EnergyCalibrated','PixelSize']]):
                res=True
        return res
    def plot(self,maxval=np.inf,minval=-np.inf,qs=[],showqscale=True,blackinvalid=False,crosshair=True,zscaling='log'):
        """Plots the scattering image in a 2D coloured plot
        
        Inputs:
            maxval: upper saturation of the colour scale. +inf (default): auto
            minval: lower saturation of the colour scale. -inf (default): auto
            qs: q-values for which concentric circles will be drawn. An empty list by default.
            showqscale: show q-scale on both the horizontal and vertical axes. True by default.
            blackinvalid: True if nonpositive and nonfinite pixels should be
                blacked out. False by default.
            crosshair: True if you want to draw a beam-center testing cross-hair.
                False by default.
            zscaling: 'log' or 'linear' (color scaling)
        """
        tmp=self._A.copy()
        tmp[tmp>maxval]=max(tmp[tmp<=maxval])
        tmp[tmp<minval]=min(tmp[tmp>=minval])
        nonpos=(tmp<=0)
        invalid=-np.isfinite(tmp)
        if zscaling.upper().startswith('LOG'):
            tmp[nonpos]=tmp[tmp>0].min()
            tmp=np.log(tmp);
        tmp[invalid]=tmp[-invalid].min();
        if showqscale:
            if not self._qscalepossible():
                raise ValueError('field *param* not defined in SASImage and q-scaling requested in SASImage.plot().')
            # x: row direction (vertical on plot)
            xmin=0-(self._param['BeamPosX']-1)*self._param['PixelSize']
            xmax=(tmp.shape[0]-(self._param['BeamPosX']-1))*self._param['PixelSize']
            ymin=0-(self._param['BeamPosY']-1)*self._param['PixelSize']
            ymax=(tmp.shape[1]-(self._param['BeamPosY']-1))*self._param['PixelSize']
            qxmin=4*np.pi*np.sin(0.5*np.arctan(xmin/self._param['Dist']))*self._param['EnergyCalibrated']/float(HC)
            qxmax=4*np.pi*np.sin(0.5*np.arctan(xmax/self._param['Dist']))*self._param['EnergyCalibrated']/float(HC)
            qymin=4*np.pi*np.sin(0.5*np.arctan(ymin/self._param['Dist']))*self._param['EnergyCalibrated']/float(HC)
            qymax=4*np.pi*np.sin(0.5*np.arctan(ymax/self._param['Dist']))*self._param['EnergyCalibrated']/float(HC)
            extent=[qymin,qymax,qxmax,qxmin]
            bcrow=0
            bccol=0
        else: #not q-scaling requested
            extent=None
            if (self._param is not None) and ('BeamPosX' in self._param.keys()) and ('BeamPosY' in self._param.keys()):
                bcrow=self._param['BeamPosX']
                bccol=self._param['BeamPosY']
            else:
                bcrow=None
                bccol=None
            extent=[1,tmp.shape[0],1,tmp.shape[1]]
        pylab.imshow(tmp,extent=extent,interpolation='nearest');
        if blackinvalid:
            black=np.zeros((tmp.shape[0],tmp.shape[1],4))
            black[:,:,3][nonpos|invalid]=1
            pylab.imshow(black,extent=extent,interpolation='nearest')
            del black;
        if self._mask is not None:
            white=np.ones((self._mask.shape[0],self._mask.shape[1],4))
            white[:,:,3]=np.array(1-self._mask).astype('float')*0.7
            pylab.imshow(white,extent=extent,interpolation='nearest')
            del white;
        for q in qs:
            a=pylab.gca().axis()
            pylab.plot(q*np.cos(np.linspace(0,2*np.pi,2000)),
                       q*np.sin(np.linspace(0,2*np.pi,2000)),
                       color='white',linewidth=3)
            pylab.gca().axis(a)
        if (self._param is not None) and ('Title' in self._param.keys()) and ('FSN' in self._param.keys()):
            pylab.title("#%d: %s" % (self._param['FSN'], self._param['Title']))
        if crosshair and (bcrow is not None):
            a=pylab.gca().axis()
            pylab.plot([extent[0],extent[1]],[bcrow,bcrow],'-',color='white')
            pylab.plot([bccol,bccol],[extent[2],extent[3]],'-',color='white')
            pylab.gca().axis(a)
        del tmp;
    def radint(self,q=None,returnmask=False):
        q1,I1,err1,Area1,mask1=utils2d.radintC(self._A,self._Aerr,
                                       self._param['EnergyCalibrated'],
                                       self._param['Dist'],
                                       self._param['PixelSize'],
                                       self._param['BeamPosX'],
                                       self._param['BeamPosY'],
                                       self._mask,q,returnavgq=True,
                                       returnmask=returnmask)
        return SASDict(q=q1,Intensity=I1,Error=err1,Area=Area1)
    @staticmethod
    def loadmatfile(matfile):
        a=scipy.io.loadmat(matfile)
        if ('Intensity' in a.keys()) and ('Error' in a.keys()):
            return SASImage(a['Intensity'],a['Error'])
        else:
            vars=[x for x in a.keys() if not ((x.startswith('__') or x.endswith('__')))]
            if len(vars)==1:
                return SASImage(a[vars[0]])
            else:
                raise IOError('Invalid mat file %s'%matfile)
    @staticmethod
    def loadnumpyfile(numpyfile):
        a=np.load(numpyfile)
        if ('Intensity' in a.files) and ('Error' in a.files):
            return SASImage(a['Intensity'],a['Error'])
        else:
            vars=[x for x in a.files if not ((x.startswith('__') or (x.endswith('__'))))]
            if len(vars)==1:
                return SASImage(a[vars[0]])
            else:
                raise IOError('Invalid numpy file %s'%numpyfile)
        
        
    
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
    if isinstance(data,SASDict):
        return data.trimq(qmin,qmax)
    indices=(data['q']<=qmax) & (data['q']>=qmin)
    data1={}
    for k in data.keys():
        data1[k]=data[k][indices]
    return data1


def sasdicts_commonq(*args):
    """Reduce SAS dictionaries to a common q-range. Only the boundaries are checked!
    
    Inputs:
        arbitrary number of SAS dicts
        
    Outputs:
        a list of SAS dicts, in the same order as the arguments
    
    Notes:
        only the smallest and largest q is checked for each SAS dict.
    """
    qmin=-np.inf
    qmax=np.inf
    for a in args:
        if a['q'].min()>qmin:
            qmin=a['q'].min()
        if a['q'].max()<qmax:
            qmax=a['q'].max()
    res=[]
    for a in args:
        res.append(trimq(a,qmin=qmin,qmax=qmax))
    return res

def multsasdict(data,mult,errmult=0):
    """Multiply a SAS dict by a scalar, possibly with error propagation
    
    Inputs:
        data: SAS dictionary
        mult: scalar factor
        errmult: error of scalar factor
    
    Outputs:
        a dictionary, multiplied by the scalar factor. Fields 'Intensity' and
        'Error' are treated, other ones are copied.
    """
    if isinstance(data,SASDict):
        return data*(mult,errmult)
    newdict={}
    for i in data.keys():
        newdict[i]=data[i]
    newdict['Error']=np.sqrt((newdict['Intensity']*errmult)**2+(newdict['Error']*mult)**2)
    newdict['Intensity']=newdict['Intensity']*mult
    return newdict

def sortsasdict(data,*args):
    """Sorts 1D SAS dicts.
    
    Inputs:
        data: SAS dict
        *args: key names, sorting sequence
    
    Output:
        sorted dict
    """
    if isinstance(data,SASDict):
        return data.sort(args)
#    print "Sorting SAS dict."
    if len(args)==0:
        args='q'
    #create a structured array. This may look like the darkest voodoo magic,
    # but it isn't. Check the help text of np.sort and zip. Also the tuple
    # expansion operator (*) is used.
    array=np.array(zip(*(data.values())),dtype=zip(data.keys(),[np.double]*len(data.keys())))
    sorted=np.sort(array,order=args)
    ret={}
    for k in data.keys():
        ret[k]=sorted[k]
    return ret
def combinesasdicts(*args,**kwargs):
    """Combines 1D SAS dictionaries.
    
    Inputs:
        arbitrary number of SAS dictionaries
        
    Allowed keyword arguments:
        'accordingto': the key name in the dicts according to which the combining
            will take place. Usually (default) 'q'.
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
    try:
        accordingto=kwargs['accordingto']
    except KeyError:
        accordingto='q'
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
            newdict[i]=d1[i].copy() # make a copy of d1 and add d2 later.
        # add elements of d2:
        for i in range(len(d2[accordingto])): # for each item in d2:
            idx=(newdict[accordingto]==d2[accordingto][i]) #find if the current 'q' value in d2 is already present in newdict
            if idx.sum()==0: # if not:
                for j in d2.keys(): # append.
                    newdict[j]=np.append(newdict[j],d2[j][i])
            for j in d2.keys():
                if j=='Error':
                    newdict['Error'][idx]=np.sqrt(newdict['Error'][idx]**2+d2['Error'][i]**2)*0.5
                elif j==accordingto:
                    continue
                else:
                    newdict[j][idx]=(newdict[j][idx]+d2[j][i])*0.5
        d1=newdict
#    print "combine: new length is",len(newdict[accordingto])
    return sortsasdict(newdict)
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
    c=[x for x in a if x in b]
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
    if isinstance(data,SASDict):
        return data
    d1={}
    for k in data.keys():
        if hasattr(data[k],'flatten'):
            d1[k]=data[k].flatten()
    return d1
def sanitizeint(data,accordingto='Intensity'):
    """Remove points with nonpositive values of a given field from 1D SAXS dataset
    
    Input:
        data: 1D SAXS dictionary
        accordingto: the name of the key where the nonpositive values should be
            checked for. Defaults to 'Intensity'
        
    Output:
        a new dictionary of which the points with nonpositive values of a given
            key (see parameter 'accordingto') are omitted
    """
    if isinstance(data,SASDict):
        return data.sanitize(accordingto)
    indices=(data[accordingto]>0)
    data1={}
    for k in data.keys():
        if type(data[k])!=type(data[accordingto]):
            continue
        if type(data[k])==np.ndarray:
            if data[k].shape!=data[accordingto].shape:
                continue
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
    """Calculate the error of inv(A) according to squared error
    propagation.
    
    Inputs:
        A: The matrix (square shaped)
        DA: The error of the matrix (same size as A)
    
    Output:
        The error of the inverse matrix
    """
    B=np.linalg.linalg.inv(A);
    return np.sqrt(np.dot(np.dot(B**2,DA**2),B**2))
    
def multiple(list,comparefun=(lambda a,b:(a==b))):
    """Select elements occurring more than one times in the list
    
    Inputs:
        list: list
        comparefun: comparing function. By default the '==' operator is used.
    
    Outputs:
        a list with the non-unique elements (each only once)
    """
    newlist=[]
    ul=unique(list,comparefun=comparefun)
    for u in ul:
        if len([l for l in list if comparefun(l,u)])>1:
            newlist.append(u)
    return newlist
def classify_params_fields(params, *args):
    """Classify parameter structures according to field names.
    
    Inputs:
        params: list of parameter dictionaries
        variable arguments: either strings or functions (callables).
            strings: They should be keys available in all of the parameter
                structures found in params. Distinction will be made regarding
                these.
            callables: they should accept param structures and return something
                (string, value or as you like). Distinction will be made on the
                returned value.
    
    Output:
        A list of lists of parameter structures
        
    Example:
        a typical use scenario is to read logfiles from a range of FSNs. Usually
        one wants to summarize measurements made from the same sample, using
        the same energy and at the same sample-to-detector distance. In this
        case, classify_params_fields can be useful:
        
        classify_params_fields(params,'Title','Energy','Dist')
        
        will result in a list of lists. Each sublist will contain parameter
        dictionaries with the same 'Title', 'Energy' and 'Dist' fields.
    """
    if len(args)<1: # fallback, if no field names were given
        return [params]
    list=[]
    #classify according to the first argument
    if hasattr(args[0],'__call__'): # if it is callable, use args[0](p) for classification
        valuespresent=unique([args[0](p) for p in params])
        for val in valuespresent:
            list.append([p for p in params if args[0](p)==val])
    else: # if it is not callable, assume it is a string, treat it as a key
        valuespresent=unique([p[args[0]] for p in params])
        for val in valuespresent:
            list.append([p for p in params if p[args[0]]==val])
    # now list contains sub-lists, containing parameters grouped according to
    # the first field in *args. Let's group these further, by calling ourselves
    # with a reduced set as args
    toplist=[]
    for l in list:
        toplist.extend(classify_params_fields(l,*(args[1:])))
    return toplist