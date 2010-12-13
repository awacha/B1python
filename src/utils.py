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
import string

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
        self._dict={'q':None,'Intensity':None,'Error':None}
        self._dict['q']=np.array(q)
        self._transform=None
        self._plotaxes=None
        if len(Intensity)!=len(self._dict['q']):
            raise ValueError('Intensity should be of the same length as q!')
        self._dict['Intensity']=np.array(Intensity)
        if Error is not None:
            if len(Error)!=len(self._dict['q']):
                raise ValueError('Error, if defined, should be of the same length as q!')
            self._dict['Error']=np.array(Error)
        for k in kwargs.keys():
            if k in ['q','Intensity','Error','s']:
                raise ValueError('%s cannot appear as a variable in a SAS dictionary.'%k)
            if len(kwargs[k])!=len(self._dict['q']):
                raise ValueError('Argument %s should be of the same length as q!'%k)
            self._dict[k]=np.array(kwargs[k])
        SASDict.__instances+=1
    def __getattr__(self,key):
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
        if key=='_dict':
            return object.__setattr__(self,key,value)
        elif key in self._dict.keys():
            if key=='q':
                self._setq(value)
            elif key=='s':
                self._setq(value*2*np.pi)
            elif key in ['x','y','dy']:
                raise ValueError('Attribute %s is read-only!'%key)
            else:
                if len(value)==len(self._dict['q']):
                    self._dict[key]=np.array(value)
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
        return self.__getattr__(key)
    def __setitem__(self,key,value):
        return self.__setattr__(key,value)
    def _setq(self,q1):
        q1=np.array(q1)
        if (len(self._dict['q'])==len(q1)):
            self._dict['q']=q1
        else:
            # new q differs, Intensity, Error and Area have to be reset.
            del self._dict
            self._dict={'q':q1}
    def keys(self):
        return self._dict.keys()
    def __len__(self):
        return self._dict.__len__()
    def values(self):
        return self._dict.values()
    def save(self,filename):
        f=open(filename,'wt')
        f.write('#%s\n'%string.join([str(k) for k in self.keys()]))
        np.savetxt(f,np.array(self))
        f.close()
    def copy(self):
        return SASDict(self)
    def trimq(self,qmin=-np.inf,qmax=np.inf,inplace=False):
        """Trim the 1D scattering data to a given q-range
        
        Inputs:
            data: scattering data
            qmin: lowest q-value to include (default: ignore)
            qmax: highest q-value to include (default: ignore)

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
        return self.trimq(qmin=smin*2*np.pi,qmax=smax*2*np.pi,*args,**kwargs)
    def loglog(self,*args,**kwargs):
        self.do_transform()
        pylab.loglog(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def semilogy(self,*args,**kwargs):
        self.do_transform()
        pylab.semilogy(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def semilogx(self,*args,**kwargs):
        self.do_transform()
        pylab.semilogx(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def plot(self,*args,**kwargs):
        self.do_transform()
        pylab.plot(self.x,self.y,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def errorbar(self,*args,**kwargs):
        self.do_transform()
        pylab.errorbar(self.x,self.y,self.dy,*args,**kwargs)
        if self._transform is not None:
            pylab.xlabel(self._transform.xlabel())
            pylab.ylabel(self._transform.ylabel())
        self._plotaxes=pylab.gca()
    def _check_compat(self,x,die=False):
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
        if isinstance(x,tuple) and len(x)==2:
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
    def __array__(self):
        a=np.array(zip(*(self.values())),dtype=zip(self.keys(),[np.double]*len(self.keys())))
        return a
    def sort(self,order='q'):
        a=self.__array__()
        sorted=np.sort(a,order=order)
        for k in self._dict.keys():
            self._dict[k]=sorted[k]
        return self
    def sanitize(self,accordingto='Intensity',thresholdmin=0,thresholdmax=np.inf,function=None):
        if hasattr(function,'__call__'):
            indices=function(self._dict[accordingto])
        else:
            indices=(self._dict[accordingto]>thresholdmin) & (self._dict[accordingto]<thresholdmax)
        for k in self._dict.keys():
            self._dict[k]=self._dict[k][indices]
        return self
    def modulus(self,exponent=0,errorrequested=False):
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
        return self.modulus(errorrequested=errorrequested)
    def __del__(self):
        for k in self._dict.keys():
            del self._dict[k]
        del self._dict
        SASDict.__instances-=1
        #print "An instance of SASDict has been disposed of. Remaining instances:",SASDict.__instances
    def do_transform(self):
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
        p,cov_x,infodict,mesg,ier=scipy.optimize.leastsq(lambda p:(function(self._dict['q'],*p)-self._dict['Intensity'])/self._dict['Error'],params_initial,full_output=1,**kwargs)
        chisquare=(infodict['fvec']**2).sum()
        degrees_of_freedom=len(self._dict['q'])-len(p)
        if ier<1 or ier>4:
            raise ValueError('Fitting did not succeed. Reason: %s'%mesg)
        errors=[ np.sqrt(cov_x[i,i]*chisquare/degrees_of_freedom) for i in range(len(p))]
        if full_output:
            return p,errors,function(self._dict['q'],*p),chisquare,degrees_of_freedom
        else:
            return p,errors
    def _fitting_base(self,function,transformdatasettolinear=None, transformparamfromlinear=None,params_initial=None,plotinfo=None):
        #Base fitting function, for internal usage only.
        # parameters:
        #    function is the fitting function to be fitted. Will be forwarded to self.fit().
        #    transformdatasettolinear: this is a callable, eg. a subclass of SASTransform, which
        #        will transform q,Intensity to a line, if possible. If such a linearization is
        #        not viable, should be None
        #    transformparamfromlinear: a callable, which transforms the parameters of a fitted line
        #        back to values usable for a proper least-squares fit of the function. Can be
        #        None, if no such transformation is needed.
        #    params_initial: first guess for the parameters. Can be None if
        #        transformdatasettolinear is defined. In other cases, this should either be a list
        #        which will be forwarded to self.fit(). Or, it can be a callable, accepting
        #        q and Intensity and returning a list of initial parameters forwardable to self.fit()
        #    plotinfo: None, a dictionary, or anything.
        #        a) None: no plotting is desired.
        #        b) dictionary: fields needed are 'funcname' (description of the function) and
        #            'paramnames' (list of names of the parameters). An optional field is 
        #            'otherstringforlegend', which can be a string, which will be appended to
        #            the legend, or a callable, which provides that string. Then its arguments
        #            will be (q,Intensity,Error,paramsfitted,errorofparamsfitted).
        #
        # Returns: params,errors,curve,chi2,dof
        #    params: values fitted
        #    errors: errors of the fitted params
        #    curve: the fitted curve
        #    chi2: chi-squared (not reduced)
        #    dof: degrees of freedom
        
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
        # if linearization did not succeed, check if params_initial is supplied
        elif params_initial is None:
            raise ValueError('params_initial should not be None if transformdatasettolinear is None.')
        # if params_initial is callable, call it to get the first guess
        elif hasattr(params_initial,'__call__'): #in this case, it is a guessing function
            params_initial=params_initial(self._dict['q'],self._dict['Intensity'])
        # at this point, we have params_initial. Fitting can be carried out.
        params,errors,curve,chi2,dof=self.fit(function,params_initial,full_output=True)
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
                if hasattr(plotinfo,'otherstringforlegend'):
                    if hasattr(plotinfo['otherstringforlegend'],'__call__'):
                        fittinglog=fittinglog+plotinfo['otherstringforlegend'](self._dict['q'],self._dict['Intensity'],self._dict['Error'],params,errors)
                    else:
                        fittinglog=fittinglog+plotinfo['otherstringforlegend']
            #plot the legend
            pylab.text(0.95,0.95,fittinglog,ha='right',va='top',transform=pylab.gca().transAxes)
        return params,errors,curve,chi2,dof
        
    def guinierfit(self,qpower=0,plot=True):
        divisor=3-qpower
        fitfunction=lambda q,G,R:np.power(q,qpower)*G*np.exp(-q**2*R**2/divisor)
        paramtransform=lambda lnG,mR2div:(np.exp(lnG),np.sqrt(-divisor*mR2div))
        if plot:
            def legend_addendum(q,I,E,G,R,dG,dR):
                return 'q_max*R = %g\n'%(q.max()*R)
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
        return self.guinierfit(qpower=2,*args,**kwargs)
    def guiniercrosssectionfit(self,*args,**kwargs):
        return self.guinierfit(qpower=1,*args,**kwargs)
    def powerlawfit(self,plot=True):
        fitfunction=lambda q,A,B:np.power(q,B)*A
        paramtransform=lambda lnA,B:(np.exp(lnA),B)
        if plot:
            plotinfo={'funcname':'A*q^B',
                      'paramnames':['A','B']}
        else:
            plotinfo=None
        p,e,curve,chi2,dof=self._fitting_base(fitfunction,
                                              SASTransformLogLog(),
                                              paramtransform,
                                              params_initial=None,
                                              plotinfo=plotinfo)
        return p,e
    def powerlawconstantbackgroundfit(self,plot=True):
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
        
    def trimzoomed(self,inplace=False):
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
        
TransformGuinier=SASTransformGuinier()
TransformGuinierThickness=SASTransformGuinier(2)
TransformGuinierCrosssection=SASTransformGuinier(1)
TransformPorod=SASTransformPorod()
TransformLogLog=SASTransformLogLog(True,True)
TransformSemilogX=SASTransformLogLog(True,False)
TransformSemilogY=SASTransformLogLog(False,True)
TransformLinLin=SASTransformLogLog(False,False)


class SASImage(object):
    def __init__(self,A,Aerr=None,param=None,mask=None):
        self._A=A
        self._Aerr=Aerr
        self._param=param
        self._mask=mask
    def __getitem__(self,item):
        try:
            return self._param[item]
        except KeyError:
            raise
    def _getIntensity(self):
        return np.array(self._A)
    def _setIntensity(self,A1):
        if (self._A is None) or (self._A.shape==A1.shape):
            self._A=np.array(A1)
        else:
            raise ValueError('Incompatible shape for Intensity matrix!')
    def _getError(self):
        return np.array(self._Aerr)
    def _setError(self,A1):
        if (self._Aerr is None) or (self._Aerr.shape==A1.shape):
            self._Aerr=np.array(A1)
        else:
            raise ValueError('Incompatible shape for Error matrix!')
    def _getq(self):
        raise NotImplementedError
    Intensity=property(fget=_getIntensity,fset=_setIntensity,doc='Two-dimensional scattering intensity')
    Error=property(fget=_getError,fset=_setError,doc='Absolute error of two-dimensional scattering intensity')
    q=property(fget=_getq,doc='length of momentum transfer vector')
    

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
    """Calculate the error of np.inv(A) according to squared error
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