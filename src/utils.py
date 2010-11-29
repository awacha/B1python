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
    def __init__(self,**kwargs):
        self._q=None
        self._Intensity=None
        self._Error=None
        self._Area=None
        try:
            self._q=np.array(kwargs['q'])
        except KeyError:
            raise ValueError('SASDict has to be initialized with q given.')
        try:
            if len(kwargs['Intensity'])!=len(self._q):
                raise ValueError('Length of Intensity should be equal to length of q.')
            self._Intensity=np.array(kwargs['Intensity'])
        except TypeError:
            self._Intensity=None
        except KeyError:
            raise ValueError('SASDict has to be initialized with Intensity given.')
        try:
            if len(kwargs['Error'])!=len(self._q):
                raise ValueError('Length of Error should be equal to length of q.')
            self._Error=np.array(kwargs['Error'])
        except TypeError:
            self._Error=None
        except KeyError:
            self._Error=np.zeros(self._Intensity.shape)
        try:
            if len(kwargs['Area'])!=len(self._q):
                raise ValueError('Length of Area should be equal to length of q.')
            self._Area=np.array(kwargs['Area'])
        except TypeError:
            self._Area=None
        except KeyError:
            self._Area=np.zeros(self._Intensity.shape)
        SASDict.__instances+=1
    def get_q(self):
        return self._q
    def get_Intensity(self):
        return self._Intensity
    def get_Error(self):
        return self._Error
    def get_Area(self):
        return self._Area
    def get_s(self):
        return self._q/(2*np.pi)
    def set_q(self,q1):
        q1=np.array(q1)
        if (self._q is None) or (len(self._q)==len(q1)):
            self._q=q1
        else:
            # new q differs, Intensity, Error and Area have to be reset.
            self._Intensity=None
            self._Error=None
            self._Area=None
    def set_s(self,s1):
        return self.set_q(s1/(2*np.pi))
    def set_Intensity(self,I1):
        I1=np.array(I1)
        if (len(self._q)==len(I1)):
            self._Intensity=I1
        else:
            raise ValueError("Intensity should be of the same length as q.")
    def set_Error(self,E1):
        E1=np.array(E1)
        if (len(self._q)==len(E1)):
            self._Error=E1
        else:
            raise ValueError("Error should be of the same length as q.")
    def set_Area(self,A1):
        A1=np.array(A1)
        if (len(self._q)==len(A1)):
            self._Area=A1
        else:
            raise ValueError("Area should be of the same length as q.")
    q=property(fget=get_q,fset=set_q,doc='Scattering variable, 4*pi*sin(theta)/lambda')
    s=property(fget=get_s,fset=set_s,doc='Scattering variable, 2*sin(theta)/lambda')
    Intensity=property(fget=get_Intensity,fset=set_Intensity,doc='Scattered intensity')
    Error=property(fget=get_Error,fset=set_Error,doc='Absolute error of scattered intensity')
    Area=property(fget=get_Area,fset=set_Area,doc='Effective area during integration')
    def save(self,filename):
        f=open(filename,'wt')
        f.write('#%s\n'%string.join([str(k) for k in self.keys()]))
        np.savetxt(f,np.array(self))
        f.close()
    def keys(self):
        list=[]
        if self._q is not None:
            list.append('q')
        if self._Intensity is not None:
            list.append('Intensity')
        if self._Error is not None:
            list.append('Error')
        if self._Area is not None:
            list.append('Area')
        return list
    def values(self):
        list=[]
        if self._q is not None:
            list.append(self.get_q())
        if self._Intensity is not None:
            list.append(self.get_Intensity())
        if self._Error is not None:
            list.append(self.get_Error())
        if self._Area is not None:
            list.append(self.get_Area())
        return list
    def __getitem__(self,item):
        if item=='q':
            return self.q
        if item=='Intensity':
            return self.Intensity
        if item=='Error':
            return self.Error
        if item=='Area':
            return self.Area
        raise KeyError(item)
    def copy(self):
        return SASDict(self)
    def trimq(self,qmin=-np.inf,qmax=np.inf):
        """Trim the 1D scattering data to a given q-range
        
        Inputs:
            data: scattering data
            qmin: lowest q-value to include (default: ignore)
            qmax: highest q-value to include (default: ignore)

        Intensity, Error and Area (if present) will be trimmed.
        """
        indices=(self._q<=qmax) & (self._q>=qmin)
        self._q=self._q[indices]
        if self._Intensity is not None:
            self._Intensity=self._Intensity[indices]
        if self._Error is not None:
            self._Error=self._Error[indices]
        if self._Area is not None:
            self._Area=self._Area[indices]
        return self
    def trims(self,smin=-np.inf,smax=np.inf):
        return self.trimq(qmin=smin*2*np.pi,qmax=smax*2*np.pi)
    def loglog(self,*args,**kwargs):
        pylab.loglog(self._q,self._Intensity,*args,**kwargs)
    def semilogy(self,*args,**kwargs):
        pylab.semilogy(self._q,self._Intensity,*args,**kwargs)
    def semilogx(self,*args,**kwargs):
        pylab.semilogx(self._q,self._Intensity,*args,**kwargs)
    def plot(self,*args,**kwargs):
        pylab.plot(self._q,self._Intensity,*args,**kwargs)
    def errorbar(self,*args,**kwargs):
        pylab.errorbar(self._q,self._Intensity,self._Error,*args,**kwargs)
    def _check_compat(self,x,die=False):
        if len(x._q)!=len(self._q):
            if die:
                raise ValueError('Incompatible SAS dicts (q-scales have different lengths)!')
            return False
        if (2*np.absolute(x._q-self._q)/(x._q+self._q)).sum()/len(self._q)>SASDict.qtolerance:
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
            self._q=0.5*(self._q+x._q)
            self._Intensity=self._Intensity*x._Intensity
            self._Error=np.sqrt((x._Intensity*self._Error)**2+(self._Intensity*x._Error)**2)
            self._Area=None
            return self
        else:
            val=x
            err=0
        self._Error=np.sqrt((self._Intensity*err)**2+(self._Error*val)**2)
        self._Intensity=self._Intensity*val
        return self
    def __idiv__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self*=1/x
            return self
        else:
            val=x
            err=0
        self._Error=np.sqrt((self._Intensity/(val*val)*err)**2+(self._Error/val)**2)
        self._Intensity=self._Intensity/val
        return self
    def __iadd__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            self._q=0.5*(self._q+x._q)
            self._Intensity=self._Intensity+x._Intensity
            self._Error=np.sqrt(self._Error**2+x._Error**2)
            self._Area=None
            return self
        else:
            val=x
            err=0
        self._Error=np.sqrt((err)**2+(self._Error)**2)
        self._Intensity=self._Intensity+val
        return self
    def __isub__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            self._q=0.5*(self._q+x._q)
            self._Intensity=self._Intensity-x._Intensity
            self._Error=np.sqrt(self._Error**2+x._Error**2)
            self._Area=None
            return self
        else:
            val=x
            err=0
        self._Error=np.sqrt((err)**2+(self._Error)**2)
        self._Intensity=self._Intensity-val
        return self
    def __mul__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            q1=0.5*(self._q+x._q)
            Intensity1=self._Intensity*x._Intensity
            Error1=np.sqrt((x._Intensity*self._Error)**2+(self._Intensity*x._Error)**2)
            Area1=None
            return SASDict(q=q1,Intensity=Intensity1,Error=Error1,Area=Area1)
        else:
            val=x
            err=0
        err=np.sqrt((self._Intensity*err)**2+(self._Error*val)**2)
        val=self._Intensity*val
        return SASDict(q=self._q,Intensity=val,Error=err,Area=self._Area)
    def __div__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            return self*(1/x)
        else:
            val=x
            err=0
        err=np.sqrt((self._Intensity/(val*val)*err)**2+(self._Error/val)**2)
        val=self._Intensity/val
        return SASDict(q=self._q,Intensity=val,Error=err,Area=self._Area)
    __itruediv__=__idiv__
    def __add__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            return SASDict(q=0.5*(self._q+x._q),Intensity=self._Intensity+x._Intensity,Error=np.sqrt(self._Error**2+x._Error**2),Area=None)
        else:
            val=x
            err=0
        err=np.sqrt((err)**2+(self._Error)**2)
        val=self._Intensity+val
        return SASDict(q=self._q,Intensity=val,Error=err,Area=self._Area)
    def __sub__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            self._check_compat(x,die=True)
            return SASDict(q=0.5*(self._q+x._q),Intensity=self._Intensity-x._Intensity,Error=np.sqrt(self._Error**2+x._Error**2),Area=None)
        else:
            val=x
            err=0
        err=np.sqrt((err)**2+(self._Error)**2)
        val=self._Intensity-val
        return SASDict(q=self._q,Intensity=val,Error=err,Area=self._Area)
    def __neg__(self):
        return SASDict(q=self._q,Intensity=-self._Intensity,Error=self._Error,Area=self._Area)
    __itruediv__=__idiv__
    __truediv__=__div__
    __rmul__=__mul__
    def __rdiv__(self,x):
        if isinstance(x,tuple) and len(x)==2:
            val=x[0]
            err=x[1]
        elif isinstance(x,SASDict):
            return x*(1/self)
        else:
            val=x
            err=0
        err=np.sqrt((err/self._Intensity)**2+(val/(self._Error)**2)**2)
        val=self._Intensity/val
        return SASDict(q=self._q,Intensity=val,Error=err,Area=self._Area)
    __rtruediv__=__truediv__
    __radd__=__add__
    def __rsub__(self,x):
        return -(self-x)
    def __pow__(self,exponent,modulus=None):
        if modulus is not None:
            return NotImplemented # this is hard to implement for SAS curves.
        if exponent==0:
            return SASDict(q=self._q,Intensity=np.zeros(self._q.shape),Error=np.zeros(self._q.shape),Area=self._Area)
        else:
            return SASDict(q=self._q,Intensity=np.power(self._Intensity,exponent),
                           Error=self._Error*np.absolute((exponent)*np.power(self._Intensity,exponent-1)))
    def __array__(self):
        a=np.array(zip(*(self.values())),dtype=zip(self.keys(),[np.double]*len(self.keys())))
        return a
    def sort(self,order='q'):
        a=self.__array__()
        sorted=np.sort(a,order=order)
        if self._q is not None:
            self._q=sorted['q']
        if self._Intensity is not None:
            self._Intensity=sorted['Intensity']
        if self._Error is not None:
            self._Error=sorted['Error']
        if self._Area is not None:
            self._Area=sorted['Area']
        return self
    def sanitize(self,accordingto='Intensity',thresholdmin=0,thresholdmax=np.inf,function=None):
        if hasattr(function,'__call__'):
            indices=function(self.__getattribute__(accordingto))
        else:
            indices=(self.__getattribute__(accordingto)>thresholdmin) & (self.__getattribute__(accordingto)<thresholdmax)
        if self._q is not None:
            self._q=self._q[indices]
        if self._Intensity is not None:
            self._Intensity=self._Intensity[indices]
        if self._Error is not None:
            self._Error=self._Error[indices]
        if self._Error is not None:
            self._Area=self._Area[indices]
        return self
    def modulus(self,exponent=0,errorrequested=False):
        x=self._q
        y=self._Intensity*self._q**exponent
        err=self._Error*self._q**exponent
        m=np.trapz(y,x)
        dm=errtrapz(x,err)
        if errorrequested:
           return (m,dm) 
        else:
            return m
    def integral(self,errorrequested=False):
        return self.modulus(errorrequested=errorrequested)
    def __len__(self):
        return len(self.keys())
    def __del__(self):
        if self._q is not None:
            del self._q
        if self._Intensity is not None:
            del self._Intensity
        if self._Error is not None:
            del self._Error
        if self._Area is not None:
            del self._Area
        SASDict.__instances-=1
#        print "An instance of SASDict has been disposed of. Remaining instances:",SASDict.__instances

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