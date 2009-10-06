"""B1python: various Pythonic functions for Small-Angle X-ray Scattering
analysis. Created by Andras Wacha for beamline B1 @HASYLAB/DESY, Hamburg,
Germany, but hopefully other scientists can benefit from these too. These
functions are partially based on Ulla Vainio's Matlab(R) scripts for B1 data
analysis.

Legal note: I donate these scripts to the public. You may freely use,
    distribute, modify the sources, as long as the legal notices remain in
    place, and no functionality is removed. However, using it as a main part
    for commercial (ie. non-free) software is not allowed (you got it free,
    you should give it free). The author(s) take no warranty on this program,
    nor on the outputs (like the copyleft from GNU). However, if you modify
    this, find a bug, suggest a new feature to be implemented, please feel
    free to contact the authors (Andras Wacha: awacha at gmail dot com).

Note for developers: If you plan to enhance this program, please be so kind to
    contact the original author (Andras Wacha: awacha at gmail dot com). I ask
    this because I am happy when I hear that somebody finds my work useful for
    his/her tasks. And for the coding style: please comment every change by
    your monogram/nickname and the date. And you should add your name,
    nickname and e-mail address to the authors clause in this notice as well.
    You deserve it.

General concepts:

    As it was already said, these functions are based on Matlab(R) scripts.
        It was kept iln mind therefore to retain Compatibility to the Matlab(R)
        version more or less. However, we are in Python and it would be
        foolish not to use the possibilities and tools it provides. In the
        next lines I note the differences.
    
    the numbering of the pixels of the detector starts from 1, in case of the
        beam coordinates, thus the values from the Matlab(R)-style
        intnorm*.log files are usable without modification. On contrary to
        Matlab(R), Python counts the indices from 0, so to get the real value
        of the beam coordinates, one should look at the pixel bcx-1,bcy-1. All
        functions in this file which need the beam coordinates as input or
        supply them as output, handle this automatically.
    
    the radially averaged datasets are in a data dictionary with fields 'q',
        'Intensity', 'Error' and possible 'Area'. The approach is similar to
        that of the Matlab(R) version, however in Python, dictionary is the
        best suited container.
        
    the mask matrices and corrected 2D data are saved to and loaded from mat
        files.
        
    if a function classifies measurements depending on energies, it uses
        always the uncalibrated (apparent) energies, which were set up in the
        measurement program at the beamline.
        
    the difference between header and param structures (dictionaries in
        Python) are planned to be completely removed. In the Matlab(R) version
        the fields of a header structure consist a subset of the ones of the
        param structure. When the data evaluation routines get implemented,
        the header dictionary extracted from the input datasets will get
        extended during the evaluation run by newly calculated values.

Dependencies:
    This set of functions depend---apart from the standard Python library---on
    various 3rd party modules. A complete list of these:
        matplotlib (pylab)
        scipy
    
A final note: functions labelled by EXPERIMENTAL!!!! in the online help-text
    are REALLY experimental. When I say experimental, I mean experimental,
    possibly not fully implemented code. No kidding. They are not thoroughly
    tested, so use on your own risk. They may not do what you expect, or they
    won't do anything at all. You have been warned. But if you are really
    curious, you can look at their source code... :-)
"""

import string
import pylab
import scipy
import scipy.io
import types
import zipfile
import gzip
import Tkinter
import sys
import time
import os
import shutil
import matplotlib.widgets
import matplotlib.nxutils
from scipy.optimize.minpack import leastsq
from scipy.optimize import fmin
import scipy.special
import scipy.stats.stats
import scipy.interpolate
import radint_ng

HC=12396.4 #Planck's constant times speed of light, in eV*Angstrom units
def theorspheres(qrange, spheres):
    """Calculate the theoretical scattering intensity of the sphere structure
    
    Inputs:
        qrange: vector of q values or just a single value
        spheres: sphere structure. Could be:
            1) a single number: it is then assumed that the scattering of a single
                sphere with this radius is the scatterer
            2) a python list of numbers: a sphere population with these radii is assumed
            3) a numpy array: the columns correspond to x, y, z, R respectively.
                If supplied, the 5th column is the real, the 6th is the imaginary
                part of the scattering length density.
            
    Output:
        a vector of the same size that of qrange. It contains the scattering
        intensities.
    """
    def fsphere(q,R):
        return 1/q**3*(pylab.sin(q*R)-q*R*pylab.cos(q*R))
    
    if (type(qrange)!=types.ListType) and (type(qrange)!=pylab.ndarray):
        qrange=[qrange]
    if (type(spheres)!=types.ListType) and (type(spheres)!=pylab.ndarray):
        spheres=[spheres]
    Intensity=pylab.zeros(qrange.size)
    if (type(spheres)==types.ListType):
        for i in len(spheres):
            Intensity=Intensity+fsphere(qrange,spheres[i])**2
    if (type(spheres)==pylab.ndarray):
        if spheres.shape[1]<4:
            raise ValueError("Not enough columns in spheres structure")
        if spheres.shape[1]<5:
            s1=pylab.zeros((spheres.shape[0],6))
            s1[:,0:4]=spheres
            s1[:,4]=1;
            s1[:,5]=0;
            spheres=s1;
        for i in range(spheres.shape[0]):
            f1=fsphere(qrange,spheres[i,3])
            Intensity+=(spheres[i,4]**2+spheres[i,5]**2)*f1**2;
            for j in range(i+1,spheres.shape[0]):
                f2=fsphere(qrange,spheres[j,3])
                dist=pylab.sqrt((spheres[i,0]-spheres[j,0])**2+(spheres[i,1]-spheres[j,1])**2+(spheres[i,2]-spheres[j,2])**2)
                if dist!=0:
                    fact=pylab.sin(qrange*dist)/(qrange*dist)
                else:
                    fact=1;
                Intensity+=2*(spheres[i,4]*spheres[j,4]+spheres[i,5]*spheres[j,5])*f1*f2*fact;
    return Intensity            
                
def polartransform(data,r,phi,origx,origy):
    """Calculates a matrix of a polar representation of the image.
    
    Inputs:
        data: the 2D matrix
        r: vector of polar radii
        phi: vector of polar angles
        origx: the x (row) coordinate of the origin
        origy: the y (column) coordinate of the origin
    Outputs:
        pdata: a matrix of len(phi) rows and len(r) columns which contains the
            polar representation of the image.
    """
    pdata=pylab.zeros((len(phi),len(r)))
    for i in range(len(phi)):
        for j in range(len(r)):
            x=origx-1+r[j]*pylab.cos(phi[i]);
            y=origy-1+r[j]*pylab.sin(phi[i]);
            if (x>=0) and (y>=0) and (x<data.shape[0]) and (y<data.shape[1]):
                pdata[i,j]=data[x,y];
    return pdata
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
    a,b,aerr,berr=linfit(energymeas,energycalib)
    if type(energy1)==pylab.np.ndarray:
        return a*energy1+b
    elif type(energy1)==types.ListType:
        return [a*e+b for e in energy1]
    else:
        return a*energy1+b
def rebin(data,qrange):
    """Rebin 1D data. Note: if 2D data is present, reintegrateB1 is more accurate.
    
    Inputs:
        data: one or more (=list) of scattering data dictionaries (with fields
            'q', 'Intensity' and 'Error')
        qrange: the new q-range to which the binning should be carried out.
    
    Outputs:
        the re-binned dataset in a scattering data dictionary
    """
    qrange=pylab.array(qrange)
    if type(data)!=types.ListType:
        data=[data]
    data2=[];
    counter=0;
    for d in data:
        #print counter
        counter=counter+1
        tmp={};
        tmp['q']=qrange
        tmp['Intensity']=pylab.interp(qrange,d['q'],d['Intensity'])
        tmp['Error']=pylab.interp(qrange,d['q'],d['Error'])
        data2.append(tmp)
    return data2;
def energiesfromparam(param):
    """Return the (uncalibrated) energies from the measurement files
    
    Inputs:
        param dictionary
        
    Outputs:
        a list of sorted energies
    """
    return unique([p['Energy'] for p in param],lambda a,b:(abs(a-b)<2))
def samplenamesfromparam(param):
    """Return the sample names
    
    Inputs:
        param dictionary
        
    Output:
        a list of sorted sample titles
    """
    return unique([p['Title'] for p in param])
def findbeam(data,orig_initial,mask=None,maxiter=20):
    """Find beam center
    
    Inputs:
        data: scattering matrix
        orig_initial: estimated value for x (row) and y (column) coordinates
            of the beam center, starting from 1.
        mask: mask matrix. If None, nothing will be masked. Otherwise it should be
            of the same size as data. Nonzero means non-masked.
    Output:
        a vector of length 2 with the x and y coordinates of the origin.
    """
    orig=pylab.array(orig_initial)
    if mask is None:
        mask=pylab.ones(data.shape)
    def targetfunc(orig,data,mask):
        c1,nc1=imageint(data,orig,mask,35,20)
        c2,nc2=imageint(data,orig,mask,35+90,20)
        c3,nc3=imageint(data,orig,mask,35+180,20)
        c4,nc4=imageint(data,orig,mask,35+270,20)
        commonlen=min(len(c1),len(c2),len(c3),len(c4))
        first=pylab.array([pylab.find(nc1!=0).min(),pylab.find(nc2!=0).min(),pylab.find(nc3!=0).min(),pylab.find(nc4!=0).min()]).max()
        return pylab.sum(pylab.sqrt((c1[first:commonlen]-c3[first:commonlen])**2+(c2[first:commonlen]-c4[first:commonlen])**2))/commonlen
    orig=fmin(targetfunc,pylab.array(orig_initial),args=(data,1-mask),maxiter=maxiter,disp=True)
    return orig
def imageint(data,orig,mask,fi=None,dfi=None):
    """Perform radial averaging on the image.
    
    Inputs:
        data: matrix to average
        orig: vector of beam center coordinates, starting from 1.
        mask: mask matrix; 1 means masked, 0 means non-masked
        fi: starting angle for sector averaging, in degrees
        dfi: angle extent for sector averaging, in degrees
    Outputs:
        vector of integrated values
        vector of effective areas
        
    Note: based on the work of Mika Torkkeli
    """
    Y,X=pylab.meshgrid(pylab.arange(data.shape[1]),pylab.arange(data.shape[0]))
    X=X-orig[0]+1
    Y=Y-orig[1]+1
    D=pylab.sqrt(X**2+Y**2)
    d=D[mask==0]
    dat=data[mask==0]
    if (fi is not None) and (dfi is not None):
        x=X[mask==0]
        y=Y[mask==0]
        phi=pylab.fmod(pylab.arctan2(y,x)-pylab.pi/180.0*fi+10*pylab.pi,2*pylab.pi)
        d=d[phi<=dfi*pylab.pi/180.0]
        dat=dat[phi<=dfi*pylab.pi/180.0]
    C=pylab.zeros(pylab.ceil(d.max()))
    NC=pylab.zeros(pylab.ceil(d.max()))
    for i in range(len(dat)):
        C[pylab.floor(d[i])]+=dat[i]
        NC[pylab.floor(d[i])]+=1
    return C,NC
def sectint(data,fi,orig,mask):
    """Calculate sector-delimited radial average
    
    Inputs:
        data: matrix to average
        fi: a vector of length 2 of starting and ending angles, in degree.
        orig: the origin. Coordinates are counted from 1.
        mask: mask matrix

    Outputs:
        vector of integrated values
        vector of effective areas
        
    Note: based on the work of Mika Torkkeli
    """
    fi=pylab.array(fi)
    return imageint(data,orig,mask,fi=fi.min(),dfi=fi.max()-fi.min())
def radint(data,dataerr,energy,distance,res,bcx,bcy,mask,q=None,shutup=True):
    """Do radial integration on 2D scattering images
    
    Inputs:
        data: the intensity matrix
        dataerr: the error matrix (of the same size as data). Should not be
            zero!
        energy: the (calibrated) beam energy (eV)
        distance: the distance from the sample to the detector (mm)
        res: pixel size in mm-s. Both x and y (row and column) direction can
            be given if wished, in a list with two elements. A scalar value
            means that the pixel size is equal in both directions
        bcx: the coordinate of the beam center in the x (row) direction,
            starting from ZERO
        bcy: the coordinate of the beam center in the y (column) direction,
            starting from ZERO
        mask: the mask matrix (of the same size as data). Nonzero is masked,
            zero is not masked
        q: the q points at which the integration is requested. If None, the
            q-range is automatically guessed from the energy, the distance and
            the mask. It should be defined in 1/Angstroems.
        shutup: if True, work quietly (do not print messages).
        
    Outputs: four ndarrays.
        the q vector
        the intensity vector
        the error vector
        the area vector
    """
    if type(res)!=types.ListType:
        res=[res,res];
    if len(res)==1:
        res=[res[0], res[0]]
    if len(res)>2:
        raise ValueError('res should be a scalar or a nonempty vector of length<=2')
    if data.shape!=dataerr.shape or data.shape!=mask.shape:
        raise ValueError('data, dataerr and mask should be of the same shape')
    M=data.shape[0] # number of rows
    N=data.shape[1] # number of columns
    
    if not shutup:
        print "Creating D matrix...",
    # Creating D matrix which is the distance of the sub-pixels from the origin.
    Y,X=pylab.meshgrid(pylab.arange(data.shape[1]),pylab.arange(data.shape[0]));
    D=pylab.sqrt((res[0]*(X-bcx))**2+
                 (res[1]*(Y-bcy))**2)
    if not shutup:
        print "done"
        print "Calculating q-matrix...",
    # Q-matrix is calculated from the D matrix
    q1=4*pylab.pi*pylab.sin(0.5*pylab.arctan(D/float(distance)))*energy/float(HC)
    if not shutup:
        print "done"
        print "Masking...",
    # eliminating masked pixels
    data=data[mask==0]
    dataerr=dataerr[mask==0]
    q1=q1[mask==0]
    #del datalin
    #del dataerrlin
    #del masklin
    #del qlin
    if not shutup:
        print "done"
    # if the q-scale was not supplied, create one.
    if q is None:
        if not shutup:
            print "Creating q-scale...",
        qmin=min(q1) # the lowest non-masked q-value
        qmax=max(q1) # the highest non-masked q-value
        #qstep=(qmax-qmin)/10
        qstep=(qmax-qmin)/(2*pylab.sqrt(M*M+N*N))
        q=pylab.arange(qmin,qmax,qstep)
        if not shutup:
            print "done"
    else:
        q=pylab.array(q)
    # initialize the output vectors
    Intensity=pylab.zeros(q.size)
    Error=pylab.zeros(q.size)
    Area=pylab.zeros(q.size)
    # square the error
    dataerr=dataerr**2
    if not shutup:
        print "Integrating..."
    # set the bounds of the q-bins in qmin and qmax
    qmin=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmin.insert(0,q[0])
    qmin=pylab.array(qmin)
    qmax=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmax.append(q[-1])
    qmax=pylab.array(qmax)
    # go through every q-bin
    for l in range(len(q)):
        indices=((q1<=qmax[l])&(q1>qmin[l])) # the indices of the pixels which belong to this q-bin
        Intensity[l]=pylab.sum((data[indices])/(dataerr[indices]**2)) # sum the intensities weighted by 1/sigma**2
        Error[l]=1/pylab.sum(1/(dataerr[indices]**2)) # error of the weighted average
        Area[l]=pylab.sum(indices) # collect the area
        # normalization by the area
        Intensity[l]=Intensity[l]*Error[l] # Error[l] is 1/sum_i(1/sigma^2_i)
        Error[l]=pylab.sqrt(Error[l])
    if not shutup:
        print "done"
        
    return q,Intensity,Error,Area # return
    
def calculateDmatrix(mask,res,bcx,bcy):
    """Calculate distances of pixels from the origin
    
    Inputs:
        mask: mask matrix (only its shape is used)
        res: pixel size in mm-s. Can be a vector of length 2 or a scalar
        bcx: Beam center in pixels, in the row direction
        bcy: Beam center in pixels, in the column direction
        
    Output:
        A matrix of the shape of <mask>. Each element contains the distance
        of the centers of the pixels from the origin (bcx,bcy), expressed in
        mm-s.
    """
    if type(res)!=types.ListType:
        res=[res]
    if len(res)<2:
        res=res*2
    Y,X=pylab.meshgrid(pylab.arange(mask.shape[1]),pylab.arange(mask.shape[0]));
    D=pylab.sqrt((res[0]*(X-bcx))**2+
                 (res[1]*(Y-bcy))**2)
    return D
def scalewaxs(fsns,mask2d):
    """Scale waxs curves to saxs files
    
    Inputs:
        fsns: fsn range
        mask2d: mask for the 2d scattering matrices. Zero is masked, nonzero is non-masked.
        
    Outputs:
        waxsscaled%d.dat files are saved
        
    Notes:
        The overlap of the SAXS and WAXS q-ranges is found. After it the SAXS
        curves are re-binned (re-integrated) to those q-bins, and the
        multiplication factor is calculated from the integrals.
    """
    if type(fsns)!=types.ListType:
        fsns=[fsns]
    for fsn in fsns:
        A,Aerr,param=read2dintfile(fsn)
        if len(A)<1:
            continue
        waxsdata=readwaxscor(fsn)
        if len(waxsdata)<1:
            continue
        D=calculateDmatrix(mask2d,param[0]['PixelSize'],param[0]['BeamPosX']-1,
                           param[0]['BeamPosY']-1)
        Dmax=D[mask2d!=0].max()
        qmax=4*pylab.pi*pylab.sin(0.5*pylab.arctan(Dmax/float(param[0]['Dist'])))*param[0]['EnergyCalibrated']/float(HC)
        qmin=min([waxsdata[0]['q'][i] for i in range(len(waxsdata[0]['q'])) 
                  if waxsdata[0]['Intensity'][i]>0])
        qrange=waxsdata[0]['q'][waxsdata[0]['q']<=qmax];
        qrange=qrange[qrange>=qmin];
        if len(qrange)<1:
            print 'No overlap between SAXS and WAXS data in q, for FSN %d: cannot scale WAXS. Skipping.' % fsn
            print 'Details: q_max for SAXS: ',qmax,' q_min for WAXS:',qmin
            continue
        print 'Common q-range consists of %d points.'%len(qrange)
        print 'Re-integrating 2D data for FSN %d'% fsn
        [q,I,E,Area]=radint(A[0],Aerr[0],param[0]['EnergyCalibrated'],param[0]['Dist'],
                         param[0]['PixelSize'],param[0]['BeamPosX']-1,
                         param[0]['BeamPosY']-1,mask2d==0,qrange)
        q=q[Area>0]
        I=I[Area>0]
        E=E[Area>0]
        print 'Common q-range after integration consists of %d points.' %len(q)
        print 'q-points with 0 effective area: %d' % (len(qrange)-len(q))
        if len(q)<1:
            print 'No overlap between SAXS and WAXS in q, after integration of FSN %d.' %fsn
            continue
        waxsindices=[i for i in range(len(waxsdata[0]['q'])) if waxsdata[0]['q'][i] in q]
        Iw=waxsdata[0]['Intensity'][waxsindices]
        Ew=waxsdata[0]['Error'][waxsindices]
        mult,errmult=multfactor(q,I,E,Iw,Ew)
        mult1=param[0]['NormFactor']
        errmult1=param[0]['NormFactorRelativeError']*mult1*0.01
        waxsdata[0]['Error']=pylab.sqrt((waxsdata[0]['Error']*mult)**2+
                                     (errmult*waxsdata[0]['Intensity'])**2)
        waxsdata[0]['Intensity']=waxsdata[0]['Intensity']*mult
        print 'mult: ',mult,'+/-',errmult
#        print 'mult1: ',mult1,'+/-',errmult1
        writeintfile(waxsdata[0]['q'],waxsdata[0]['Intensity'],waxsdata[0]['Error'],param[0],filetype='waxsscaled')
        [q,I,E,Area]=radint(A[0],Aerr[0],param[0]['EnergyCalibrated'],param[0]['Dist'],
                            param[0]['PixelSize'],param[0]['BeamPosX']-1,
                            param[0]['BeamPosY']-1,mask2d==0)
        pylab.figure()
        pylab.subplot(1,1,1)
        pylab.loglog(q,I,label='SAXS')
        pylab.loglog(waxsdata[0]['q'],waxsdata[0]['Intensity'],label='WAXS')
        pylab.legend()
        pylab.title('FSN %d: %s' % (param[0]['FSN'], param[0]['Title']))
        pylab.xlabel(u'q (1/%c)' % 197)
        pylab.ylabel('Scattering cross-section (1/cm)')
        pylab.savefig('scalewaxs%d.eps' % param[0]['FSN'],dpi=300,transparent='True',format='eps')
        pylab.close(pylab.gcf())
#XANES and EXAFS analysis
def smoothabt(muddict,smoothing):
    """Smooth mu*d data with splines
    
    Inputs:
        muddict: mu*d dictionary
        smoothing: smoothing parameter for scipy.interpolate.splrep.
        
    Outputs:
        a mud dictionary with the smoothed data.
    """
    tck=scipy.interpolate.splrep(muddict['Energy'],muddict['Mud'],s=smoothing);
    return {'Energy':muddict['Energy'][:],
            'Mud':scipy.interpolate.splev(muddict['Energy'],tck),
            'Title':("%s_smooth%lf" % (muddict['Title'],smoothing)),
            'scan':muddict['scan']}
def execchooch(mud,element,edge,choochexecutable='/opt/chooch/chooch/bin/chooch',resolution=None):
    """Execute CHOOCH
    
    Inputs:
        mud: mu*d dictionary.
        element: the name of the element, eg. 'Cd'
        edge: the absorption edge to use, eg. 'K' or 'L1'
        choochexecutable: the path where the CHOOCH executable can be found.
        resolution: the resolution of the monochromator, if you want to take
            this into account.
    
    Outputs:
        f1f2 matrix. An exception is raised if running CHOOCH fails.
    """
    writechooch(mud,'choochin.tmp');
    if resolution is None:
        cmd='%s -v 0 -e %s -a %s -o choochout.tmp choochin.tmp' % (choochexecutable, element, edge)
    else:
        cmd='%s -v 0 -e %s -a %s -r %lf -o choochout.tmp choochin.tmp' % (choochexecutable, element, edge, resolution)
    print 'Running CHOOCH with command: ', cmd
    a=os.system(cmd);
    if (a==32512):
        raise IOError( "The chooch executable cannot be found at %s. Please supply another path." % choochexecutable)
    tmp=pylab.loadtxt('choochout.tmp');
    data=pylab.zeros((tmp.shape[0],3))
    data[:,0]=tmp[:,0];
    data[:,1]=tmp[:,2];
    data[:,2]=tmp[:,1];
    return data;
def xanes2f1f2(mud,smoothing,element,edge,title,substitutepoints=[],startpoint=-pylab.inf,endpoint=pylab.inf,postsmoothing=[]):
    """Calculate anomalous correction factors from a XANES scan.
    
    Inputs:
        mud: mud dictionary
        smoothing: smoothing parameter for smoothabt().
        element: the short name of the element, as 'Cd'...
        edge: the absorption edge (eg. 'K', 'L1', ...)
        title: the title for saving files
        substitutepoints: a list of energy values, at which the mud value should
            be substituted by the average of the two neighbours. Use this to get
            rid of outiler points...
        startpoint: lower cutoff energy
        endpoint: upper cutoff energy.
        postsmoothing: list of 3-tuples. Each 3-tuple will be interpreted as
            (lower energy, upper energy, smoothing parameter). Apply this for
            elimination of non-physical oscillations from the curve.
        
    Outputs:
        the calculated anomalous scattering factors (f' and f'')
        files xanes_smoothing_<title>.png and xanes_chooch_<title>.png will be
            saved, as well as f1f2_<title>.dat with the f' and f'' values. The
            external program CHOOCH (by Gwyndaf Evans) is used to convert
            mu*d data to anomalous scattering factors.
    """
    pylab.clf()
    for p in substitutepoints:
        index=pylab.find(pylab.absolute(mud['Energy']-p)<1)
        mud['Mud'][index]=0.5*(mud['Mud'][index-1]+mud['Mud'][index+1])
    
    indices=mud['Energy']<endpoint;
    mud['Energy']=mud['Energy'][indices];
    mud['Mud']=mud['Mud'][indices];
    
    indices=mud['Energy']>startpoint;
    mud['Energy']=mud['Energy'][indices];
    mud['Mud']=mud['Mud'][indices];
    
    if smoothing is None:
        smoothing=testsmoothing(mud['Energy'],mud['Mud'],1e-5)
        print "Using %lf for smoothing parameter" % smoothing
    pylab.clf()
    pylab.plot(mud['Energy'],mud['Mud']);
    B=smoothabt(mud,smoothing)
    pylab.plot(B['Energy'],B['Mud'])
    pylab.legend(['$\mu d$ measured','$\mu d$ smoothed'],loc='best');
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('$\mu d$')
    pylab.title(title)
    pylab.savefig("xanes_smoothing_%s.png" % title,dpi=300,papertype='a4',format='png',transparent=True)
    pylab.clf()
    writechooch(B,'choochin.tmp')
    f1f2=execchooch(B,element,edge)
    for p in postsmoothing:
        indices=(f1f2[:,0]<=p[1]) & (f1f2[:,0]>=p[0])
        x1=f1f2[indices,0]
        y1=f1f2[indices,1]
        z1=f1f2[indices,2]
        s=p[2]
        if p[2] is None:
            s=testsmoothing(x1,y1,1e-1,1e-2,1e1)
        tck=scipy.interpolate.splrep(x1,y1,s=s)
        f1f2[indices,1]=scipy.interpolate.splev(x1,tck)
        tck=scipy.interpolate.splrep(x1,z1,s=s)
        f1f2[indices,2]=scipy.interpolate.splev(x1,tck)
    
    pylab.plot(f1f2[:,0],f1f2[:,1:3]);
    #pylab.legend(['$f^\'$ from CHOOCH','$f^{\'\'}$ from CHOOCH'],loc='best');
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('$f^\'$ and $f^{\'\'}$')
    pylab.title(title)
    pylab.savefig("xanes_chooch_%s.png" % title,dpi=300,papertype='a4',format='png',transparent=True)
    writef1f2(f1f2,("f1f2_%s.dat" % title));
    return f1f2
#data quality tools
def testsmoothing(x,y,smoothing=1e-5,slidermin=1e-6,slidermax=1e-2):
    ax=pylab.axes((0.2,0.85,0.7,0.05));
    sl=matplotlib.widgets.Slider(ax,'',pylab.log10(slidermin),pylab.log10(slidermax),pylab.log10(smoothing));
    fig=pylab.gcf()
    fig.smoothingdone=False
    ax=pylab.axes((0.1,0.85,0.1,0.05));
    def butoff(a=None):
        pylab.gcf().smoothingdone=True
    but=matplotlib.widgets.Button(ax,'Ok')
    but.on_clicked(butoff)
    pylab.axes((0.1,0.1,0.8,0.7))
    pylab.cla()
    pylab.plot(x,y,'.')
    smoothing=pow(10,sl.val);
    tck=scipy.interpolate.splrep(x,y,s=smoothing)
    y1=scipy.interpolate.splev(x,tck)
    pylab.plot(x,y1,linewidth=2)
    def fun(a):
        ax=pylab.axis()
        pylab.cla()
        pylab.plot(x,y,'.')
        smoothing=pow(10,sl.val);
        tck=scipy.interpolate.splrep(x,y,s=smoothing)
        y1=scipy.interpolate.splev(x,tck)
        pylab.plot(x,y1,linewidth=2)
        pylab.axis(ax)
    sl.on_changed(fun)
    fun(1e-5)
    while not fig.smoothingdone:
        pylab.waitforbuttonpress()
    pylab.clf()
    pylab.plot(x,y,'.')
    tck=scipy.interpolate.splrep(x,y,s=pow(10,sl.val))
    y1=scipy.interpolate.splev(x,tck)
    pylab.plot(x,y1,linewidth=2)
    pylab.draw()
    return pow(10,sl.val)

def testorigin(data,orig,mask=None):
    """Shows several test plots by which the validity of the determined origin
    can  be tested.
    
    Inputs:
        data: the 2d scattering image
        orig: the origin [row,column]
        mask: the mask matrix.
    """
    if mask is None:
        mask=pylab.ones(data.shape)
    pylab.subplot(2,2,1)
    plot2dmatrix(data,mask=mask)
    pylab.plot([0,data.shape[1]],[orig[0],orig[0]],color='white')
    pylab.plot([orig[1],orig[1]],[0,data.shape[0]],color='white')
    pylab.gca().axis('tight')
    pylab.subplot(2,2,2)
    c1,nc1=imageint(data,orig,1-mask,35,20)
    c2,nc2=imageint(data,orig,1-mask,35+90,20)
    c3,nc3=imageint(data,orig,1-mask,35+180,20)
    c4,nc4=imageint(data,orig,1-mask,35+270,20)
    pylab.plot(c1,marker='.',color='blue',markersize=3)
    pylab.plot(c3,marker='o',color='blue',markersize=6)
    pylab.plot(c2,marker='.',color='red',markersize=3)
    pylab.plot(c4,marker='o',color='red',markersize=6)
    pylab.subplot(2,2,3)
    maxr=max([len(c1),len(c2),len(c3),len(c4)])
    pdata=polartransform(data,pylab.arange(0,maxr),pylab.linspace(0,4*pylab.pi,600),orig[0],orig[1])
    pmask=polartransform(mask,pylab.arange(0,maxr),pylab.linspace(0,4*pylab.pi,600),orig[0],orig[1])
    plot2dmatrix(pdata,mask=pmask)
def assesstransmission(fsns,titleofsample,mode='Gabriel'):
    """Plot transmission, beam center and Doris current vs. FSNs of the given
    sample.
    
    Inputs:
        fsns: range of file sequence numbers
        titleofsample: the title of the sample which should be investigated
        mode: 'Gabriel' if the measurements were made with the gas-detector, 
            and 'Pilatus300k' if that detector was used.            
    """
    if type(fsns)!=types.ListType:
        fsns=[fsns]
    if mode=='Gabriel':
        header1=readheader('ORG',fsns,'.DAT')
    elif mode=='Pilatus300k':
        header1=readheader('org_',fsns,'.header')
    else:
        print "invalid mode argument. Possible values: 'Gabriel', 'Pilatus300k'"
        return
    params1=readlogfile(fsns)
    header=[]
    for h in header1:
        if h['Title']==titleofsample:
            header.append(h.copy())
    params=[]
    for h in params1:
        if h['Title']==titleofsample:
            params.append(h.copy())
    energies=unique([h['Energy'] for h in header],(lambda a,b:abs(a-b)<2))

    doris=[h['Current1'] for h in header]
    orix=[h['BeamPosX'] for h in params]
    oriy=[h['BeamPosY'] for h in params]
    legend1=[]
    legend2=[]
    legend3=[]
    legend4=[]
    for l in range(len(energies)):
        pylab.subplot(4,1,1)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        fsn=[h['FSN'] for h in header if abs(h['Energy']-energies[l])<2]
        transm1=[h['Transm'] for h in params if abs(h['Energy']-energies[l])<2]
        pylab.plot(fsn,transm1,'-o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Transmission')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend1=legend1+['Energy (not calibrated) = %.1f eV\n Mean T = %.4f, std %.4f' % (energies[l],pylab.mean(transm1),pylab.std(transm1))]
        pylab.subplot(4,1,2)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        orix1=[h['BeamPosX'] for h in params if abs(h['Energy']-energies[l])<2]
        pylab.plot(fsn,orix1,'-o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Position of beam center in X')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend2=legend2+['Energy (not calibrated) = %.1f eV\n Mean x = %.4f, std %.4f' % (energies[l],pylab.mean(orix1),pylab.std(orix1))]
        pylab.subplot(4,1,3)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        oriy1=[h['BeamPosY'] for h in params if abs(h['Energy']-energies[l])<2]
        pylab.plot(fsn,oriy1,'-o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Position of beam center in Y')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend3=legend3+['Energy (not calibrated) = %.1f eV\n Mean y = %.4f, std %.4f' % (energies[l],pylab.mean(oriy1),pylab.std(oriy1))]
        pylab.subplot(4,1,4)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        doris1=[h['Current1'] for h in header if abs(h['Energy']-energies[l])<2]
        pylab.plot(fsn,doris1,'o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Doris current (mA)')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend4=legend4+['Energy (not calibrated) = %.1f eV\n Mean I = %.4f' % (energies[l],pylab.mean(doris1))]
        
    pylab.subplot(4,1,1)
    pylab.legend(legend1,loc=(1.03,0))
    pylab.subplot(4,1,2)
    pylab.legend(legend2,loc=(1.03,0))
    pylab.subplot(4,1,3)
    pylab.legend(legend3,loc=(1.03,0))
    pylab.subplot(4,1,4)
    pylab.legend(legend4,loc=(1.03,0))
    

#ASAXS evaluation and post-processing
def reintegrateB1(fsnrange,mask,qrange=None,samples=None):
    """Re-integrate (re-bin) 2d intensity data
    
    Inputs:
        fsnrange: FSN-s of measurements. Measurement files around only one edge
            should be given.
        mask: mask matrix. Zero means masked, nonzero means non-masked.
        qrange [optional]: If it has more than one elements, then the q-points
            in which the intensity is requested. If a single value, the number
            of q-bins between the minimal and maximal q-value determined auto-
            matically. If not given, a default common q-range is calculated
            taking the sample-detector-distance, the lowest and highest energy
            and the mask into account.
        samples [optional]: a list of strings or a single string containing the
            names of samples to be treated. If omitted, all samples will be
            reintegrated.
    Outputs:
        intbinned*.dat files are saved to the disk.
        
    Note:
        the int2dnorm*.mat files along with the respective intnorm*.log files
        should reside in the current directory
    """
    if qrange is not None:
        if type(qrange)!=types.ListType and type(qrange)!=pylab.ndarray:
            qrange=[qrange]
        qrange=pylab.array(qrange)
        original_qrange=qrange.copy(); # take a copy of it
    else:
        original_qrange=None
    if type(fsnrange)!=types.ListType:
        fsnrange=[fsnrange];
    params=readlogfile(fsnrange);
    if len(params)<1:
        return
    if samples is None:
        samples=unique([p['Title'] for p in params]);
    if type(samples)!=types.ListType:
        samples=[samples]
    for s in samples:
        print 'Reintegrating measurement files for sample %s' % s
        sparams=[p for p in params if p['Title']==s];
        if len(sparams)<1:
            print 'No measurements of %s in the current sequence.' % s
            continue # with the next sample
        dists=unique([p['Dist'] for p in sparams]);
        for d in dists:
            if original_qrange is None:
                qrange=None
            else:
                qrange=original_qrange[:];
            sdparams=[p for p in sparams if p['Dist']==d];
            print 'Evaluating measurements with distance %f' %d
            if qrange is not None:
                if (type(qrange) != types.ListType) and (type(qrange) != pylab.ndarray):
                    qrange=[qrange];
            if (qrange is None) or (len(qrange)<2) :
                print 'Generating common q-range'
                energymin=min([p['EnergyCalibrated'] for p in sdparams])
                energymax=max([p['EnergyCalibrated'] for p in sdparams])
                Y,X=pylab.meshgrid(pylab.arange(mask.shape[1]),pylab.arange(mask.shape[0]));
                D=pylab.sqrt((sdparams[0]['PixelSize']*(X-sdparams[0]['BeamPosX']-1))**2+
                            (sdparams[0]['PixelSize']*(Y-sdparams[0]['BeamPosY']-1))**2)
                Dlin=D[mask!=0]
                qmin=4*pylab.pi*pylab.sin(0.5*pylab.arctan(Dlin.min()/d))*energymax/HC;
                qmax=4*pylab.pi*pylab.sin(0.5*pylab.arctan(Dlin.max()/d))*energymin/HC;
                print 'Auto-determined qmin:',qmin
                print 'Auto-determined qmax:',qmax
                print 'qmin=4pi*sin(0.5*atan(Rmin/L))*energymax/HC'
                print 'qmax=4pi*sin(0.5*atan(Rmax/L))*energymin/HC'
                if qrange is None:
                    NQ=pylab.ceil((Dlin.max()-Dlin.min())/sdparams[0]['PixelSize']*2)
                    print 'Auto-determined number of q-bins:',NQ
                else:
                    NQ=qrange[0];
                    print 'Number of q-bins (as given by the user):',NQ
                qrange=pylab.linspace(qmin,qmax,NQ)
            for p in sdparams:
                print 'Loading 2d intensity for FSN %d' % p['FSN']
                data,dataerr,tmp=read2dintfile(p['FSN']);
                if len(data)<1:
                    continue
                print 'Re-integrating...'
                qs,ints,errs,areas=radint(data[0],dataerr[0],p['EnergyCalibrated'],
                                        p['Dist'],p['PixelSize'],p['BeamPosX']-1,
                                        p['BeamPosY']-1,1-mask,qrange);
                writeintfile(qs,ints,errs,p,areas,filetype='intbinned')
                print 'done.'
                del data
                del dataerr
                del qs
                del ints
                del errs
                del areas
def asaxsbasicfunctions(I,Errors,f1,f2,df1=None,df2=None,element=0):
    """Calculate the basic functions (nonresonant, mixed, resonant)
    
    Inputs:
        I: a matrix of intensity (scattering cross section) data. The columns
            should contain the intensities for each energy
        Errors: a matrix of absolute errors of the intensity data. Of the same
            shape as I.
        f1: vector of the f' values for the corresponding columns of I.
        f2: vector of the f'' values for the corresponding columns of I.
        element: the atomic number of the resonant atom. If zero (default),
            derive the basic functions according to Stuhrmann. If nonzero, the
            partial structure factors of the nonresonant part (N), and the
            resonant part (R) are returned, along with the cross-term S_{NR}.
            
    Outputs:
        N: vector of the nonresonant term
        M: vector of the mixed term
        R: vector of the pure resonant term
    """
    I=pylab.array(I,dtype='float');
    f1=pylab.array(f1,dtype='float');
    f2=pylab.array(f2,dtype='float');
    Nenergies=I.shape[1];
    Ilen=I.shape[0];
    if len(f1) != Nenergies:
        print "length of the f' vector should match the number of rows in I."
        return
    if len(f2) != Nenergies:
        print "length of the f'' vector should match the number of rows in I."
        return
    N=pylab.zeros((Ilen,1));
    M=pylab.zeros((Ilen,1));
    R=pylab.zeros((Ilen,1));
    DN=pylab.zeros((Ilen,1));
    DM=pylab.zeros((Ilen,1));
    DR=pylab.zeros((Ilen,1));

    A=pylab.ones((Nenergies,3));
    A[:,1]=2*(element+f1);
    A[:,2]=(element+f1)**2+f2**2;
    DA=pylab.zeros(A.shape)
    if df1 is not None:
        DA[:,1]=2*df1;
        DA[:,2]=pylab.sqrt(4*(element+f1)**2*df1**2+4*f2**2*df2**2)
    B=pylab.dot(pylab.inv(pylab.dot(A.T,A)),A.T);
    ATA=pylab.dot(A.T,A)
    ATAerr=dot_error(A.T,A,DA.T,DA)
    invATA=pylab.inv(ATA)
    invATAerr=inv_error(ATA,ATAerr)
    Berror=dot_error(invATA,A.T,invATAerr,DA.T)
    print Berror
    print "Condition number of inv(A'*A)*A' is ",pylab.cond(B)
    for j in range(0,Ilen):
        tmp=pylab.dot(B,I[j,:])
        N[j]=tmp[0];
        M[j]=tmp[1];
        R[j]=tmp[2];
        tmpe=dot_error(B,I[j,:],Berror,Errors[j,:])
        DN[j]=tmpe[0];
        DM[j]=tmpe[1];
        DR[j]=tmpe[2];
    return N,M,R,DN,DM,DR
def asaxspureresonant(I1,I2,I3,DI1,DI2,DI3,f11,f12,f13,f21,f22,f23):
    """Calculate the pure resonant as the "difference of differences"
    
    Inputs:
        I1,I2,I3: intensity curves for the three energies
        DI1,DI2,DI3: error data for the intensity curves
        f11,f12,f13: f' values
        f21,f22,f23: f'' values
    
    Outputs:
        sep12: (I1-I2)/(f11-f12)
        dsep12: error of sep12
        sep23: (I2-I3)/(f12-f13)
        dsep23: error of I2-I3
        R: the pure resonant term
        DR: the error of the pure resonant term
    """
    factor=f11-f13+(f22**2-f21**2)/(f12-f11)-(f22**2-f23**2)/(f12-f13)
    DR=pylab.sqrt((DI1*DI1)/(f12-f11)**2+
                  (DI2*DI2)*(1/(f12-f11)**2+1/(f12-f13)**2)+
                  (DI3*DI3)/(f12-f13)**2)/pylab.absolute(factor);
    sep12=(I1-I2)/(f11-f12)
    sep23=(I2-I3)/(f12-f13)
    R=(sep12 -sep23)/factor;
    dsep12=pylab.absolute(pylab.sqrt((DI1*DI1)+(DI2*DI2))/(f11-f12))
    dsep23=pylab.absolute(pylab.sqrt((DI2*DI2)+(DI3*DI3))/(f12-f13))
    return sep12,dsep12,sep23,dsep23,R,DR
def asaxsseqeval(data,param,asaxsenergies,chemshift,fprimefile,samples=None,seqname=None,element=0):
    """Evaluate an ASAXS sequence, derive the basic functions
    
    Inputs:
        data: list of data structures as read by eg. readintnorm
        param: list of parameter structures as read by eg. readintnorm
        asaxsenergies: the UNCALIBRATED (aka. "apparent") energy values for
            the ASAXS evaluation. At least 3 should be supplied.
        chemshift: chemical shift. The difference of the calibrated edge energy
            measured on the sample (E_s) and the theoretical edge energy for an
            isolated atom (E_t). If E_s>E_t then chemshift is positive.
        fprimefile: file name (can include path) for the f' data, as created
            by Hephaestus. The file should have three columns:
            enegy<whitespace>fprime<whitespace>fdoubleprime<newline>.
            Lines beginning with # are ignored.
        samples [optional]: a string or a list of strings of samplenames to be
            treated. If omitted, all samples are evaluated.
        seqname [optional]: if given, the following files will be created:
            seqname_samplename_ie.txt : summarized intensities and errors
            seqname_samplename_basicfun.txt: the asaxs basic functions with
                their errors
            seqname_samplename_separation.txt: I_0, (I_1-I_2)/(f1_1-f1_2),
                (I_2-I_3)/(f1_2-f1_3) and the pure resonant term, with their
                errors
            seqname_f1f2.eps: f' and f'' diagram
            seqname_samplename_basicfun.eps: basic functions displayed
            seqname_samplename_separation.eps: separated curves, I_0 and pure
                resonant displayed
            seqname.log: logging
        element [optional]: if nonzero, this is the atomic number of the
            resonant element. If zero (default), the evaluation is carried out
            according to Stuhrmann. Nonzero yields the PSFs.
    """
    if samples is None:
        samples=unique([param[i]['Title'] for i in range(0,len(data))]);
        print "Found samples: ", samples
    if type(samples)!=types.ListType:
        samples=[samples];
    if seqname is not None:
        logfile=open('%s.log' % seqname,'wt')
        logfile.write('ASAXS sequence name: %s\n' % seqname)
        logfile.write('Time: %s' % time.asctime())
    asaxsecalib=[];
    #asaxsenergies=pylab.array(unique(asaxsenergies,lambda a,b:(abs(a-b)<2)))
    asaxsenergies=pylab.array(asaxsenergies);
    for j in range(0,len(asaxsenergies)):
        asaxsecalib.append([param[i]['EnergyCalibrated']
                             for i in range(0,len(data)) 
                             if abs(param[i]['Energy']-asaxsenergies[j])<2][0]);
    asaxsecalib=pylab.array(asaxsecalib);
    
    print "Calibrated ASAXS energies:", asaxsecalib
    fprimes=readf1f2(fprimefile);
    pylab.plot(fprimes[:,0],fprimes[:,1],'b-');
    pylab.plot(fprimes[:,0],fprimes[:,2],'r-');
    asaxsf1=pylab.interp(asaxsecalib-chemshift,fprimes[:,0],fprimes[:,1]);
    asaxsf2=pylab.interp(asaxsecalib-chemshift,fprimes[:,0],fprimes[:,2]);
    print "f' values", asaxsf1
    print "f'' values", asaxsf2
    if seqname is not None:
        logfile.write('Calibrated ASAXS energies:\n')
        for i in range(len(asaxsenergies)):
            logfile.write("%f -> %f\tf1=%f\tf2=%f\n" % (asaxsenergies[i],asaxsecalib[i],asaxsf1[i],asaxsf2[i]))
        logfile.write('Chemical shift (eV): %f\n' % chemshift)
        logfile.write('Atomic number supplied by the user: %d\n' % element)
        logfile.write('fprime file: %s\n' % fprimefile)
    pylab.plot(asaxsecalib-chemshift,asaxsf1,'b.',markersize=10);
    pylab.plot(asaxsecalib-chemshift,asaxsf2,'r.',markersize=10);
    pylab.legend(['f1','f2'],loc='upper left');
    pylab.xlabel('Photon energy (eV)');
    pylab.ylabel('Anomalous corrections (e.u.)');
    pylab.title('Anomalous correction factors')
    if seqname is not None:
        pylab.savefig('%s_f1f2.eps' % seqname,dpi=300,transparent='True',format='eps')
    if len(asaxsenergies)<3:
        print "At least 3 energies should be given!"
        return
    for s in samples:
        print "Evaluating sample %s" % s
        if seqname is not None:
            logfile.write('Sample: %s\n' % s)
        q=None;
        I=None;
        E=None;
        counter=None;
        fsns=None
        for k in range(0,len(data)): #collect the intensities energy-wise.
            if param[k]['Title']!=s:
                continue
            if q is None:
                q=pylab.array(data[k]['q']);
                NQ=len(q);
                Intensity=pylab.zeros((len(q),len(asaxsenergies)))
                Errors=pylab.zeros((len(q),len(asaxsenergies)))
                counter=pylab.zeros((1,len(asaxsenergies)))
                fsns=[[] for l in range(len(asaxsenergies))]
            if pylab.sum(q-pylab.array(data[k]['q']))>0:
                print "Check the datasets once again: different q-scales!"
                continue;
            energyindex=pylab.absolute(asaxsenergies-param[k]['Energy'])<2
            Intensity[:,energyindex]=Intensity[:,energyindex]+pylab.array(data[k]['Intensity']).reshape(NQ,1);
            Errors[:,energyindex]=Intensity[:,energyindex]+(pylab.array(data[k]['Error']).reshape(NQ,1))**2;
            counter[0,energyindex]=counter[0,energyindex]+1;
            if pylab.find(len(energyindex))>0:
                print pylab.find(energyindex)[0]
                fsns[pylab.find(energyindex)[0]].append(param[k]['FSN']);
        Errors=pylab.sqrt(Errors)
        Intensity=Intensity/pylab.kron(pylab.ones((NQ,1)),counter)
        if seqname is not None:
            for i in range(0,len(asaxsenergies)):
                logfile.write('FSNs for energy #%d:' % i)
                for j in fsns[i]:
                    logfile.write('%d' % j)
                logfile.write('\n')
            datatosave=pylab.zeros((len(q),2*len(asaxsenergies)+1))
            datatosave[:,0]=q;
            for i in range(len(asaxsenergies)):
                datatosave[:,2*i+1]=Intensity[:,i]
                datatosave[:,2*i+2]=Errors[:,i]
            pylab.savetxt('%s_%s_ie.txt' % (seqname, s),datatosave,delimiter='\t')
        # now we have the Intensity and Error matrices fit to feed to asaxsbasicfunctions()
        N,M,R,DN,DM,DR=asaxsbasicfunctions(Intensity,Errors,asaxsf1,asaxsf2,element=element);
        sep12,dsep12,sep23,dsep23,R1,dR1=asaxspureresonant(Intensity[:,0],Intensity[:,1],Intensity[:,2],
                                                           Errors[:,0],Errors[:,1],Errors[:,2],
                                                           asaxsf1[0],asaxsf1[1],asaxsf1[2],
                                                           asaxsf2[0],asaxsf2[1],asaxsf2[2])
        Ireconst=N+M*2*asaxsf1[0]+R*(asaxsf1[0]**2+asaxsf2[0]**2)
        if seqname is not None:
            datatosave=pylab.zeros((len(q),7))
            datatosave[:,0]=q;
            datatosave[:,1]=N.flatten();  datatosave[:,2]=DN.flatten();
            datatosave[:,3]=M.flatten();  datatosave[:,4]=DM.flatten();
            datatosave[:,5]=R.flatten();  datatosave[:,6]=DR.flatten();
            pylab.savetxt('%s_%s_basicfun.txt' % (seqname, s),datatosave,delimiter='\t')
            datatosave[:,1]=sep12.flatten(); datatosave[:,2]=dsep12.flatten();
            datatosave[:,3]=sep23.flatten(); datatosave[:,4]=dsep23.flatten();
            datatosave[:,5]=R1.flatten(); datatosave[:,6]=dR1.flatten();
            pylab.savetxt('%s_%s_separation.txt' % (seqname, s),datatosave,delimiter='\t')
        pylab.figure()
        #pylab.errorbar(q,Intensity[:,0],Errors[:,0],label='I_0',marker='.')
        #pylab.errorbar(q,N.flatten(),DN.flatten(),label='Nonresonant',marker='.')
        #pylab.errorbar(q,M.flatten(),DM.flatten(),label='Mixed',marker='.')
        #pylab.errorbar(q,R.flatten(),DR.flatten(),label='Resonant',marker='o')
        pylab.plot(q,Intensity[:,0],label='I_0',marker='.')
        pylab.plot(q,N.flatten(),label='Nonresonant',marker='.')
        pylab.plot(q,M.flatten(),label='Mixed',marker='.')
        pylab.plot(q,R.flatten(),label='Resonant',marker='o')
        pylab.plot(q,Ireconst.flatten(),label='I_0_reconstructed',marker='.')
        pylab.title("ASAXS basic functions for sample %s" % s)
        pylab.xlabel(u"q (1/%c)" % 197)
        pylab.ylabel("Scattering cross-section (1/cm)")
        pylab.gca().set_xscale('log');
        pylab.gca().set_yscale('log');
        pylab.legend();
        pylab.savefig('%s_%s_basicfun.eps'%(seqname,s),dpi=300,format='eps',transparent=True)
        pylab.figure()
        #pylab.errorbar(q,Intensity[:,0],Errors[:,0],label='I_0',marker='.')
        #pylab.errorbar(q,sep12,dsep12,label='(I_0-I_1)/(f1_0-f1_1)',marker='.')
        #pylab.errorbar(q,sep23,dsep23,label='(I_1-I_2)/(f1_1-f1_2)',marker='.')
        #pylab.errorbar(q,R1.flatten(),dR1.flatten(),label='Pure resonant',marker='.')
        pylab.plot(q,Intensity[:,0],label='I_0',marker='.')
        pylab.plot(q,sep12,label='(I_0-I_1)/(f1_0-f1_1)',marker='.')
        pylab.plot(q,sep23,label='(I_1-I_2)/(f1_1-f1_2)',marker='.')
        pylab.plot(q,R1.flatten(),label='Pure resonant',marker='.')
        
        pylab.title("ASAXS separated and pure resonant terms for sample %s" % s)
        pylab.xlabel(u"q (1/%c)" % 197)
        pylab.ylabel("Scattering cross-section (1/cm)")
        pylab.gca().set_xscale('log');
        pylab.gca().set_yscale('log');
        pylab.legend();
        pylab.savefig('%s_%s_separation.eps'%(seqname,s),dpi=300,format='eps',transparent=True)
    logfile.close()
    pylab.show()
def sumfsns(fsns,samples=None,filetype='intnorm',waxsfiletype='waxsscaled'):
    """Summarize scattering data.
    
    Inputs:
        fsns: FSN range
        samples: samples to evaluate. Leave it None to auto-determine
        filetype: 1D SAXS filetypes (ie. the beginning of the file) to summarize. 
        waxsfiletype: WAXS filetypes (ie. the beginning of the file) to summarize.
    """
    if type(fsns)!=types.ListType:
        fsns=[fsns]
    params=readlogfile(fsns)
    if samples is None:
        samples=unique([p['Title'] for p in params])
    if type(samples)!=types.ListType:
        samples=[samples]
    for s in samples:
        print 'Summing measurements for sample %s' % s
        sparams=[p for p in params if p['Title']==s]
        energies=unique([p['Energy'] for p in sparams],lambda a,b:abs(a-b)<2)
        for e in energies:
            print 'Processing energy %f for sample %s' % (e,s)
            esparams=[p for p in sparams if abs(p['Energy']-e)<2]
            dists=unique([p['Dist'] for p in esparams])
            for d in dists:
                print 'Processing distance %f for energy %f for sample %s'% (d,e,s)
                edsparams=[p for p in esparams if p['Dist']==d]
                counter=0
                q=None
                w=None
                Isum=None
                Esum=None
                for p in edsparams:
                    filename='%s%d.dat' % (filetype,p['FSN'])
                    intdata=readintfile(filename)
                    if len(intdata)<1:
                        continue
                    if counter==0:
                        q=intdata['q']
                        w=1/(intdata['Error']**2)
                        Isum=intdata['Intensity']/(intdata['Error']**2)
                    else:
                        if q.size!=intdata['q'].size:
                            print 'q-range of file %s differs from the others read before. Skipping.' % filename
                            continue
                        if pylab.sum(q-intdata['q'])!=0:
                            print 'q-range of file %s differs from the others read before. Skipping.' % filename
                            continue
                        Isum=Isum+intdata['Intensity']/(intdata['Error']**2)
                        w=w+1/(intdata['Error']**2)
                    counter=counter+1
                if counter>0:
                    Esum=1/w
                    Isum=Isum/w
                    writeintfile(q,Isum,Esum,edsparams[0],filetype='summed')
                else:
                    print 'No files were found for summing.'
            waxscounter=0
            qwaxs=None
            Iwaxs=None
            wwaxs=None
            print 'Processing waxs files for energy %f for sample %s' % (e,s)
            for p in esparams:
                waxsfilename='%s%d.dat' % (waxsfiletype,p['FSN'])
                waxsdata=readintfile(waxsfilename)
                if len(waxsdata)<1:
                    continue
                if waxscounter==0:
                    qwaxs=waxsdata['q']
                    Iwaxs=waxsdata['Intensity']/(waxsdata['Error']**2)
                    wwaxs=1/(waxsdata['Error']**2)
                else:
                    if qwaxs.size!=waxsdata['q'].size:
                        print 'q-range of file %s differs from the others read before. Skipping.' % waxsfilename
                        continue
                    if pylab.sum(qwaxs-waxsdata['q'])!=0:
                        print 'q-range of file %s differs from the others read before. Skipping.' % waxsfilename
                        continue
                    Iwaxs=Iwaxs+waxsdata['Intensity']/(waxsdata['Error']**2)
                    wwaxs=wwaxs+1/(waxsdata['Error']**2)
                waxscounter=waxscounter+1
            if waxscounter>0:
                Ewaxs=1/wwaxs
                Iwaxs=Iwaxs/wwaxs
                writeintfile(qwaxs,Iwaxs,Ewaxs,esparams[0],filetype='waxssummed')
            else:
                print 'No waxs file was found'
#GUI utilities
def makemask(mask,A,savefile=None):
    """Make mask matrix.
    
    Inputs:
        mask: preliminary mask matrix. Give None to create a fresh one
        A: background image. The size of mask and this should be equal.
        savefile [optional]: a file name to save the mask to.
    Output:
        the mask matrix.
    """
    def clickevent(event):
        fig=pylab.gcf()
        if (fig.canvas.manager.toolbar.mode!='') and (fig.mydata['backuptitle'] is None):
            fig.mydata['backuptitle']=fig.mydata['ax'].get_title()
            fig.mydata['ax'].set_title('%s mode is on. Turn it off to continue editing.' % fig.canvas.manager.toolbar.mode)
            return
        if (fig.canvas.manager.toolbar.mode=='') and (fig.mydata['backuptitle'] is not None):
            fig.mydata['ax'].set_title(fig.mydata['backuptitle'])
            fig.mydata['backuptitle']=None
        if event.inaxes==fig.mydata['ax']:
            if fig.mydata['mode']=='RECT0':
                fig.mydata['selectdata']=[event.xdata,event.ydata]
                fig.mydata['mode']='RECT1'
                return
            elif fig.mydata['mode']=='RECT1':
                x0=min(event.xdata,fig.mydata['selectdata'][0])
                y0=min(event.ydata,fig.mydata['selectdata'][1])
                x1=max(event.xdata,fig.mydata['selectdata'][0])
                y1=max(event.ydata,fig.mydata['selectdata'][1])
                Col,Row=pylab.meshgrid(pylab.arange(fig.mydata['mask'].shape[1]),
                                       pylab.arange(fig.mydata['mask'].shape[0]))
                fig.mydata['selection']=(Col<=x1) & (Col>=x0) & (Row<=y1) & (Row>=y0)
                fig.mydata['ax'].set_title('Mask/unmask region with the appropriate button!')
                fig.mydata['selectdata']=[]
                fig.mydata['mode']=None
                a=fig.mydata['ax'].axis()
                fig.mydata['ax'].plot([x0,x0],[y0,y1],color='white')
                fig.mydata['ax'].plot([x0,x1],[y1,y1],color='white')
                fig.mydata['ax'].plot([x1,x1],[y1,y0],color='white')
                fig.mydata['ax'].plot([x1,x0],[y0,y0],color='white')
                fig.mydata['ax'].axis(a)
                return
            elif fig.mydata['mode']=='CIRC0':
                fig.mydata['selectdata']=[event.xdata,event.ydata]
                fig.mydata['mode']='CIRC1'
                fig.mydata['ax'].set_title('Select a boundary point for the circle!')
                return
            elif fig.mydata['mode']=='CIRC1':
                x0=fig.mydata['selectdata'][0]
                y0=fig.mydata['selectdata'][1]
                fig.mydata['selectdata']=[];
                R=pylab.sqrt((x0-event.xdata)**2+(y0-event.ydata)**2)
                Col,Row=pylab.meshgrid(pylab.arange(fig.mydata['mask'].shape[1])-x0,
                                       pylab.arange(fig.mydata['mask'].shape[0])-y0)
                fig.mydata['selection']=pylab.sqrt(Col**2+Row**2)<=R
                fig.mydata['ax'].set_title('Mask/unmask region with the appropriate button!')
                a=fig.mydata['ax'].axis()
                fig.mydata['ax'].plot(x0+R*pylab.cos(pylab.linspace(0,2*pylab.pi,2000)),
                                      y0+R*pylab.sin(pylab.linspace(0,2*pylab.pi,2000)),
                                      color='white')
                fig.mydata['ax'].axis(a)
                fig.mydata['mode']=None
            elif fig.mydata['mode']=='POLY0':
                fig.mydata['selectdata']=[[event.xdata,event.ydata]]
                fig.mydata['mode']='POLY1'
                return
            elif fig.mydata['mode']=='POLY1':
                if event.button==3:
                    fig.mydata['selectdata'].append(fig.mydata['selectdata'][0])
                else:
                    fig.mydata['selectdata'].append([event.xdata,event.ydata])
                p1=fig.mydata['selectdata'][-2]
                p2=fig.mydata['selectdata'][-1]
                a=fig.mydata['ax'].axis()
                fig.mydata['ax'].plot([p1[0],p2[0]],[p1[1],p2[1]],color='white')
                fig.mydata['ax'].axis(a)
                if event.button==3:
                    Col,Row=pylab.meshgrid(pylab.arange(fig.mydata['mask'].shape[1]),
                                           pylab.arange(fig.mydata['mask'].shape[0]))
                    Points=pylab.zeros((Col.size,2))
                    Points[:,0]=Col.flatten()
                    Points[:,1]=Row.flatten()
                    fig.mydata['selection']=pylab.zeros(Col.shape).astype('bool')
                    ptsin=matplotlib.nxutils.points_inside_poly(Points,fig.mydata['selectdata'])
                    fig.mydata['selection'][ptsin.reshape(Col.shape)]=True
                    fig.mydata['selectdata']=[]
                    fig.mydata['mode']=None
            elif fig.mydata['mode']=='PHUNT':
                fig.mydata['mask'][pylab.floor(event.ydata+.5),pylab.floor(event.xdata+.5)]=not(fig.mydata['mask'][pylab.floor(event.ydata+.5),pylab.floor(event.xdata+.5)])
                fig.mydata['redrawneeded']=True
                return
        elif event.inaxes==fig.mydata['bax9']: # pixel hunting
            if fig.mydata['mode']!='PHUNT':
                fig.mydata['ax'].set_title('Mask/unmask pixels by clicking them!')
                fig.mydata['bhunt'].label.set_text('End pixel hunting')
                fig.mydata['mode']='PHUNT'
            else:
                fig.mydata['ax'].set_title('')
                fig.mydata['bhunt'].label.set_text('Pixel hunt')
                fig.mydata['mode']=None
                return
        elif event.inaxes==fig.mydata['bax8']: # select rectangle
            fig.mydata['ax'].set_title('Select rectangle with its two opposite corners!')
            fig.mydata['mode']='RECT0'
            return
        elif event.inaxes==fig.mydata['bax7']: # select circle
            fig.mydata['ax'].set_title('Select the center of the circle!')
            fig.mydata['mode']='CIRC0'
            return
        elif event.inaxes==fig.mydata['bax6']: # select polygon
            fig.mydata['ax'].set_title('Select the corners of the polygon!\nRight button to finish')
            fig.mydata['mode']='POLY0'
            return
        elif event.inaxes==fig.mydata['bax5']: # remove selection
            fig.mydata['selection']=None
            fig.mydata['redrawneeded']=True
            fig.mydata['ax'].set_title('')
            fig.mydata['mode']=None
            return
        elif event.inaxes==fig.mydata['bax4']: # mask it
            if fig.mydata['selection'] is not None:
                fig.mydata['mask'][fig.mydata['selection']]=0
                fig.mydata['redrawneeded']=True
                fig.mydata['selection']=None
                return
            else:
                fig.mydata['ax'].set_title('Please select something first!')
                return
        elif event.inaxes==fig.mydata['bax3']: # unmask it
            if fig.mydata['selection'] is not None:
                fig.mydata['mask'][fig.mydata['selection']]=1
                fig.mydata['redrawneeded']=True
                fig.mydata['selection']=None
                return
            else:
                fig.mydata['ax'].set_title('Please select something first!')
                return
        elif event.inaxes==fig.mydata['bax2']: # flip mask on selection
            if fig.mydata['selection'] is not None:
                fig.mydata['mask'][fig.mydata['selection']]=fig.mydata['mask'][fig.mydata['selection']] ^ True
                fig.mydata['redrawneeded']=True
                fig.mydata['selection']=None
                return
            else:
                fig.mydata['ax'].set_title('Please select something first!')
                return
        elif event.inaxes==fig.mydata['bax1']: # flip mask
            fig.mydata['mask']=fig.mydata['mask'] ^ True
            fig.mydata['redrawneeded']=True
            return
        elif event.inaxes==fig.mydata['bax0']: # done
            pylab.gcf().toexit=True
    if mask is None:
        mask=pylab.ones(A.shape)
    if A.shape!=mask.shape:
        print 'The shapes of A and mask should be equal.'
        return None
    fig=pylab.figure();
    fig.mydata={}
    fig.mydata['ax']=fig.add_axes((0.3,0.1,0.6,0.8))
    for i in range(10):
        fig.mydata['bax%d' % i]=fig.add_axes((0.05,0.07*i+0.1,0.2,0.05))
    fig.mydata['bhunt']=matplotlib.widgets.Button(fig.mydata['bax9'],'Pixel hunt')
    fig.mydata['brect']=matplotlib.widgets.Button(fig.mydata['bax8'],'Rectangle')
    fig.mydata['bcirc']=matplotlib.widgets.Button(fig.mydata['bax7'],'Circle')
    fig.mydata['bpoly']=matplotlib.widgets.Button(fig.mydata['bax6'],'Polygon')
    fig.mydata['bpoint']=matplotlib.widgets.Button(fig.mydata['bax5'],'Clear selection')
    fig.mydata['bmaskit']=matplotlib.widgets.Button(fig.mydata['bax4'],'Mask selection')
    fig.mydata['bunmaskit']=matplotlib.widgets.Button(fig.mydata['bax3'],'Unmask selection')
    fig.mydata['bflipselection']=matplotlib.widgets.Button(fig.mydata['bax2'],'Flipmask selection')
    fig.mydata['bflipmask']=matplotlib.widgets.Button(fig.mydata['bax1'],'Flip mask')
    fig.mydata['breturn']=matplotlib.widgets.Button(fig.mydata['bax0'],'Done')
    fig.mydata['selection']=None
    fig.mydata['clickdata']=None
    fig.mydata['backuptitle']=None
    fig.mydata['mode']=None
    fig.mydata['mask']=mask.astype('bool')
    fig.mydata['redrawneeded']=True
    conn_id=fig.canvas.mpl_connect('button_press_event',clickevent)
    fig.toexit=False
    fig.show()
    firstdraw=1;
    while fig.toexit==False:
        if fig.mydata['redrawneeded']:
            if not firstdraw:
                ax=fig.mydata['ax'].axis();
            fig.mydata['redrawneeded']=False
            fig.mydata['ax'].cla()
            pylab.axes(fig.mydata['ax'])
            plot2dmatrix(A,mask=fig.mydata['mask'])
            fig.mydata['ax'].set_title('')
            if not firstdraw:
                fig.mydata['ax'].axis(ax);
            firstdraw=0;
        pylab.draw()
        fig.waitforbuttonpress()
    #ax.imshow(maskplot)
    #pylab.show()
    mask=fig.mydata['mask']
    pylab.close(fig)
    if savefile is not None:
        print 'Saving file'
        scipy.io.savemat(savefile,{'mask':mask})
    return mask
    

#display routines
def plotints(data,param,samplename,energies,marker='.',mult=1,gui=False):
    """Plot intensities
    
    Inputs:
        data: list of scattering data. Each element of this list should be
            a dictionary, with the fields 'q','Intensity' and 'Error' present.
        param: a list of header data. Each element should be a dictionary.
        samplename: the name of the sample which should be plotted. Also a list
            can be supplied if multiple samples are to be plotted.
        energies: one or more energy values in a list. This decides which 
            energies should be plotted
        marker [optional] : the marker symbol of the plot. Possible values are '.', 'o',
            'x'... If plotting of multiple samples is requested
            (parameter <samplenames> is a list) then this can also be a list,
            but of the same size as samplenames. Default value is '.'.
        mult [optional]: multiplicate the intensity by this number when plotting. The same
            applies as to symboll. Default value is 1.
        gui [optional]: display graphical user interface to show/hide plotted
            lines independently. Default value is False (no gui)
    """
    if type(energies)!=types.ListType:
        energies=[energies];
    colors=['blue','green','red','black','magenta'];
    if type(samplename)==types.StringType:
        samplename=[samplename]
    if type(marker)!=types.ListType:
        marker=[marker]
    if type(mult)!=types.ListType:
        mult=[mult]
    if len(marker)==1:
        marker=marker*len(samplename)
    if len(mult)==1:
        mult=mult*len(samplename)
    if (len(marker)!=len(samplename)) or (len(mult) !=len(samplename)):
        raise ValueError
    if gui==True:
        fig=pylab.figure()
        buttonax=fig.add_axes((0.65,0.1,0.3,0.05))
        guiax=fig.add_axes((0.65,0.15,0.3,0.75))
        ax=fig.add_axes((0.1,0.1,0.5,0.8))
        btn=matplotlib.widgets.Button(buttonax,'Close GUI')
        def fun(event):
            fig=pylab.gcf()
            fig.delaxes(fig.axes[0])
            fig.delaxes(fig.axes[0])
            fig.axes[0].set_position((0.1,0.1,0.8,0.85))
        btn.on_clicked(fun)
        guiax.set_title('Visibility selector')
        texts=[]
        handles=[]
    else:
        fig=pylab.gcf()
        ax=pylab.gca()
    for k in range(len(data)):
        for s in range(len(samplename)):
            if param[k]['Title']==samplename[s]:
                for e in range(min(len(colors),len(energies))):
                    if abs(param[k]['Energy']-energies[e])<2:
                        print 'plotints', e, param[k]['FSN'], param[k]['Title'],k
                        h=ax.loglog(data[k]['q'],
                                       data[k]['Intensity']*mult[s],
                                       marker=marker[s],
                                       color=colors[e])
                        #h=ax.semilogy(data[k]['q'],
                        #                data[k]['Intensity']*mult[s],
                        #                marker=symboll[s],
                        #                color=colors[e])
                        #h=ax.plot(data[k]['q'],
                        #                 data[k]['Intensity']*mult[s],
                        #                 marker=symboll[s],
                        #                 color=colors[e])
                        if gui==True:
                            texts.append('%d(%s) @%.2f eV' % (param[k]['FSN'], param[k]['Title'], param[k]['Energy']))
                            handles.append(h[0])
    if gui==True:
        actives=[1 for x in range(len(handles))]
        cbs=matplotlib.widgets.CheckButtons(guiax,texts,actives)
        def onclicked(name,h=handles,t=texts,cb=cbs):
            index=[i for i in range(len(h)) if t[i]==name]
            if len(index)<1:
                return
            index=index[0]
            h[index].set_visible(cb.lines[index][0].get_visible())
        cbs.on_clicked(onclicked)
    ax.set_xlabel(ur'q (%c$^{-1}$)' % 197)
    ax.set_ylabel(r'$\frac{d\sigma}{d\Omega}$ (cm$^{-1}$)')
    fig.show()
    if gui==True:
        while len(fig.axes)==3:
            fig.waitforbuttonpress()
            pylab.draw()
def plot2dmatrix(A,maxval=None,mask=None,header=None,qs=[],showqscale=True):
    """Plots the matrix A in log-log plot
    
    Inputs:
        A: the matrix
        maxval: if not None, then before taking log(A), the elements of A,
            which are larger than this are replaced by the largest element of
            A below maxval.
        mask: a mask matrix to overlay the scattering pattern with it. Pixels
            where the mask is 0 will be faded.
        header: the header or param structure. If it is supplied, the x and y
            axes will display the q-range
        qs: q-values for which concentric circles will be drawn. To use this
            option, header should be given.
        showqscale: show q-scale on both the horizontal and vertical axes
    """
    tmp=A.copy(); # this is needed as Python uses the pass-by-object method,
                  # so A is the SAME as the version of the caller. tmp=A would
                  # render tmp the SAME (physically) as A. If we only would
                  # call pylab.log(tmp), it won't be an error, as pylab.log()
                  # does not tamper with the content of its argument, but
                  # returns a new matrix. However, when we do the magic with
                  # maxval, it would be a problem, as elements of the original
                  # matrix were modified.
    if maxval is not None:
        tmp[tmp>maxval]=max(tmp[tmp<=maxval])
    else:
        print 'maxval is None'
    tmp[tmp<=0]=tmp[tmp>0].min()
    tmp=pylab.log(tmp);
    tmp[pylab.isnan(tmp)]=0;
    if (header is not None) and (showqscale):
        xmin=0-(header['BeamPosX']-1)*header['PixelSize']
        xmax=(tmp.shape[0]-(header['BeamPosX']-1))*header['PixelSize']
        ymin=0-(header['BeamPosY']-1)*header['PixelSize']
        ymax=(tmp.shape[1]-(header['BeamPosY']-1))*header['PixelSize']
        qxmin=4*pylab.pi*pylab.sin(0.5*pylab.arctan(xmin/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qxmax=4*pylab.pi*pylab.sin(0.5*pylab.arctan(xmax/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qymin=4*pylab.pi*pylab.sin(0.5*pylab.arctan(ymin/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qymax=4*pylab.pi*pylab.sin(0.5*pylab.arctan(ymax/header['Dist']))*header['EnergyCalibrated']/float(HC)
        extent=[qymin,qymax,qxmin,qxmax]
    else:
        extent=None
    pylab.imshow(tmp,extent=extent,interpolation='nearest');
    if mask is not None:
        white=pylab.ones((mask.shape[0],mask.shape[1],4))
        white[:,:,3]=pylab.array(1-mask).astype('float')*0.7
        pylab.imshow(white,extent=extent,interpolation='nearest')
    for q in qs:
        a=pylab.gca().axis()
        pylab.plot(q*pylab.cos(pylab.linspace(0,2*pylab.pi,2000)),
                   q*pylab.sin(pylab.linspace(0,2*pylab.pi,2000)),
                   color='white',linewidth=3)
        pylab.gca().axis(a)
            
#Miscellaneous routines
def derivative(x,y=None):
    """Approximate the derivative by finite difference
    
    Inputs:
        x: x data
        y: y data. If None, x is differentiated.
        
    Outputs:
        x1, dx/dy or dx
    """
    x=pylab.array(x);
    if y is None:
        return x[1:]-x[:-1]
    else:
        y=pylab.array(y)
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
    return 2.0/(x**(n+1.0)*scipy.special.gamma((n+1.0)/2.0))*(x**n)**pylab.exp(-x**2/r0**2);
def errtrapz(x,yerr):
    """Error of the trapezoid formula
    Inputs:
        x: the abscissa
        yerr: the error of the dependent variable
        
    Outputs:
        the error of the integral
    """
    x=pylab.array(x);
    yerr=pylab.array(yerr);
    return 0.5*pylab.sqrt((x[1]-x[0])**2*yerr[0]**2+pylab.sum((x[2:]-x[:-2])**2*yerr[1:-1]**2)+
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
    S1=pylab.trapz(I1,q)
    eS1=errtrapz(q,E1)
    S2=pylab.trapz(I2,q)
    eS2=errtrapz(q,E2)
    mult=S1/S2
    errmult=pylab.sqrt((eS1/S1)**2+(eS2/S2)**2)*mult
    return mult,errmult
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
    xdata=pylab.array(xdata,dtype='float');
    ydata=pylab.array(ydata,dtype='float');
    if xdata.size != ydata.size:
        print "The sizes of xdata and ydata should be the same."
        return
    if errdata is not None:
        if ydata.size !=errdata.size:
            print "The sizes of ydata and errdata should be the same."
            return
        errdata=pylab.array(errdata,dtype='float');
        S=pylab.sum(1.0/(errdata**2))
        Sx=pylab.sum(xdata/(errdata**2))
        Sy=pylab.sum(ydata/(errdata**2))
        Sxx=pylab.sum(xdata*xdata/(errdata**2))
        Sxy=pylab.sum(xdata*ydata/(errdata**2))
    else:
        S=float(xdata.size)
        Sx=pylab.sum(xdata)
        Sy=pylab.sum(ydata)
        Sxx=pylab.sum(xdata*xdata)
        Sxy=pylab.sum(xdata*ydata)
    Delta=S*Sxx-Sx*Sx;
    a=(S*Sxy-Sx*Sy)/Delta;
    b=(Sxx*Sy-Sx*Sxy)/Delta;
    aerr=pylab.sqrt(S/Delta);
    berr=pylab.sqrt(Sxx/Delta);
    return a,b,aerr,berr
def dot_error(A,B,DA,DB):
    """Calculate the error of pylab.dot(A,B) according to squared error
    propagation.
    
    Inputs:
        A,B: The matrices
        DA,DB: The absolute error matrices corresponding to A and B, respectively
        
    Output:
        The error matrix
    """
    return pylab.sqrt(pylab.dot(DA**2,B**2)+pylab.dot(A**2,DB**2));
def inv_error(A,DA):
    """Calculate the error of pylab.inv(A) according to squared error
    propagation.
    
    Inputs:
        A: The matrix (square shaped)
        DA: The error of the matrix (same size as A)
    
    Output:
        The error of the inverse matrix
    """
    B=pylab.inv(A);
    return pylab.sqrt(pylab.dot(pylab.dot(B**2,DA**2),B**2))
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
    return G*pylab.exp(-q**2*Rg**2/3.0)+B*pow(pow(scipy.special.erf(q*Rg/pylab.sqrt(6)),3)/q,P)
#IO routines
def readheader(filename,fsn=None,fileend=None):
    """Reads header data from measurement files
    
    Inputs:
        filename: the beginning of the filename, or the whole filename
        fsn: the file sequence number or None if the whole filenam was supplied
            in filename. It can be a list as well.
        fileend: the end of the file. If it ends with .gz, then the file is
            treated as a gzip archive.
    
    Output:
        A list of header dictionaries. An empty list if no headers were read.
        
    Examples:
        read header data from 'ORG000123.DAT':
        
        header=readheader('ORG',123,'.DAT')
        
        or
        
        header=readheader('ORG00123.DAT')
    """
    if fsn is None:
        names=[filename]
    else:
        if type(fsn)==types.ListType:
            names=['%s%05d%s' % (filename,x,fileend ) for x in fsn]
        else:
            names=['%s%05d%s' % (filename,fsn,fileend)]
    headers=[]
    for name in names:
        try:
            header={};
            if name.upper()[-3:]=='.GZ':
                fid=gzip.GzipFile(name,'rt');
            else:
                fid=open(name,'rt');
            lines=fid.readlines()
            fid.close()
            header['FSN']=int(string.strip(lines[0]))
            header['Hour']=int(string.strip(lines[17]))
            header['Minutes']=int(string.strip(lines[18]))
            header['Month']=int(string.strip(lines[19]))
            header['Day']=int(string.strip(lines[20]))
            header['Year']=int(string.strip(lines[21]))+2000
            header['FSNref1']=int(string.strip(lines[23]))
            header['FSNdc']=int(string.strip(lines[24]))
            header['FSNsensitivity']=int(string.strip(lines[25]))
            header['FSNempty']=int(string.strip(lines[26]))
            header['FSNref2']=int(string.strip(lines[27]))
            header['Monitor']=float(string.strip(lines[31]))
            header['Anode']=float(string.strip(lines[32]))
            header['MeasTime']=float(string.strip(lines[33]))
            header['Temperature']=float(string.strip(lines[34]))
            header['Transm']=float(string.strip(lines[41]))
            header['Energy']=HC/float(string.strip(lines[43]))
            header['Dist']=float(string.strip(lines[46]))
            header['XPixel']=1/float(string.strip(lines[49]))
            header['YPixel']=1/float(string.strip(lines[50]))
            header['Title']=string.strip(lines[53])
            header['Title']=string.replace(header['Title'],' ','_')
            header['Title']=string.replace(header['Title'],'-','_')
            header['MonitorDORIS']=float(string.strip(lines[56]))
            header['Owner']=string.strip(lines[57])
            header['Rot1']=float(string.strip(lines[59]))
            header['Rot2']=float(string.strip(lines[60]))
            header['PosSample']=float(string.strip(lines[61]))
            header['DetPosX']=float(string.strip(lines[62]))
            header['DetPosY']=float(string.strip(lines[63]))
            header['MonitorPIEZO']=float(string.strip(lines[64]))
            header['BeamsizeX']=float(string.strip(lines[66]))
            header['BeamsizeY']=float(string.strip(lines[67]))
            header['PosRef']=float(string.strip(lines[70]))
            header['Monochromator1Rot']=float(string.strip(lines[77]))
            header['Monochromator2Rot']=float(string.strip(lines[78]))
            header['Heidenhain1']=float(string.strip(lines[79]))
            header['Heidenhain2']=float(string.strip(lines[80]))
            header['Current1']=float(string.strip(lines[81]))
            header['Current2']=float(string.strip(lines[82]))
            header['Detector']='Unknown'
            header['PixelSize']=(header['XPixel']+header['YPixel'])/2.0
            del lines
            headers.append(header)
        except IOError:
            print 'Cannot find file %s. Make sure the path is correct.' % name
    return headers
def read2dB1data(filename,files=None,fileend=None):
    """Read 2D measurement files, along with their header data

    Inputs:
        filename: the beginning of the filename, or the whole filename
        fsn: the file sequence number or None if the whole filenam was supplied
            in filename. It is possible to give a list of fsns here.
        fileend: the end of the file.
        
    Outputs:
        A list of 2d scattering data matrices
        A list of header data
        
    Examples:
        Read FSN 123-130:
        a) measurements with the Gabriel detector:
        data,header=read2dB1data('ORG',range(123,131),'.DAT')
        b) measurements with the Pilatus300k detector:
        #in this case the org_*.header files should be present in the same folder
        data,header=read2dB1data('org_',range(123,131),'.tif')
    """
    def readgabrieldata(filename):
        try:
            if filename.upper()[-3:]=='.GZ':
                fid=gzip.GzipFile(filename,'rt')
            else:
                fid=open(filename,'rt')
            lines=fid.readlines()
            fid.close()
            nx=int(string.strip(lines[10]))
            ny=int(string.strip(lines[11]))
            data=pylab.zeros((nx,ny),order='F')
            row=0;
            col=0;
            def incrowcol(row,col):
                if row<nx-1:
                    row=row+1;
                else:
                    row=0;
                    col=col+1;
                return row,col
            for line in lines[133:]:
                for i in line.split():
                    data[row,col]=float(i);
                    row,col=incrowcol(row,col)
            return data
        except IOError:
            print 'Cannot find file %s. Make sure the path is correct.' % filename
            return None
    def readpilatus300kdata(filename):
        try:
            fid=open(filename,'rb');
            datastr=fid.read();
            fid.close();
            data=pylab.fromstring(datastr[4096:],'uint32').reshape((619,487)).astype('double')
            return data;
        except IOError:
            print 'Cannot find file %s. Make sure the path is correct.' % filename
            return None
        
    if fileend is None:
        fileend=filename[string.rfind(filename,'.'):]
    if (files is not None) and (type(files)!=types.ListType):
        files=[files];
    if fileend.upper()=='.TIF' or fileend.upper()=='.TIFF': # pilatus300k mode
        filebegin=filename[:string.rfind(filename,'.')]
        if files is None:
            header=readheader(filebegin+'.header')
            data=readpilatus300kdata(filename)
            if (len(header)<1) or (data is None):
                return [],[]
            else:
                header=header[0]
                header['Detector']='Pilatus300k'
                return [data],[header]
        else:
            header=[];
            data=[];
            for fsn in files:
                tmp1=readheader('%s%05d%s' %(filename,fsn,'.header'))
                tmp2=readpilatus300kdata('%s%05d%s'%(filename,fsn,fileend))
                if (len(tmp1)>0) and (tmp2 is not None):
                    tmp1=tmp1[0]
                    tmp1['Detector']='Pilatus300k'
                    header.append(tmp1)
                    data.append(tmp2)
            return data,header
    else: # Gabriel mode, if fileend is neither TIF, nor TIFF, case insensitive
        if files is None: # read only 1 file
            header=readheader(filename);
            data=readgabrieldata(filename);
            if (len(header)>0) and (data is not None):
                header=header[0]
                header['Detector']='Gabriel'
                return [data],[header]
            else:
                return [],[]
        else:
            data=[];
            header=[];
            for fsn in files:
                tmp1=readheader('%s%05d%s' % (filename,fsn,fileend))
                tmp2=readgabrieldata('%s%05d%s' % (filename,fsn,fileend))
                if (len(tmp1)>0) and (tmp2 is not None):
                    tmp1=tmp1[0];
                    tmp1['Detector']='Gabriel'
                    data.append(tmp2);
                    header.append(tmp1);
            return data,header
def getsamplenames(filename,files,fileend,showtitles='Gabriel'):
    """Prints information on the measurement files
    
    Inputs:
        filename: the beginning of the filename, or the whole filename
        fsn: the file sequence number or None if the whole filenam was supplied
            in filename
        fileend: the end of the file.
        showtitles: if this is 'Gabriel', prints column headers for the gabriel
            detector. 'Pilatus300k' prints the appropriate headers for that
            detector. All other values suppress header printing.
    Outputs:
        None
    """
    if type(files) is not types.ListType:
        files=[files]
    if showtitles =='Gabriel':
        print 'FSN\tTime\tEnergy\tDist\tPos\tTransm\tSum/Tot %\tT (C)\tTitle\t\t\tDate'
    elif showtitles=='Pilatus300k':
        print 'FSN\tTime\tEnergy\tDist\tPos\tTransm\tTitle\t\t\tDate'
    else:
        pass #do not print header
    for i in files:
        d,h=read2dB1data(filename,i,fileend);
        if len(h)<1:
            continue
        h=h[0]
        d=d[0]
        if h['Detector']=='Gabriel':
            print '%d\t%d\t%.1f\t%d\t%.2f\t%.4f\t%.1f\t%.f\t%s\t%s' % (
                h['FSN'], h['MeasTime'], h['Energy'], h['Dist'],
                h['PosSample'], h['Transm'], 100*pylab.sum(d)/h['Anode'],
                h['Temperature'], h['Title'], ('%d.%d.%d %d:%d' % (h['Day'],
                                                h['Month'],
                                                h['Year'],
                                                h['Hour'],
                                                h['Minutes'])))
        else:
            print '%d\t%d\t%.1f\t%d\t%.2f\t%.4f\t%.f\t%s\t%s' % (
                h['FSN'], h['MeasTime'], h['Energy'], h['Dist'],
                h['PosSample'], h['Transm'], 
                h['Temperature'], h['Title'], ('%d.%d.%d %d:%d' % (h['Day'],
                                                h['Month'],
                                                h['Year'],
                                                h['Hour'],
                                                h['Minutes'])))
def read2dintfile(fsns):
    """Read corrected intensity and error matrices
    
    Input:
        fsns: one or more fsn-s in a list
        
    Output:
        a list of 2d intensity matrices
        a list of error matrices
        a list of param dictionaries
    
    Note:
        It tries to load int2dnorm<FSN>.mat. If it does not succeed,
        it tries int2dnorm<FSN>.dat and err2dnorm<FSN>.dat. If these do not
        exist, int2dnorm<FSN>.dat.zip and err2dnorm<FSN>.dat.zip is tried. If
        still no luck, int2dnorm<FSN>.dat.gz and err2dnorm<FSN>.dat.gz is
        opened. If this fails as well, the given FSN is skipped. If no files
        have been loaded, empty lists are returned.
    """
    def read2dfromstream(stream):
        """Read 2d ascii data from stream.
        It uses only stream.readlines()
        Watch out, this is extremely slow!
        """
        lines=stream.readlines()
        M=len(lines)
        N=len(lines[0].split())
        data=pylab.zeros((M,N),order='F')
        for l in range(len(lines)):
            data[l]=[float(x) for x in lines[l].split()];
        del lines
        return data
    def read2dascii(filename):
        """Read 2d data from an ascii file
        If filename is not found, filename.zip is tried.
        If that is not found, filename.gz is tried.
        If that is not found either, return None.
        """
        try:
            fid=open(filename,'r')
            data=read2dfromstream(fid)
            fid.close()
        except IOError:
            try:
                z=zipfile.ZipFile(filename+'.zip','r')
                fid=z.open(filename)
                data=read2dfromstream(fid)
                fid.close()
                z.close()
            except KeyError:
                z.close()
            except IOError:
                try:
                    z=gzip.GzipFile(filename+'.gz','r')
                    data=read2dfromstream(z)
                    z.close()
                except IOError:
                    print 'Cannot find file %s (also tried .zip and .gz)' % filename
                    return None
        return data
    # the core of read2dintfile
    if type(fsns)!=types.ListType: # if only one fsn was supplied, make it a list of one element
        fsns=[fsns]
    int2d=[]
    err2d=[]
    params=[]
    for fsn in fsns: # this also works if len(fsns)==1
        try: # first try to load the mat file. This is the most effective way.
            tmp0=scipy.io.loadmat('int2dnorm%d.mat' % fsn)
            tmp=tmp0['Intensity'].copy()
            tmp1=tmp0['Error'].copy()
        except IOError: # if mat file is not found, try the ascii files
            print 'Cannot find file int2dnorm%d.mat: trying to read int2dnorm%d.dat(.gz|.zip) and err2dnorm%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
            tmp=read2dascii('int2dnorm%d.dat' % fsn);
            tmp1=read2dascii('err2dnorm%d.dat' % fsn);
        except TypeError: # if mat file was found but scipy.io.loadmat was unable to read it
            print "Malformed MAT file! Skipping."
            continue
        tmp2=readlogfile(fsn) # read the logfile
        if (tmp is not None) and (tmp1 is not None) and (tmp2 is not None): # if all of int,err and log is read successfully
            int2d.append(tmp)
            err2d.append(tmp1)
            params.append(tmp2[0])
    return int2d,err2d,params # return the lists
def write2dintfile(A,Aerr,params):
    """Save the intensity and error matrices to int2dnorm<FSN>.mat
    
    Inputs:
        A: the intensity matrix
        Aerr: the error matrix
        params: the parameter dictionary
        
    int2dnorm<FSN>.mat is written. The parameter structure is not saved,
        since it should be saved already in intnorm<FSN>.log
    """
    filename='int2dnorm%d.mat' % params['FSN'];
    scipy.io.savemat(filename,{'Intensity':A,'Error':Aerr});
def readintfile(filename):
    """Read intfiles.

    Input:
        filename: the file name, eg. intnorm123.dat

    Output:
        A dictionary with 'q' holding the values for the momentum transfer,
            'Intensity' being the intensity vector and 'Error' has the error
            values. These three fields are numpy ndarrays.
    """
    try:
        fid=open(filename,'rt');
        lines=fid.readlines();
        fid.close();
        ret={'q':[],'Intensity':[],'Error':[],'Area':[]};
        for line in lines:
            sp=string.split(line);
            if len(sp)>=3:
                try:
                    tmpq=float(sp[0]);
                    tmpI=float(sp[1]);
                    tmpe=float(sp[2]);
                    if len(sp)>3:
                        tmpa=float(sp[3]);
                    else:
                        tmpa=pylab.nan;
                    ret['q'].append(tmpq);
                    ret['Intensity'].append(tmpI);
                    ret['Error'].append(tmpe);
                    ret['Area'].append(tmpa);
                except ValueError:
                    #skip erroneous line
                    pass
    except IOError:
        return []
    ret['q']=pylab.array(ret['q'])
    ret['Intensity']=pylab.array(ret['Intensity'])
    ret['Error']=pylab.array(ret['Error'])
    ret['Area']=pylab.array(ret['Area'])
    if len([1 for x in ret['Area'] if pylab.isnan(x)==False])==0:
        del ret['Area']
    return ret
def writeintfile(qs, ints, errs, header, areas=None, filetype='intnorm'):
    """Save 1D scattering data to intnorm files.
    
    Inputs:
        qs: list of q values
        ints: list of intensity (scattering cross-section) values
        errs: list of error values
        header: header dictionary (only the key 'FSN' is used)
        areas [optional]: list of effective area values or None
        filetype: 'intnorm' to save 'intnorm%d.dat' files. 'intbinned' to
            write 'intbinned%d.dat' files. Case insensitive.
    """
    filename='%s%d.dat' % (filetype, header['FSN'])
    fid=open(filename,'wt');
    for i in range(len(qs)):
        if areas is None:
            fid.write('%e %e %e\n' % (qs[i],ints[i],errs[i]))
        else:
            fid.write('%e %e %e %e\n' % (qs[i],ints[i],errs[i],areas[i]))
    fid.close();
    
def readintnorm(fsns, filetype='intnorm'):
    """Read intnorm*.dat files along with their headers
    
    Input:
        fsns: one or more fsn-s.
        
    Outputs:
        A vector of dictionaries, in each dictionary the self-explanatory
            'q', 'Intensity' and 'Error' fields are present.
        A vector of parameters, read from the logfiles.
    
    Note:
        When loading only one fsn, the outputs will be still in lists, thus
            lists with one elements will be returned.
    """
    if type(fsns) != types.ListType:
        fsns=[fsns];
    data=[];
    param=[];
    for fsn in fsns:
        filename='%s%d.dat' % (filetype, fsn)
        tmp=readintfile(filename)
        tmp2=readlogfile(fsn)
        if (tmp2!=[]) and (tmp!=[]):
            data.append(tmp);
            param.append(tmp2[0]);
    return data,param
def readbinned(fsn):
    """Read intbinned*.dat files along with their headers.
    
    This is a shortcut to readintnorm(fsn,'intbinned')
    """
    return readintnorm(fsn,'intbinned');
def readlogfile(fsn):
    """Read logfiles.
    
    Input:
        fsn: the file sequence number(s). It is possible to
            give a single value or a list
            
    Output:
        a list of dictionaries corresponding to the header files. This
            is a list with one element if only one fsn was given. Thus the
            parameter dictionary will be params[0].
    """
    # this dictionary contains the floating point parameters. The key (first)
    # part of each item is the text before the value, up to (not included) the
    # colon. Ie. the key corresponding to line "FSN: 123" is 'FSN'. The value
    # (second) part of each item is the field (key) name in the resulting param
    # dictionary. If two float params are to be read from the same line (eg. the
    # line "Beam size X Y: 123.45, 135.78", )
    logfile_dict_float={'FSN':'FSN',
                        'Sample-to-detector distance (mm)':'Dist',
                        'Sample thickness (cm)':'Thickness',
                        'Sample transmission':'Transm',
                        'Sample position (mm)':'PosSample',
                        'Temperature':'Temperature',
                        'Measurement time (sec)':'MeasTime',
                        'Scattering on 2D detector (photons/sec)':'ScatteringFlux',
                        'Dark current subtracted (cps)':'dclevel',
                        'Dark current FSN':'FSNdc',
                        'Empty beam FSN':'FSNempty',
                        'Glassy carbon FSN':'FSNref1',
                        'Glassy carbon thickness (cm)':'Thicknessref1',
                        'Energy (eV)':'Energy',
                        'Calibrated energy (eV)':'EnergyCalibrated',
                        'Beam x y for integration':('BeamPosX','BeamPosY'),
                        'Normalisation factor (to absolute units)':'NormFactor',
                        'Relative error of normalisation factor (percentage)':'NormFactorRelativeError',
                        'Beam size X Y (mm)':('BeamsizeX','BeamsizeY'),
                        'Pixel size of 2D detector (mm)':'PixelSize',
                        'Primary intensity at monitor (counts/sec)':'Monitor',
                        'Primary intensity calculated from GC (photons/sec/mm^2)':'PrimaryIntensity',
                        'Sample rotation around x axis':'RotXsample',
                        'Sample rotation around y axis':'RotYsample'
                        }
    #this dict. contains the string parameters
    logfile_dict_str={'Sample title':'Title'}
    #this dict. contains the bool parameters
    logfile_dict_bool={'Injection between Empty beam and sample measurements?':'InjectionEB',
                       'Injection between Glassy carbon and sample measurements?':'InjectionGC'
                       }
    #some helper functions
    def getname(linestr):
        return string.strip(linestr[:string.find(linestr,':')]);
    def getvaluestr(linestr):
        return string.strip(linestr[(string.find(linestr,':')+1):])
    def getvalue(linestr):
        return float(getvaluestr(linestr))
    def getfirstvalue(linestr):
        valuepart=getvaluestr(linestr)
        return float(valuepart[:string.find(valuepart,' ')])
    def getsecondvalue(linestr):
        valuepart=getvaluestr(linestr)
        return float(valuepart[(string.find(valuepart,' ')+1):])
    def getvaluebool(linestr):
        valuepart=getvaluestr(linestr)
        if string.find(valuepart,'n')>=0:
            return False
        elif string.find(valuepart,'y')>0:
            return True
        else:
            return None
    #this is the beginning of readlogfile().
    if type(fsn)!=types.ListType: # if fsn is not a list, convert it to a list
        fsn=[fsn];
    params=[]; #initially empty
    for f in fsn:
        filename='intnorm%d.log' % f #the name of the file
        try:
            param={};
            fid=open(filename,'r'); #try to open. If this fails, an exception is raised
            lines=fid.readlines(); # read all lines
            fid.close(); #close
            del fid;
            for line in lines:
                name=getname(line);
                for k in logfile_dict_float.keys():
                    if name==k:
                        if type(logfile_dict_float[k]) is types.StringType:
                            param[logfile_dict_float[k]]=getvalue(line);
                        else: # type(logfile_dict_float[k]) is types.TupleType
                            param[logfile_dict_float[k][0]]=getfirstvalue(line);
                            param[logfile_dict_float[k][1]]=getsecondvalue(line);
                for k in logfile_dict_str.keys():
                    if name==k:
                        param[logfile_dict_str[k]]=getvaluestr(line);
                for k in logfile_dict_bool.keys():
                    if name==k:
                        param[logfile_dict_bool[k]]=getvaluebool(line);
            param['Title']=string.replace(param['Title'],' ','_');
            param['Title']=string.replace(param['Title'],'-','_');
            params.append(param);
            del lines;
        except IOError, detail:
            print 'Cannot find file %s.' % filename
    return params;
def writelogfile(header,ori,thick,dc,realenergy,distance,mult,errmult,reffsn,
                 thickGC,injectionGC,injectionEB,pixelsize,mode='Pilatus300k'):
    """Write logfiles.
    
    Inputs:
        header: header structure as read by readheader()
        ori: origin vector of 2
        thick: thickness of the sample (cm)
        dc: if mode=='Pilatus300k' then this is the DC level which is subtracted.
            Otherwise it is the dark current FSN.
        realenergy: calibrated energy (eV)
        distance: sample-to-detector distance (mm)
        mult: absolute normalization factor
        errmult: error of mult
        reffsn: FSN of GC measurement
        thickGC: thickness of GC (cm)
        injectionGC: if injection occurred between GC and sample measurements:
            'y' or True. Otherwise 'n' or False
        injectionEB: the same as injectionGC but for empty beam and sample.
        pixelsize: the size of the pixel of the 2D detector (mm)
        mode: 'Pilatus300k' or 'Gabriel'. If invalid, it defaults to 'Gabriel'
        
    Output:
        a file intnorm<fsn>.log is saved to the current directory
    """
    
    if injectionEB!='y' and injectionEB!='n':
        if injectionEB:
            injectionEB='y'
        else:
            injectionEB='n'
    if injectionGC!='y' and injectionGC!='n':
        if injectionGC:
            injectionGC='y'
        else:
            injectionGC='n'
    name='intnorm%d.log' % header['FSN']
    fid=open(name,'wt')
    fid.write('FSN:\t%d\n' % header['FSN'])
    fid.write('Sample title:\t%s\n' % header['Title'])
    fid.write('Sample-to-detector distance (mm):\t%d\n' % distance)
    fid.write('Sample thickness (cm):\t%f\n' % thick)
    fid.write('Sample transmission:\t%.4f\n' % header['Transm'])
    fid.write('Sample position (mm):\t%.2f\n' % header['PosSample'])
    fid.write('Temperature:\t%.2f\n' % header['Temperature'])
    fid.write('Measurement time (sec):\t%.2f\n' % header['MeasTime'])
    fid.write('Scattering on 2D detector (photons/sec):\t%.1f\n' % (header['Anode']/header['MeasTime']))
    if mode=='Pilatus300k':
        fid.write('Dark current subtracted (cps):\t%d\n' % dclevel)
    else:
        fid.write('Dark current FSN:\t%d\n' % dc)
    fid.write('Empty beam FSN:\t%d\n' % header['FSNempty'])
    fid.write('Injection between Empty beam and sample measurements?:\t%s\n' % injectionEB)
    fid.write('Glassy carbon FSN:\t%d\n' % reffsn)
    fid.write('Glassy carbon thickness (cm):\t%.4f\n' % thickGC)
    fid.write('Injection between Glassy carbon and sample measurements?:\t%s\n' % injectionGC)
    fid.write('Energy (eV):\t%.2f\n' % header['Energy'])
    fid.write('Calibrated energy (eV):\t%.2f\n' % realenergy)
    fid.write('Beam x y for integration:\t%.2f %.2f\n' % (ori[0],ori[1]))
    fid.write('Normalisation factor (to absolute units):\t%e\n' % mult)
    fid.write('Relative error of normalisation factor (percentage):\t%.2f\n' % (100*errmult/mult))
    fid.write('Beam size X Y (mm):\t%.2f %.2f\n' % (header['BeamsizeX'],header['BeamsizeY']))
    fid.write('Pixel size of 2D detector (mm):\t%.4f\n' % pixelsize)
    fid.write('Primary intensity at monitor (counts/sec):\t%.1f\n' % (header['Monitor']/header['MeasTime']))
    fid.write('Primary intensity calculated from GC (photons/sec/mm^2):\t%e\n'% (header['Monitor']/header['MeasTime']/mult/header['BeamsizeX']/header['BeamsizeY']))
    fid.write('Sample rotation around x axis:\t%e\n'%header['Rot1'])
    fid.write('Sample rotation around y axis:\t%e\n'%header['Rot2'])
    fid.close()
def readwaxscor(fsns):
    """Read corrected waxs file
    
    Inputs:
        fsns: a range of fsns or a single fsn.
        
    Output:
        a list of scattering data dictionaries (see readintfile())
    """
    if type(fsns)!=types.ListType:
        fsns=[fsns]
    waxsdata=[];
    for fsn in fsns:
        try:
            filename='waxs_%05d.cor' % fsn
            tmp=pylab.load(filename)
            if tmp.shape[1]==3:
                tmp1={'q':tmp[:,0],'Intensity':tmp[:,1],'Error':tmp[:,2]}
            waxsdata.append(tmp1)
        except IOError:
            print '%s not found. Skipping it.' % filename
    return waxsdata
def readenergyfio(filename,files,fileend):
    """Read abt_*.fio files.
    
    Inputs:
        filename: beginning of the file name, eg. 'abt_'
        files: a list or a single fsn number, eg. [1, 5, 12] or 3
        fileend: extension of a file, eg. '.fio'
    
    Outputs: three lists:
        energies: the uncalibrated (=apparent) energies for each fsn.
        samples: the sample names for each fsn
        muds: the mu*d values for each fsn
    """
    if type(files)!=types.ListType:
        files=[files]
    samples=[]
    energies=[]
    muds=[]
    for f in files:
        mud=[];
        energy=[];
        fname='%s%05d%s' % (filename,f,fileend)
        try:
            fid=open(fname,'r')
            lines=fid.readlines()
            samples.append(lines[5].strip())
            for l in lines[41:]:
                tmp=l.strip().split()
                if len(tmp)==11:
                    try:
                        tmpe=float(tmp[0])
                        tmpmud=float(tmp[-1])
                        energy.append(tmpe)
                        mud.append(tmpmud)
                    except ValueError:
                        pass
            muds.append(mud)
            energies.append(energy)
        except IOError:
            print 'Cannot find file %s.' % fname
    return (energies,samples,muds)
def readf1f2(filename):
    """Load fprime files created by Hephaestus
    
    Input: 
        filename: the name (and path) of the file
    
    Output:
        an array. Each row contain Energy, f', f''
    """
    fprimes=pylab.loadtxt(filename)
    return fprimes
def getsequences(headers,ebname='Empty_beam'):
    """Separate measurements made at different energies in an ASAXS sequence
    
    Inputs:
        header: header (or param) dictionary
        ebname: the title of the empty beam measurements.
    
    Output:
        a list of lists. Each sub-list in this list contains the indices in the
        supplied header structure which correspond to the sub-sequence.
    
    Example:
        If measurements were carried out:
        EB_E1, Ref_before_E1, Sample1_E1, Sample2_E1, Ref_after_E1, EB_E2,...
        Ref_after_EN then the function will return:
        [[0,1,2,3,4],[5,6,7,8,9],...[(N-1)*5,(N-1)*5+1...N*5-1]].
        
        It is allowed that the last sequence is incomplete. However, all other
        sequences must have the same amount of measurements (references are 
        treated as samples in this case).
    """
    seqs=[]
    ebnums=[x for x in range(len(headers)) if headers[x]['Title']==ebname]
    for i in range(len(ebnums)-1):
        seqs.append(range(ebnums[i],ebnums[i+1]))
    seqs.append(range(ebnums[-1],len(headers)))
    return seqs
def mandelbrot(real,imag,iters):
    """Calculate the Mandelbrot set
    
    Inputs:
        real: a vector of the real values (x coordinate). pylab.linspace(-2,2)
            is a good choice.
        imag: a vector of the imaginary values (y coordinate). pylab.linspace(-2,2)
            is a good choice.
        iters: the number of iterations.
        
    Output:
        a matrix. Each element is the number of iterations which made the corresponding
            point to become larger than 2 in absolute value. 0 if no divergence
            up to the number of simulations
            
    Note:
        You may be curious how comes this function to this file. Have you ever
        heard of easter eggs? ;-D Btw it can be a good sample data for radial
        integration routines.
    """
    R,I=pylab.meshgrid(real,imag)
    C=R.astype('complex')
    C.imag=I
    Z=pylab.zeros(C.shape,'complex')
    N=pylab.zeros(C.shape)
    Z=C*C+C
    for n in range(iters):
        indices=(Z*Z.conj()>=4)
        N[indices]=n
        Z[indices]=0
        C[indices]=0
        Z=Z*Z+C
    return N              

def writechooch(mud,filename):
    """Saves the data read by readxanes to a format which can be recognized
    by CHOOCH
    
    Inputs:
        mud: a muds dictionary
        filename: the filename to write the datasets to
    
    Outputs:
        a file with filename will be saved
    """
    f=open(filename,'wt')
    f.write('%s\n' % mud['Title'])
    f.write('%d\n' % len(mud['Energy']))
    for i in range(len(mud['Energy'])):
        f.write('%f\t%f\n' % (mud['Energy'][i],pylab.exp(-mud['Mud'][i])))
    f.close()
def readxanes(filebegin,files,fileend,energymeas,energycalib):
    """Read energy scans from abt_*.fio files by readenergyfio() then
    put them on a correct energy scale.
    
    Inputs:
        filebegin: the beginning of the filename, like 'abt_'
        files: FSNs, like range(2,36)
        fileend: the end of the filename, like '.fio'
        energymeas: list of the measured energies
        energycalib: list of the true energies corresponding to the measured
            ones
    
    Output:
        a list of mud dictionaries. Each dict will have the following items:
            'Energy', 'Mud', 'Title', 'scan'. The first three are
            self-describing. The last will be the FSN.
    """
    muds=[];
    if type(files)!=types.ListType:
        files=[files]
    for f in files:
        energy,sample,mud=readenergyfio(filebegin,f,fileend)
        if len(energy)>0:
            d={}
            d['Energy']=energycalibration(energymeas,energycalib,pylab.array(energy[0]))
            d['Mud']=pylab.array(mud[0])
            d['Title']=sample[0]
            d['scan']=f
            muds.append(d);
    return muds
def writef1f2(f1f2,filename):
    """Saves f1f2 data to file
    
    Inputs:
        f1f2: matrix of anomalous correction terms
        filename: file name
    """
    pylab.savetxt(filename,f1f2,delimiter='\t')
def readabt(filename):
    """Read abt_*.fio type files.
    
    Input:
        filename: the name of the file.
        
    Output:
        A dictionary with the following fields:
            'title': the sample title
            'mode': 'Energy' or 'Motor'
            'columns': the description of the columns in 'data'
            'data': the data found in the file, in a matrix.
    """
    try:
        f=open(filename,'rt');
    except IOError:
        print 'Cannot open file %s' % filename
        return None
    rows=0;
    a=f.readline(); rows=rows+1;
    while a[:2]!='%c' and len(a)>0:
        a=f.readline();  rows=rows+1;
    if len(a)<=0:
        print 'Invalid format: %c not found'
        f.close()
        return None
    a=f.readline(); rows=rows+1;
    if a[:7]==' ENERGY':
        mode='Energy'
    elif a[:4]==' MOT':
        mode='Motor'
    else:
        print 'Unknown scan type: %s' % a
        f.close()
        return None
    f.readline(); rows=rows+1;
    title=f.readline()[:-1]; rows=rows+1;
    while a[:2]!='%d' and len(a)>0:
        a=f.readline(); rows=rows+1;
    if len(a)<=0:
        print 'Invalid format: %d not found'
        f.close()
        return None
    columns=[];
    a=f.readline(); rows=rows+1;
    while a[:4]==' Col':
        columns.append(a.split('  ')[0][17:]);
        a=f.readline(); rows=rows+1;
        #print a
    #print a[:4]
    f.seek(-len(a),1)
    rows=rows-1;
    #print rows
    matrix=pylab.loadtxt(f)
    f.close()
    return {'title':title,'mode':mode,'columns':columns,'data':matrix}

#EXPERIMENTAL (DANGER ZONE)
def trimq(data,qmin=-pylab.inf,qmax=pylab.inf):
    indices=(data['q']<=qmax) & (data['q']>=qmin)
    data1={}
    for k in data.keys():
        data1[k]=data[k][indices]
    return data1
def guinierfit(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
    data=trimq(data,qmin,qmax)
    x1=data['q']**2;
    err1=pylab.absolute(data['Error']/data['Intensity']);
    y1=pylab.log(data['Intensity']);
    Rg,G,dRg,dG=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data['q']**2,pylab.log(data['Intensity']),'.');
        pylab.plot(data['q']**2,Rg*data['q']**2+G,'-',color='red');
        pylab.xlabel('$q^2$ (1/%c$^2$)' % 197)
        pylab.ylabel('ln I');
    return pylab.sqrt(-Rg*3),G,1.5/pylab.sqrt(-Rg*3)*dRg,dG
def porodfit(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
    data=trimq(data,qmin,qmax)
    x1=data['q']**4;
    err1=data['Error']*x1;
    y1=data['Intensity']*x1;
    a,b,aerr,berr=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data['q']**4,data['Intensity']*data['q']**4,'.');
        pylab.plot(data['q']**4,a*data['q']**4+b,'-',color='red');
        pylab.xlabel('$q^4$ (1/%c$^4$)' % 197)
        pylab.ylabel('I$q^4$');
    return a,b,aerr,berr
def subconstbg(data,bg,bgerror):
    return {'q':data['q'].copy(),
           'Intensity':data['Intensity']-bg,
           'Error':pylab.sqrt(data['Error']**2+bgerror**2)};
def unifiedfit(data,B,G,Rg,qmin=-pylab.inf,qmax=pylab.inf,maxiter=1000):
    data=trimq(data,qmin,qmax)
    def fitfun(data,x,y,err):
        G=data[0]
        B=data[1]
        Rg=data[2]
        return pylab.sum((unifiedscattering(x,B,G,Rg,4)-y)**2/err**2)
    return fmin(fitfun,pylab.array([B,G,Rg]),args=(data['q'],data['Intensity'],data['Error']),maxiter=maxiter,disp=True)
def shullroess(data,r0min,r0max,r0stepping,qmin=None,qmax=None):
    """Do a Shull-Roess fitting on the scattering data dictionary.
    
    Inputs:
        data: scattering data dictionary
        r0min: the minimal value of r0
        r0max: the maximal value of r0
        r0stepping: the stepping of r0
        qmin, qmax (optional): borders of ROI
        
    Output:
        r0: the fitted value of r0
        n: the fitted value of n
    Note: This first searches sfor r0, which best linearizes the
            log(Intensity) vs. log(q**2+3/r0**2) relation.
            After this is found, the parameters of the fitted line give the
            parameters of a Maxwellian-like particle size distribution function.
            
    EXPERIMENTAL!!!!!
    """
    print "BIG FAT WARNING: shullroess() is an EXPERIMENTAL function. You may not get what you expect!"
    Iexp=pylab.array(data['Intensity'])
    qexp=pylab.array(data['q'])
    errexp=pylab.array(data['Error'])
    if qmin is not None:
        len0=len(qexp)
        Iexp=Iexp[qexp>=qmin]
        errexp=errexp[qexp>=qmin]
        qexp=qexp[qexp>=qmin]
        len1=len(qexp)
        print '%d points have been cut from the beginning.' % (len0-len1)
    if qmax is not None:
        len0=len(qexp)
        Iexp=Iexp[qexp<=qmax]
        errexp=errexp[qexp<=qmax]
        qexp=qexp[qexp<=qmax]
        len1=len(qexp)
        print '%d points have been cut from the end.' % (len0-len1)
    logIexp=pylab.log(Iexp)
    errlogIexp=errexp/Iexp
    r0s=pylab.arange(r0min,r0max,r0stepping,dtype=float)
    chi2=pylab.zeros(r0s.shape)
    for i in range(len(r0s)):
        xdata=pylab.log(qexp**2+3/r0s[i]**2)
        a,b,aerr,berr=linfit(xdata,logIexp,errlogIexp)
        chi2[i]=pylab.sum(((xdata*a+b)-logIexp)**2)
    pylab.subplot(2,2,1)
    pylab.plot(r0s,chi2)
    bestindex=(chi2==chi2.min())
    xdata=pylab.log(qexp**2+3/r0s[bestindex]**2)
    a,b,aerr,berr=linfit(xdata,logIexp,errlogIexp)
    n=-(a*2.0+4.0)
    print 'best fit: r0=', r0s[bestindex][0], ', n=',n,'with chi2: ',chi2[bestindex]
    print a
    print b
    print 'Guinier approximation: ',r0s[bestindex]*qexp.max(),'?<<? 1'
    pylab.subplot(2,2,2)
    pylab.plot(xdata,logIexp,'.',label='Measured')
    pylab.plot(xdata,a*xdata+b,label='Fitted')
    pylab.legend()
    pylab.subplot(2,2,3)
    pylab.plot(r0s,maxwellian(n,r0s[bestindex],r0s))
    return r0s[bestindex],n

def findpeak(xdata,ydata):
    """GUI tool for locating peaks by zooming on them
    
    Inputs:
    
    EXPERIMENTAL!!!!
    """
    print "BIG FAT WARNING: findpeak() is an EXPERIMENTAL function. You may not get what you expect!"

def convolutef1f2(f1f2,width):
    """Convolute f1f2 value with a Lorentzian of a given width.
    
    Input:
        f1f2: f1f2 structure as read by readf1f2().
        width: the FWHM of the Lorentzian in eV-s
        
    Output:
        the convolved f1f2 dataset.
        
    Note:
        EXPERIMENTAL!!!!
    """
    print "BIG FAT WARNING: convolutef1f2() is an EXPERIMENTAL function. You may not get what you expect!"
    
    #it is assumed that the f1f2 data consists of EQUIDISTANT points!
    Espacing=pylab.absolute(f1f2[1,0]-f1f2[0,0])
    #the smearing function will be placed to the origin. It will be cut off at
    # +/- width/2.0*100, as outside this interval the value of the lorentzian
    # is smaller than 0.01*peak value (at origin)
    
    Espread1=pylab.arange(0,width/2.0*100,Espacing)
    Espread2=-Espread1
    Espread=unique(pylab.hstack((Espread1,Espread2)))   
    spread=lorentzian(0,width/2.0,Espread)
    f1conv=pylab.convolve(f1f2[:,1],spread,'full')
    pass
def radint2(data,dataerr,energy,distance,res,bcx,bcy,mask,q):
    """EXPERIMENTAL!!!!!
    """
    print "BIG FAT WARNING: radint2() is an EXPERIMENTAL function. You may not get what you expect!"
    print "Preprocessing..."
    radata1=data.flatten()
    raerror1=dataerr.flatten()
    ramask1=mask.flatten()
    radata=radint_ng.radintArray(data.size)
    raerror=radint_ng.radintArray(data.size)
    ramask=radint_ng.radintmaskArray(data.size)
    raq=radint_ng.radintArray(len(q))
    raIntensity=radint_ng.radintArray(len(q))
    raError=radint_ng.radintArray(len(q))
    raArea=radint_ng.radintArray(len(q))
    raWeight=radint_ng.radintArray(len(q))
    for x in range(len(q)):
        raq[x]=q[x]
    for x in range(data.shape[0]):
        for y in range(data.shape[1]):
            radata[x+y*data.shape[0]]=data[x,y];
            raerror[x+y*data.shape[0]]=dataerr[x,y];
            ramask[x+y*data.shape[0]]=int(mask[x,y]);
    print "Calling doradint..."
    res=radint_ng.doradint(radata,raerror,ramask,data.shape[0],data.shape[1],energy,distance,res,res,bcx,bcy,raq,raIntensity,raError,raArea,raWeight,len(q));
    print "Postprocessing..."
    if res:
        Intensity=pylab.zeros(len(q))
        Error=pylab.zeros(len(q))
        Area=pylab.zeros(len(q))
        Weight=pylab.zeros(len(q))
        for x in range(len(q)):
            Intensity[x]=raIntensity[x]
            Error[x]=raError[x]
            Area[x]=raArea[x]
            Weight[x]=raWeight[x]
        print "Returning with success"
        return (Intensity,Error,Area,Weight)
    else:
        print "Returning with error"
        return None
def stackdata(tup):
    """Stack two or more scattering data dictionaries above each other.
    
    Inputs:
        tup: a tuple containing the dictionaries
    
    Output:
        a scattering data dictionary of the output
        
    NOTE: EXPERIMENTAL!!!!
    """
    print "BIG FAT WARNING: stackdata() is an EXPERIMENTAL function. You may not get what you expect!"
    data={}
    data['q']=pylab.vstack(tuple([t['q'].reshape(t['q'].size,1) for t in tup]))
    data['Intensity']=pylab.vstack(tuple([t['Intensity'].reshape(t['Intensity'].size,1) for t in tup]))
    data['Error']=pylab.vstack(tuple([t['Error'].reshape(t['Error'].size,1) for t in tup]))
    tmp=pylab.vstack((data['q'].reshape(1,data['q'].size),data['Intensity'].reshape(1,data['Intensity'].size),data['Error'].reshape(1,data['Error'].size)))
    tmp=tmp.transpose()
    #tmp.sort(0)
    data['q']=tmp[:,0]
    data['Intensity']=tmp[:,1]
    data['Error']=tmp[:,2]
    #print data['q'].min()
    #print data['q'].max()
    return data
def selectasaxsenergies(f1f2,energymin,energymax,Nenergies=3,kT=1000,NITERS=30000,energydistance=0,stepsize=0.5):
    """Select energies for ASAXS measurement by minimizing the condition number of the ASAXS matrix.
    
    Inputs:
        f1f2: a numpy array of 3 columns and N rows. Each row should be: energy, f1, f2
        energymin: smallest energy
        energymax: largest energy
        Nenergies: the number of energies
        kT: the temperature times Boltzmann's constant, for the Metropolis algorithm
        NITERS: how much iterations should we do
        energydistance: if two energies are nearer than this value, they are considered
            the same
        stepsize: the step size for the energies. This should be larger than
            energydistance.
            
    Returns:
        energies: array of the chosen energies
        
        also two graphs will be plotted
    """
    
    def matrixcond(f1f2,energies,atomicnumber=0):
        """Calculate the condition number of the ASAXS matrix
        
        Inputs:
            f1f2: matrix for the anomalous dispersion coefficients
            energies: the chosen energies
            atomicnumber: the atomic number of the resonant atom. Set it zero
                if you want the evaluation according to Stuhrmann.
             
        Outputs:
            the condition number. If f1 denotes the column vector of the f1
            values and f2 for the f2 values, then the ASAXS matrix is
            calculated as:
            
            B=inv(A^T.A).A^T
            
            where
            
            A=[1, 2* (Z+f1), (Z+f1)^2+f2^2]
            
            and Z is the atomic  number.
            
            The 2nd order (=euclidean) condition number of B will be returned.
            The pylab.cond() function is used to determine this.  If the matrix
            is non-square (ie. rectangular), this type of condition number can
            still be determined from the singular value decomposition.
             
        """
        f1=pylab.interp(energies,f1f2[:,0],f1f2[:,1])
        f2=pylab.interp(energies,f1f2[:,0],f1f2[:,2])
        A=pylab.ones((len(energies),3));
        A[:,1]=2*(f1+atomicnumber);
        A[:,2]=(f1+atomicnumber)**2+f2**2;
        B=pylab.dot(pylab.inv(pylab.dot(A.T,A)),A.T);
        return pylab.cond(B)

    pylab.np.random.seed()
    f1f2=f1f2[f1f2[:,0]<=(energymax+100),:]
    f1f2=f1f2[f1f2[:,0]>=(energymin-100),:]
    energies=pylab.rand(Nenergies)*(energymax-energymin)+energymin
    c=matrixcond(f1f2,energies)
    ok=False
    oldenergies=energies.copy()
    oldc=c
    cs=pylab.zeros(NITERS)
    drops=0
    eidx=0
    sign=0
    badmovements=0
    condmin=c
    energiesmin=energies
    print 'Initial energies: ',energies
    for i in range(NITERS):
        oldenergies=energies.copy()
        oldc=c
        ok=False
        while not ok:
            #which energy to modify?
            eidx=int(pylab.rand()*Nenergies)
            #modify energy in which direction?
            sign=2*pylab.floor(pylab.rand()*2)-1
            #modify energy
            energies[eidx]=energies[eidx]+sign*stepsize
            # if the modified energy is inside the bounds and the current
            # energies are different, go on.
            if energies.min()>=energymin and energies.max()<=energymax and len(unique(energies,lambda a,b:(abs(a-b)<energydistance)))==Nenergies:
                ok=True
            else: # if not, drop this and re-calculate new energy
                energies=oldenergies.copy()
                badmovements=badmovements+1
#                print 'bad: i=',i,'energies: ',energies
#        print energies
#        print oldenergies
        #calculate the condition number of the ASAXS eval. matrix with these energies.
        try:
            c=matrixcond(f1f2,energies)
        except pylab.np.linalg.LinAlgError:
            energies=oldenergies
            c=oldc
            drops=drops+1
        else:
            if c>oldc: #if the condition number is larger than the old one,
                if pylab.rand()>(pylab.exp(c-oldc)/kT): # drop with some probability
                    energies=oldenergies
                    c=oldc
                    drops=drops+1
        cs[i]=c # save the current value for the condition number
        if pylab.mod(i,1000)==0: # printing is slow, only print every 1000th step
#            print i
            pass
        if c<condmin:
            condmin=c;
            energiesmin=energies.copy()
    energies=energiesmin
    f1end=pylab.interp(energies,f1f2[:,0],f1f2[:,1])
    f2end=pylab.interp(energies,f1f2[:,0],f1f2[:,2])
    pylab.semilogx(cs)
    pylab.xlabel('Step number')
    pylab.ylabel('Condition number of the matrix')
    a=pylab.gca().axis()
    pylab.gca().axis((a[0],a[1],a[2],cs[0]))
    pylab.figure()
    pylab.plot(f1f2[:,0],f1f2[:,1])
    pylab.plot(f1f2[:,0],f1f2[:,2])
    pylab.plot(energies,f1end,markersize=10,marker='o',linestyle='')
    pylab.plot(energies,f2end,markersize=10,marker='o',linestyle='')
    ax=pylab.gca().axis()
    pylab.plot([energymin,energymin],[ax[2],ax[3]],color='black',linestyle='--')
    pylab.plot([energymax,energymax],[ax[2],ax[3]],color='black',linestyle='--')
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('f1 and f2')
    pylab.title('f1 and f2 values from Monte Carlo simulation.\nkT=%f, N=%d, cond_opt=%f' % (kT,Nenergies,condmin))
    print 'Drops: ',drops
    print 'Bad movements: ',badmovements
    print 'Energies: ',energies
    print 'f1 values: ',f1end
    print 'f2 values: ',f2end
    print 'Optimal condition number: ',condmin
    print 'Step size: ',stepsize
    print 'kT: ',kT
    return energies
def radhist(data,energy,distance,res,bcx,bcy,mask,q,I):
    """Do radial histogramming on 2D scattering images, according to the idea
    of Teemu Ikonen
    
    Inputs:
        data: the intensity matrix
        energy: the (calibrated) beam energy (eV)
        distance: the distance from the sample to the detector (mm)
        res: pixel size in mm-s. Both x and y (row and column) direction can
            be given if wished, in a list with two elements. A scalar value
            means that the pixel size is equal in both directions
        bcx: the coordinate of the beam center in the x (row) direction,
            starting from ZERO
        bcy: the coordinate of the beam center in the y (column) direction,
            starting from ZERO
        mask: the mask matrix (of the same size as data). Nonzero is masked,
            zero is not masked
        q: the q bins at which the histogram is requested. It should be 
            defined in 1/Angstroems.
        I: the intensity bins
        
    Output:
        the histogram matrix
    """
    if type(res)!=types.ListType:
        res=[res,res];
    if len(res)==1:
        res=[res[0], res[0]]
    if len(res)>2:
        raise ValueError('res should be a scalar or a nonempty vector of length<=2')
    if data.shape!=mask.shape:
        raise ValueError('data and mask should be of the same shape')
    M=data.shape[0] # number of rows
    N=data.shape[1] # number of columns
    
    # Creating D matrix which is the distance of the sub-pixels from the origin.
    Y,X=pylab.meshgrid(pylab.arange(data.shape[1]),pylab.arange(data.shape[0]));
    D=pylab.sqrt((res[0]*(X-bcx))**2+
                 (res[1]*(Y-bcy))**2)
    # Q-matrix is calculated from the D matrix
    q1=4*pylab.pi*pylab.sin(0.5*pylab.arctan(D/float(distance)))*energy/float(HC)
    # eliminating masked pixels
    data=data[mask==0]
    q1=q1[mask==0]
    q=pylab.array(q)
    q1=q1[pylab.isfinite(data)]
    data=data[pylab.isfinite(data)]
    # initialize the output matrix
    hist=pylab.zeros((len(I),len(q)))
    # set the bounds of the q-bins in qmin and qmax
    qmin=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmin.insert(0,q[0])
    qmin=pylab.array(qmin)
    qmax=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmax.append(q[-1])
    qmax=pylab.array(qmax)
    # go through every pixel
    for l in range(len(q)):
        indices=((q1<=qmax[l])&(q1>qmin[l])) # the indices of the pixels which belong to this q-bin
        hist[:,l]=scipy.stats.stats.histogram2(data[indices],I)/pylab.sum(indices.astype('float'))
    return hist
def directdesmear(data,smoothing,x0,pixelsize,dist,beamlength):
    data1=data[(x0-1):];
    tck=scipy.interpolate.splrep(pylab.arange(len(data1)),data1,1/(data1**2),s=smoothing)
    
    pass

