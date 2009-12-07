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
#import Tkinter
import sys
import time
import os
import shutil
import matplotlib.widgets
import matplotlib.nxutils
import scipy.optimize
import scipy.special
import scipy.stats.stats
import scipy.interpolate

_B1config={'measdatadir':'.',
           'evaldatadir':'.',
           'calibdir':'.',
           'distancetoreference':219,
           'pixelsize':0.798,
           'detector':'Gabriel',
           '2dfileprefix':'ORG',
           '2dfilepostfix':'.DAT',
           'GCareathreshold':10,
           'detshift':0,
           'refdata':[{'thick':143e-4,'pos':129,'data':'GC155.dat'},
                      {'thick':508e-4,'pos':139,'data':'GC500.dat'},
                      {'thick':992e-4,'pos':159,'data':'GC1000.dat'}],
            'refposprecision':0.5
           }
_pausemode=True
HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units
def savespheres(spheres,filename):
    """Save sphere structure in a file.
    
    Inputs:
        spheres: sphere matrix
        filename: filename
    """
    pylab.savetxt(filename,spheres,delimiter='\t')
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
    
    if (type(qrange)!=types.ListType) and (type(qrange)!=pylab.ndarray):
        qrange=[qrange]
    if (type(spheres)!=types.ListType) and (type(spheres)!=pylab.ndarray):
        spheres=[spheres]
    Intensity=pylab.zeros(qrange.size)
    if (type(spheres)==types.ListType):
        for i in range(len(spheres)):
            Intensity=Intensity+fsphere(qrange,spheres[i])**2
    if (type(spheres)==pylab.ndarray):
        if spheres.ndim==1:
            for i in range(len(spheres)):
                Intensity=Intensity+fsphere(qrange,spheres[i])**2
            return Intensity
        elif spheres.shape[1]<4:
            raise ValueError("Not enough columns in spheres structure")
        elif spheres.shape[1]<5:
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
        maxiter: maximum number of iterations for scipy.optimize.fmin
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
    orig=scipy.optimize.fmin(targetfunc,pylab.array(orig_initial),args=(data,1-mask),maxiter=maxiter,disp=True)
    
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
def xanes2f1f2(mud,smoothing,element,edge,title,substitutepoints=[],startpoint=-pylab.inf,endpoint=pylab.inf,postsmoothing=[],prechoochcutoff=[-pylab.inf,pylab.inf]):
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
        prechoochcutoff: A vector of two. It determines the interval which should
            be supplied to CHOOCH. You can use this to eliminate truncation
            effects introduced by spline smoothing.
        
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
    #pre-chooch cutoff
    indices=(B['Energy']<=prechoochcutoff[1]) & (B['Energy']>=prechoochcutoff[0])
    B['Energy']=B['Energy'][indices]
    B['Mud']=B['Mud'][indices]
    #plotting
    pylab.plot(B['Energy'],B['Mud'])
    pylab.legend(['$\mu d$ measured','$\mu d$ smoothed'],loc='best');
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('$\mu d$')
    pylab.title(title)
    #saving figure of the smoothed dataset
    pylab.savefig("xanes_smoothing_%s.svg" % title,dpi=300,papertype='a4',format='svg',transparent=True)
    pylab.gcf().show()
    pylab.figure()
    # CHOOCH-ing
    writechooch(B,'choochin.tmp')
    f1f2=execchooch(B,element,edge)
    # post-CHOOCH smoothing
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
    #plotting
    pylab.plot(f1f2[:,0],f1f2[:,1:3]);
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('$f^\'$ and $f^{\'\'}$')
    pylab.title(title)
    pylab.savefig("xanes_chooch_%s.svg" % title,dpi=300,papertype='a4',format='svg',transparent=True)
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
        mask: the mask matrix. Nonzero means nonmasked
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
def reintegrateB1(fsnrange,mask,qrange=None,samples=None,savefiletype='intbinned'):
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
                writeintfile(qs,ints,errs,p,areas,filetype=savefiletype)
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
    

def basicfittinggui(data,title=''):
    """Graphical user interface to carry out basic (Guinier, Porod) fitting
    to 1D scattering data.
    
    Inputs:
        data: 1D dataset
        title: title to display
    Output:
        None, this leaves a figure open for further user interactions.
    """
    data=flatten1dsasdict(data)
    fig=pylab.figure()
    plots=['Guinier','Guinier thickness','Guinier cross-section','Porod','lin-lin','lin-log','log-lin','log-log']
    buttons=['Guinier','Guinier thickness','Guinier cross-section','Porod','Power law','Power law with background']
    fitfuns=[guinierfit,guinierthicknessfit,guiniercrosssectionfit,porodfit,powerfit,powerfitwithbackground]
    for i in range(len(buttons)):
        ax=pylab.axes((0.05,0.9-(i+1)*(0.8)/(len(buttons)+len(plots)),0.3,0.7/(len(buttons)+len(plots))))
        but=matplotlib.widgets.Button(ax,buttons[i])
        def onclick(A=None,B=None,data=data,type=buttons[i],fitfun=fitfuns[i]):
            a=pylab.axis()
            plottype=pylab.gcf().plottype
            pylab.figure()
            if plottype=='Guinier':
                xt=data['q']**2
                yt=pylab.log(data['Intensity'])
            elif plottype=='Guinier thickness':
                xt=data['q']**2
                yt=pylab.log(data['Intensity'])*xt
            elif plottype=='Guinier cross-section':
                xt=data['q']**2
                yt=pylab.log(data['Intensity'])*data['q']
            elif plottype=='Porod':
                xt=data['q']**4
                yt=data['Intensity']*xt
            else:
                xt=data['q']
                yt=data['Intensity']
            intindices=(yt>=a[2])&(yt<=a[3])
            qindices=(xt>=a[0])&(xt<=a[1])
            indices=intindices&qindices
            qmin=min(data['q'][indices])
            qmax=max(data['q'][indices])
            res=fitfun(data,qmin,qmax,testimage=True)
            if len(res)==4:
                pylab.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg' % (type,res[0],res[2],res[1],res[3]))
            elif len(res)==6:
                pylab.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg;\n %lg +/- %lg' % (type,res[0],res[3],res[1],res[4],res[2],res[5]))
            elif len(res)==8:
                pylab.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg;\n %lg +/- %lg; %lg +/- %lg' % (type,res[0],res[4],res[1],res[5],res[2],res[6],res[3],res[7]))
            pylab.gcf().show()
        but.on_clicked(onclick)
    ax=pylab.axes((0.05,0.9-(len(buttons)+len(plots))*(0.8)/(len(buttons)+len(plots)),0.3,0.7/(len(buttons)+len(plots))*len(plots) ))
    pylab.title('Plot types')
    rb=matplotlib.widgets.RadioButtons(ax,plots,active=7)
    pylab.axes((0.4,0.1,0.5,0.8))
    def onselectplottype(plottype,q=data['q'],I=data['Intensity'],title=title):
        pylab.cla()
        pylab.gcf().plottype==plottype
        if plottype=='Guinier':
            x=q**2
            y=pylab.log(I)
            pylab.plot(x,y,'.')
            pylab.xlabel('q^2')
            pylab.ylabel('ln I')
        elif plottype=='Guinier thickness':
            x=q**2
            y=pylab.log(I)*q**2
            pylab.plot(x,y,'.')
            pylab.xlabel('q^2')
            pylab.ylabel('ln I*q^2')
        elif plottype=='Guinier cross-section':
            x=q**2
            y=pylab.log(I)*q
            pylab.plot(x,y,'.')
            pylab.xlabel('q^2')
            pylab.ylabel('ln I*q')            
        elif plottype=='Porod':
            x=q**4
            y=I*q**4
            pylab.plot(x,y,'.')
            pylab.xlabel('q^4')
            pylab.ylabel('I*q^4')
        elif plottype=='lin-lin':
            pylab.plot(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        elif plottype=='lin-log':
            pylab.semilogx(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        elif plottype=='log-lin':
            pylab.semilogy(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        elif plottype=='log-log':
            pylab.loglog(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        pylab.title(title)
        pylab.gcf().plottype=plottype
        pylab.gcf().show()
    rb.on_clicked(onselectplottype)
    pylab.title(title)
    pylab.loglog(data['q'],data['Intensity'],'.')
    pylab.gcf().plottype='log-log'
    pylab.gcf().show()
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
def plot2dmatrix(A,maxval=None,mask=None,header=None,qs=[],showqscale=True,contour=None,pmin=0,pmax=1,blacknegative=False):
    """Plots the matrix A in logarithmic coloured plot
    
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
        contour: if this is None, plot a colour-mapped image of the matrix. If
            this is a positive integer, plot that much automatically selected
            contours. If a list (sequence), draw contour lines at the elements
            of the sequence.
        pmin: colour-scaling. See parameter pmax for description. 
        pmax: colour-scaling. imshow() will be called with vmin=A.max()*pmin,
            vmax=A.max()*pmax
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
    nonpos=(tmp<=0)
    tmp[nonpos]=tmp[tmp>0].min()
    tmp=pylab.log(tmp);
    tmp[pylab.isnan(tmp)]=tmp[1-pylab.isnan(tmp)].min();
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
    if contour is None:
        pylab.imshow(tmp,extent=extent,interpolation='nearest',vmin=tmp.min()+pmin*(tmp.max()-tmp.min()),vmax=tmp.min()+pmax*(tmp.max()-tmp.min()));
    else:
        if extent is None:
            extent1=[1,tmp.shape[0],1,tmp.shape[1]]
        else:
            extent1=extent;
        X,Y=pylab.meshgrid(pylab.linspace(extent1[2],extent1[3],tmp.shape[1]),
                           pylab.linspace(extent1[0],extent1[1],tmp.shape[0]))
        pylab.contour(X,Y,tmp,contour)
    if mask is not None:
        white=pylab.ones((mask.shape[0],mask.shape[1],4))
        white[:,:,3]=pylab.array(1-mask).astype('float')*0.7
        pylab.imshow(white,extent=extent,interpolation='nearest')
    if blacknegative:
        black=pylab.zeros((A.shape[0],A.shape[1],4))
        black[:,:,3][nonpos]=1
        pylab.imshow(black,extent=extent,interpolation='nearest')
    for q in qs:
        a=pylab.gca().axis()
        pylab.plot(q*pylab.cos(pylab.linspace(0,2*pylab.pi,2000)),
                   q*pylab.sin(pylab.linspace(0,2*pylab.pi,2000)),
                   color='white',linewidth=3)
        pylab.gca().axis(a)
    if header is not None:
        pylab.title("#%s: %s" % (header['FSN'], header['Title']))
#Miscellaneous routines
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
            while pylab.gcf().waitforbuttonpress()==False:
                pass
    else:
        try:
            a=float(_pausemode)
        except:
            return
        if a>0:
            time.sleep(a)
        return
def fsphere(q,R):
    """Scattering factor of a sphere
    
    Inputs:
        q: q value(s) (scalar or an array of arbitrary size and shape)
        R: radius (scalar)
        
    Output:
        the values of the scattering factor in an array of the same shape as q
    """
    return 1/q**3*(pylab.sin(q*R)-q*R*pylab.cos(q*R))
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
    return 2.0/(r0**(n+1.0)*scipy.special.gamma((n+1.0)/2.0))*(x**n)*pylab.exp(-x**2/r0**2);
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
    xdata=pylab.array(xdata);
    ydata=pylab.array(ydata);
    if xdata.size != ydata.size:
        print "The sizes of xdata and ydata should be the same."
        return
    if errdata is not None:
        if ydata.size !=errdata.size:
            print "The sizes of ydata and errdata should be the same."
            return
        errdata=pylab.array(errdata);
        S=pylab.sum(1.0/(errdata**2))
        Sx=pylab.sum(xdata/(errdata**2))
        Sy=pylab.sum(ydata/(errdata**2))
        Sxx=pylab.sum(xdata*xdata/(errdata**2))
        Sxy=pylab.sum(xdata*ydata/(errdata**2))
    else:
        S=xdata.size
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
def readheader(filename,fsn=None,fileend=None,dirs=[]):
    """Reads header data from measurement files
    
    Inputs:
        filename: the beginning of the filename, or the whole filename
        fsn: the file sequence number or None if the whole filenam was supplied
            in filename. It can be a list as well.
        fileend: the end of the file. If it ends with .gz, then the file is
            treated as a gzip archive.
        dirs [optional]: a list of directories to try
        
    Output:
        A list of header dictionaries. An empty list if no headers were read.
        
    Examples:
        read header data from 'ORG000123.DAT':
        
        header=readheader('ORG',123,'.DAT')
        
        or
        
        header=readheader('ORG00123.DAT')
    """
    jusifaHC=12396.4
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if fsn is None:
        names=[filename]
    else:
        if type(fsn)==types.ListType:
            names=['%s%05d%s' % (filename,x,fileend ) for x in fsn]
        else:
            names=['%s%05d%s' % (filename,fsn,fileend)]
    headers=[]
    for name in names:
        filefound=False
        for d in dirs:
            try:
                name1='%s%s%s' % (d,os.sep,name)
                header={};
                if name1.upper()[-3:]=='.GZ':
                    fid=gzip.GzipFile(name1,'rt');
                else:
                    fid=open(name1,'rt');
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
                header['Energy']=jusifaHC/float(string.strip(lines[43]))
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
                filefound=True
                break # we have already found the file, do not search for it in other directories
            except IOError:
                print 'Cannot find file %s.' %name1
                pass #continue with the next directory
    return headers
def read2dB1data(filename,files=None,fileend=None,dirs=[]):
    """Read 2D measurement files, along with their header data

    Inputs:
        filename: the beginning of the filename, or the whole filename
        fsn: the file sequence number or None if the whole filenam was supplied
            in filename. It is possible to give a list of fsns here.
        fileend: the end of the file.
        dirs [optional]: a list of directories to try
        
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
    def readgabrieldata(filename,dirs):
        for d in dirs:
            try:
                filename1='%s%s%s' %(d,os.sep,filename)
                if filename1.upper()[-3:]=='.GZ':
                    fid=gzip.GzipFile(filename1,'rt')
                else:
                    fid=open(filename1,'rt')
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
                pass
        print 'Cannot find file %s. Tried directories:' % filename,dirs
        return None
    def readpilatus300kdata(filename,dirs):
        for d in dirs:
            try:
                filename1='%s%s%s' % (d,os.sep,filename)
                fid=open(filename1,'rb');
                datastr=fid.read();
                fid.close();
                data=pylab.fromstring(datastr[4096:],'uint32').reshape((619,487)).astype('double')
                return data;
            except IOError:
                pass
        print 'Cannot find file %s. Make sure the path is correct.' % filename
        return None
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if fileend is None:
        fileend=filename[string.rfind(filename,'.'):]
    if (files is not None) and (type(files)!=types.ListType):
        files=[files];
    if fileend.upper()=='.TIF' or fileend.upper()=='.TIFF': # pilatus300k mode
        filebegin=filename[:string.rfind(filename,'.')]
        if files is None:
            header=readheader(filebegin+'.header',dirs=dirs)
            data=readpilatus300kdata(filename,dirs=dirs)
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
                tmp1=readheader('%s%05d%s' %(filename,fsn,'.header'),dirs=dirs)
                tmp2=readpilatus300kdata('%s%05d%s'%(filename,fsn,fileend),dirs=dirs)
                if (len(tmp1)>0) and (tmp2 is not None):
                    tmp1=tmp1[0]
                    tmp1['Detector']='Pilatus300k'
                    header.append(tmp1)
                    data.append(tmp2)
            return data,header
    else: # Gabriel mode, if fileend is neither TIF, nor TIFF, case insensitive
        if files is None: # read only 1 file
            header=readheader(filename,dirs=dirs);
            data=readgabrieldata(filename,dirs=dirs);
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
                tmp1=readheader('%s%05d%s' % (filename,fsn,fileend),dirs=dirs)
                tmp2=readgabrieldata('%s%05d%s' % (filename,fsn,fileend),dirs=dirs)
                if (len(tmp1)>0) and (tmp2 is not None):
                    tmp1=tmp1[0];
                    tmp1['Detector']='Gabriel'
                    data.append(tmp2);
                    header.append(tmp1);
            return data,header
def getsamplenames(filename,files,fileend,showtitles='Gabriel',dirs=[]):
    """Prints information on the measurement files
    
    Inputs:
        filename: the beginning of the filename, or the whole filename
        fsn: the file sequence number or None if the whole filenam was supplied
            in filename
        fileend: the end of the file.
        showtitles: if this is 'Gabriel', prints column headers for the gabriel
            detector. 'Pilatus300k' prints the appropriate headers for that
            detector. All other values suppress header printing.
        dirs [optional]: a list of directories to try
    
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
        d,h=read2dB1data(filename,i,fileend,dirs);
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
def read2dintfile(fsns,dirs=[]):
    """Read corrected intensity and error matrices
    
    Input:
        fsns: one or more fsn-s in a list
        
    Output:
        a list of 2d intensity matrices
        a list of error matrices
        a list of param dictionaries
        dirs [optional]: a list of directories to try
    
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
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if type(fsns)!=types.ListType: # if only one fsn was supplied, make it a list of one element
        fsns=[fsns]
    int2d=[]
    err2d=[]
    params=[]
    for fsn in fsns: # this also works if len(fsns)==1
        for d in dirs:
            try: # first try to load the mat file. This is the most effective way.
                tmp0=scipy.io.loadmat('%s%sint2dnorm%d.mat' % (d,os.sep,fsn))
                tmp=tmp0['Intensity'].copy()
                tmp1=tmp0['Error'].copy()
            except IOError: # if mat file is not found, try the ascii files
                print 'Cannot find file int2dnorm%d.mat: trying to read int2dnorm%d.dat(.gz|.zip) and err2dnorm%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
                tmp=read2dascii('%s%sint2dnorm%d.dat' % (d,os.sep,fsn));
                tmp1=read2dascii('%s%serr2dnorm%d.dat' % (d,os.sep,fsn));
            except TypeError: # if mat file was found but scipy.io.loadmat was unable to read it
                print "Malformed MAT file! Skipping."
                continue # try from another directory
            tmp2=readlogfile(fsn,d) # read the logfile
            if (tmp is not None) and (tmp1 is not None) and (tmp2 is not None): # if all of int,err and log is read successfully
                int2d.append(tmp)
                err2d.append(tmp1)
                params.append(tmp2[0])
                break # file was found, do not try to load it again from another directory
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
        dirs [optional]: a list of directories to try

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
        return {}
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
def write1dsasdict(data, filename):
    """Save 1D scattering data to file
    
    Inputs:
        data: 1D SAXS dictionary
        filename: filename
    """
    fid=open(filename,'wt');
    for i in range(len(data['q'])):
        fid.write('%e %e %e\n' % (data['q'][i],data['Intensity'][i],data['Error'][i]))
    fid.close();
def readintnorm(fsns, filetype='intnorm',dirs=[]):
    """Read intnorm*.dat files along with their headers
    
    Inputs:
        fsns: one or more fsn-s.
        filetype: prefix of the filename
        dirs [optional]: a list of directories to try
        
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
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    data=[];
    param=[];
    for fsn in fsns:
        for d in dirs:
            filename='%s%s%s%d.dat' % (d,os.sep,filetype, fsn)
            tmp=readintfile(filename)
            tmp2=readlogfile(fsn,d)
            if (tmp2!=[]) and (tmp!=[]):
                data.append(tmp);
                param.append(tmp2[0]);
                break # file was already found, do not try in another directory
    return data,param
def readbinned(fsn,dirs=[]):
    """Read intbinned*.dat files along with their headers.
    
    This is a shortcut to readintnorm(fsn,'intbinned',dirs)
    """
    return readintnorm(fsn,'intbinned',dirs);
def readlogfile(fsn,dirs=[]):
    """Read logfiles.
    
    Inputs:
        fsn: the file sequence number(s). It is possible to
            give a single value or a list
        dirs [optional]: a list of directories to try
            
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
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if type(fsn)!=types.ListType: # if fsn is not a list, convert it to a list
        fsn=[fsn];
    params=[]; #initially empty
    for f in fsn:
        for d in dirs:
            filename='%s%sintnorm%d.log' % (d,os.sep,f) #the name of the file
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
                break # file was already found, do not try in another directory
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
def readwaxscor(fsns,dirs=[]):
    """Read corrected waxs file
    
    Inputs:
        fsns: a range of fsns or a single fsn.
        dirs [optional]: a list of directories to try
        
    Output:
        a list of scattering data dictionaries (see readintfile())
    """
    if type(fsns)!=types.ListType:
        fsns=[fsns]
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    waxsdata=[];
    for fsn in fsns:
        for d in dirs:
            try:
                filename='%s%swaxs_%05d.cor' % (d,os.sep,fsn)
                tmp=pylab.load(filename)
                if tmp.shape[1]==3:
                    tmp1={'q':tmp[:,0],'Intensity':tmp[:,1],'Error':tmp[:,2]}
                waxsdata.append(tmp1)
                break # file was found, do not try in further directories
            except IOError:
                print '%s not found. Skipping it.' % filename
    return waxsdata
def readenergyfio(filename,files,fileend,dirs=[]):
    """Read abt_*.fio files.
    
    Inputs:
        filename: beginning of the file name, eg. 'abt_'
        files: a list or a single fsn number, eg. [1, 5, 12] or 3
        fileend: extension of a file, eg. '.fio'
        dirs [optional]: a list of directories to try
    
    Outputs: three lists:
        energies: the uncalibrated (=apparent) energies for each fsn.
        samples: the sample names for each fsn
        muds: the mu*d values for each fsn
    """
    if type(files)!=types.ListType:
        files=[files]
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    samples=[]
    energies=[]
    muds=[]
    for f in files:
        for d in dirs:
            mud=[];
            energy=[];
            fname='%s%s%s%05d%s' % (d,os.sep,filename,f,fileend)
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
                break #file found, do not try further directories
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
def readxanes(filebegin,files,fileend,energymeas,energycalib,dirs=[]):
    """Read energy scans from abt_*.fio files by readenergyfio() then
    put them on a correct energy scale.
    
    Inputs:
        filebegin: the beginning of the filename, like 'abt_'
        files: FSNs, like range(2,36)
        fileend: the end of the filename, like '.fio'
        energymeas: list of the measured energies
        energycalib: list of the true energies corresponding to the measured
            ones
        dirs [optional]: a list of directories to try
    
    Output:
        a list of mud dictionaries. Each dict will have the following items:
            'Energy', 'Mud', 'Title', 'scan'. The first three are
            self-describing. The last will be the FSN.
    """
    muds=[];
    if type(files)!=types.ListType:
        files=[files]

    for f in files:
        energy,sample,mud=readenergyfio(filebegin,f,fileend,dirs)
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
#fitting
def intintensity(data,alpha,alphaerr,qmin=-pylab.inf,qmax=pylab.inf,m=0):
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
    ret2=pylab.trapz(data['Intensity']*(data['q']**(-alpha)),data['q'])
    dret2=errtrapz(data['q'],data['Error']*(data['q']**(-alpha)))
    ret3=q1*data['Intensity'][data['q']==q1][0]*q1**(-alpha)
    dret3=q1*data['Error'][data['q']==q1][0]*q1**(-alpha)
    
    #print ret1, "+/-",dret1
    #print ret2, "+/-",dret2
    #print ret3, "+/-",dret3
    
    return ret1+ret2+ret3,pylab.sqrt(dret1**2+dret2**2+dret3**2)
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
def findpeak(xdata,ydata,prompt=None,mode='Lorentz',scaling='lin',blind=False):
    """GUI tool for locating peaks by zooming on them
    
    Inputs:
        xdata: x dataset
        ydata: y dataset
        prompt: prompt to display as a title
        mode: 'Lorentz' or 'Gauss'
        scaling: scaling of the y axis. 'lin' or 'log' 
        blind: do everything blindly (no user interaction)
        
    Outputs:
        the peak position
        
    Usage:
        Zoom to the desired peak then press ENTER on the figure.
    """
    xdata=xdata.flatten()
    ydata=ydata.flatten()
    if not blind:
        if scaling=='log':
            pylab.semilogy(xdata,ydata,'b.')
        else:
            pylab.plot(xdata,ydata,'b.')
        if prompt is None:
            prompt='Please zoom to the peak you want to select, then press ENTER'
        pylab.title(prompt)
        pylab.gcf().show()
        print(prompt)
        while (pylab.waitforbuttonpress() is not True):
            pass
        a=pylab.axis()
        indices=((xdata<=a[1])&(xdata>=a[0]))&((ydata<=a[3])&(ydata>=a[2]))
        x1=xdata[indices]
        y1=ydata[indices]
    else:
        x1=xdata
        y1=ydata
    def gausscostfun(p,x,y):  #p: A,sigma,x0,y0
        tmp= y-p[3]-p[0]/(pylab.sqrt(2*pylab.pi)*p[1])*pylab.exp(-(x-p[2])**2/(2*p[1]**2))
        return tmp
    def lorentzcostfun(p,x,y):
        tmp=y-p[3]-p[0]*lorentzian(p[2],p[1],x)
        return tmp
    if mode=='Gauss':
        sigma0=0.25*(x1[-1]-x1[0])
        p0=((y1.max()-y1.min())/(1/pylab.sqrt(2*pylab.pi*sigma0**2)),
            sigma0,
            0.5*(x1[-1]+x1[0]),
            y1.min())
        p1,ier=scipy.optimize.leastsq(gausscostfun,p0,args=(x1,y1),maxfev=10000)
        if not blind:
            if scaling=='log':
                pylab.semilogy(x1,p1[3]+p1[0]/(pylab.sqrt(2*pylab.pi)*p1[1])*pylab.exp(-(x1-p1[2])**2/(2*p1[1]**2)),'r-')
            else:
                pylab.plot(x1,p1[3]+p1[0]/(pylab.sqrt(2*pylab.pi)*p1[1])*pylab.exp(-(x1-p1[2])**2/(2*p1[1]**2)),'r-')
    elif mode=='Lorentz':
        sigma0=0.25*(x1[-1]-x1[0])
        p0=((y1.max()-y1.min())/(1/sigma0),
            sigma0,
            0.5*(x1[-1]+x1[0]),
            y1.min())
        p1,ier=scipy.optimize.leastsq(lorentzcostfun,p0,args=(x1,y1),maxfev=10000)
        if not blind:
            if scaling=='log':
                pylab.semilogy(x1,p1[3]+p1[0]*lorentzian(p1[2],p1[1],x1),'r-')
            else:
                pylab.plot(x1,p1[3]+p1[0]*lorentzian(p1[2],p1[1],x1),'r-')
    else:
        raise ValueError('Only Gauss and Lorentz modes are supported in findpeak()')
    if not blind:
        pylab.gcf().show()
    return p1[2]
def trimq(data,qmin=-pylab.inf,qmax=pylab.inf):
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
           'Error':pylab.sqrt(data['Error']**2+bgerror**2)};
def shullroess(data,qmin=-pylab.inf,qmax=pylab.inf,gui=False):
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
        Iexp=pylab.array(data1['Intensity'])
        qexp=pylab.array(data1['q'])
        errexp=pylab.array(data1['Error'])
        print "---Shull-Roess-fitting-with-qmin:-%lf-and-qmax:-%lf----" % (qexp.min(),qexp.max())
        logIexp=pylab.log(Iexp)
        errlogIexp=errexp/Iexp
    
        r0s=pylab.linspace(1,2*pylab.pi/qexp.min(),200)
        chi2=pylab.zeros(r0s.shape)
        for i in range(len(r0s)): # calculate the quality of the line for each r0.
            xdata=pylab.log(qexp**2+3/r0s[i]**2)
            a,b,aerr,berr=linfit(xdata,logIexp,errlogIexp)
            chi2[i]=pylab.sum(((xdata*a+b)-logIexp)**2)
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
    
        xdata=pylab.log(qexp**2+3/r0s[bestindex]**2)
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
        pylab.plot(r0s,maxwellian(n,r0s[bestindex],r0s))
        print "First approximation:"
        print "r0: ",r0s[bestindex]
        print "n: ",n
        print "K: ",b
        # do a proper least squares fitting
        def fitfun(p,x,y,err): # p: K,n,r0
            return (y-pylab.exp(p[0])*(x**2+3/p[2]**2)**(-(p[1]+4)/2.0))/err
        res=scipy.optimize.leastsq(fitfun,pylab.array([b,n,r0s[bestindex]]), 
                                    args=(qexp,Iexp,errexp),maxfev=1000,full_output=1)
        K,n,R0=res[0]
        print "After lsq fit:"
        print "r0: ",R0
        print "n: ",n
        print "K: ",K
        print "Covariance matrix:",res[1]
        dR0=pylab.sqrt(res[1][2][2])
        dn=pylab.sqrt(res[1][1][1])
        # plot the measured and the fitted curves
        pylab.axes(ax4)
        pylab.title('After LSQ fit')
        pylab.xlabel('q (1/%c)'%197)
        pylab.ylabel('Intensity')
        pylab.plot(pylab.log(qexp**2+3/R0**2),-(n+4)/2.0*pylab.log(qexp**2+3/R0**2)+K,label='Fitted')
        pylab.plot(pylab.log(qexp**2+3/R0**2),logIexp,'.',label='Measured')
        pylab.legend()
        # plot the new maxwellian
        pylab.axes(ax3)
        pylab.plot(r0s,maxwellian(n,R0,r0s))
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
def tweakfit(xdata,ydata,modelfun,fitparams):
    """"Fit" an arbitrary model function on the given dataset.
    
    Inputs:
        xdata: vector of abscissa
        ydata: vector of ordinate
        modelfun: model function. Should be of form fun(x,p1,p2,p3,...,pN)
        fitparams: list of parameter descriptions. Each element of this list
            should be a dictionary with the following fields:
                'Label': the short description of the parameter
                'Min': minimal value of the parameter
                'Max': largest possible value of the parameter
                'Val': default (starting) value of the parameter
                'mode': 'lin' or 'log'
    
    Outputs:
        None. This function leaves a window open for further user interactions.
        
    Notes:
        This opens a plot window. On the left sliders will appear which can
        be used to set the values of various parameters. On the right the
        dataset and the fitted function will be plotted.
        
        Please note that this is only a visual trick and a tool to help you
        understand how things work with your model. However, do not use the
        resulting parameters as if they were made by proper least-squares
        fitting. Once again: this is NOT a fitting routine in the correct
        scientific sense.
    """
    def redraw(keepzoom=True):
        if keepzoom:
            ax=pylab.gca().axis()
        pylab.cla()
        pylab.loglog(xdata,ydata,'.',color='blue')
        pylab.loglog(xdata,modelfun(xdata,*(pylab.gcf().params)),color='red')
        pylab.draw()
        if keepzoom:
            pylab.gca().axis(ax)
    fig=pylab.figure()
    ax=[]
    sl=[]
    fig.params=[]
    for i in range(len(fitparams)):
        ax.append(pylab.axes((0.1,0.1+i*0.8/len(fitparams),0.3,0.75/len(fitparams))))
        if fitparams[i]['mode']=='lin':
            sl.append(matplotlib.widgets.Slider(ax[-1],fitparams[i]['Label'],fitparams[i]['Min'],fitparams[i]['Max'],fitparams[i]['Val']))
        elif fitparams[i]['mode']=='log':
            sl.append(matplotlib.widgets.Slider(ax[-1],fitparams[i]['Label'],pylab.log10(fitparams[i]['Min']),pylab.log10(fitparams[i]['Max']),pylab.log10(fitparams[i]['Val'])))
        else:
            raise ValueError('Invalid mode %s in fitparams' % fitparams[i]['mode']);
        fig.params.append(fitparams[i]['Val'])
        def setfun(val,parnum=i,sl=sl[-1],mode=fitparams[i]['mode']):
            if mode=='lin':
                pylab.gcf().params[parnum]=sl.val;
            elif mode=='log':
                pylab.gcf().params[parnum]=pow(10,sl.val);
            else:
                pass
            redraw()
        sl[-1].on_changed(setfun)
    pylab.axes((0.5,0.1,0.4,0.8))
    redraw(False)
def guiniercrosssectionfit(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
    """Do a cross-section Guinier fit on the dataset.
    
    Inputs:
        data: 1D scattering data dictionary
        qmin: lowest q-value to take into account. Default is -infinity
        qmax: highest q-value to take into account. Default is infinity
        testimage: if a test image is desired. Default is false.
    
    Outputs:
        the Guinier radius (radius of gyration) of the cross-section
        the prefactor
        the calculated error of Rg
        the calculated error of the prefactor
    """
    data1=trimq(data,qmin,qmax)
    x1=data1['q']**2;
    err1=pylab.absolute(data1['Error']/data1['Intensity']*data1['q'])
    y1=pylab.log(data1['Intensity'])*data1['q']
    Rgcs,Gcs,dRgcs,dGcs=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data1['q']**2,pylab.log(data1['Intensity'])*data1['q'],'.')
        pylab.plot(data1['q']**2,Rgcs*data1['q']**2+Gcs,'-',color='red');
        pylab.xlabel('$q^2$ (1/%c$^2$)' % 197)
        pylab.ylabel('$q\ln I$')
    return pylab.sqrt(-Rgcs*2),Gcs,1/pylab.sqrt(-Rgcs)*dRgcs,dGcs
def guinierthicknessfit(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
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
    err1=pylab.absolute(data1['Error']/data1['Intensity']*data1['q']**2)
    y1=pylab.log(data1['Intensity'])*data1['q']**2
    Rgt,Gt,dRgt,dGt=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data1['q']**2,pylab.log(data1['Intensity'])*data1['q']**2,'.')
        pylab.plot(data1['q']**2,Rgt*data1['q']**2+Gt,'-',color='red');
        pylab.xlabel('$q^2$ (1/%c$^2$)' % 197)
        pylab.ylabel('$q^2\ln I$')
    return pylab.sqrt(-Rgt),Gt,0.5/pylab.sqrt(-Rgt)*dRgt,dGt
def guinierfit(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
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
    err1=pylab.absolute(data1['Error']/data1['Intensity']);
    y1=pylab.log(data1['Intensity']);
    Rg,G,dRg,dG=linfit(x1,y1,err1)
    if testimage:
        pylab.plot(data1['q']**2,pylab.log(data1['Intensity']),'.');
        pylab.plot(data1['q']**2,Rg*data1['q']**2+G,'-',color='red');
        pylab.xlabel('$q^2$ (1/%c$^2$)' % 197)
        pylab.ylabel('ln I');
    return pylab.sqrt(-Rg*3),G,1.5/pylab.sqrt(-Rg*3)*dRg,dG
def porodfit(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
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
def powerfit(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
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
    x1=pylab.log(data1['q']);
    err1=pylab.absolute(data1['Error']/data1['Intensity']);
    y1=pylab.log(data1['Intensity']);
    a,b,aerr,berr=linfit(x1,y1)
    if testimage:
        pylab.loglog(data1['q'],data1['Intensity'],'.');
        pylab.loglog(data1['q'],pylab.exp(b)*pow(data1['q'],a),'-',color='red');
        pylab.xlabel('$q$ (1/%c)' % 197)
        pylab.ylabel('I');
    return a,b,aerr,berr
def powerfitwithbackground(data,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
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
    res=scipy.optimize.leastsq(costfunc,pylab.array([Ainit,Binit,Cinit]),args=(x1,y1,err1),full_output=1)
    if testimage:
        pylab.loglog(data1['q'],data1['Intensity'],'.');
        pylab.loglog(data1['q'],res[0][1]*pow(data1['q'],res[0][0])+res[0][2],'-',color='red');
        pylab.xlabel('$q$ (1/%c)' % 197)
        pylab.ylabel('I');
    return res[0][0],res[0][1],res[0][2],pylab.sqrt(res[1][0][0]),pylab.sqrt(res[1][1][1]),pylab.sqrt(res[1][2][2])    
def unifiedfit(data,B,G,Rg,P,qmin=-pylab.inf,qmax=pylab.inf,maxiter=1000):
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
    res=scipy.optimize.leastsq(fitfun,pylab.array([B,G,Rg,P]),args=(data['q'],data['Intensity'],data['Error']),full_output=1)
    return res[0][0],res[0][1],res[0][2],res[0][3],pylab.sqrt(res[1][0][0]),pylab.sqrt(res[1][1][1]),pylab.sqrt(res[1][2][2]),pylab.sqrt(res[1][3][3])
def fitspheredistribution(data,distfun,R,params,qmin=-pylab.inf,qmax=pylab.inf,testimage=False):
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
    tsI=pylab.zeros((len(q),len(R)))
    for i in range(len(R)):
        tsI[:,i]=fsphere(q,R[i])
    R.reshape((len(R),1))
    def fitfun(params,R,q,I,Err,dist=distfun,tsI=tsI):
        return (params[-1]*pylab.dot(tsI,dist(R,*(params[:-1])))-I)/Err
    res=scipy.optimize.leastsq(fitfun,params1,args=(R,q,Int,Err),full_output=1)
    print "Fitted values:",res[0]
    print "Covariance matrix:",res[1]
    if testimage:
        pylab.semilogy(data['q'],data['Intensity'],'.');
        tsIfull=pylab.zeros((len(data['q']),len(R)))
        for i in range(len(R)):
            tsIfull[:,i]=fsphere(data['q'],R[i])
        print data['q'].shape
        print pylab.dot(tsIfull,distfun(R,*(res[0][:-1]))).shape
        pylab.semilogy(data['q'],res[0][-1]*pylab.dot(tsIfull,distfun(R,*(res[0][:-1]))),'-',color='red');
        pylab.xlabel('$q$ (1/%c)' % 197)
        pylab.ylabel('I');
    return res[0]
def lognormdistrib(x,mu,sigma):
    """Evaluate the PDF of the log-normal distribution
    
    Inputs:
        x: the points in which the values should be evaluated
        mu: parameter mu
        sigma: parameter sigma
    
    Outputs:
        y: 1/(x*sigma*sqrt(2*pi))*exp(-(log(x)-mu)^2/(2*sigma^2))
    """
    return 1/(x*sigma*pylab.sqrt(2*pylab.pi))*pylab.exp(-(pylab.log(x)-mu)**2/(2*sigma**2))
#1D data treatment
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
def directdesmear(data,smoothing,params):
    """Desmear the scattering data according to the direct desmearing
    algorithm by Singh, Ghosh and Shannon
    
    Inputs:
        data: measured intensity vector of arbitrary length (numpy array)
        smoothing: smoothing parameter for scipy.optimize.splrep. A scalar
            number. If not exactly known, a dictionary may be supplied with
            the following fields:
                low: lower threshold
                high: upper threshold
                val: initial value
                mode: 'lin' or 'log'
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
                
    Outputs: (pixels,desmeared,smoothed,mat,params,smoothing)
        pixels: the pixel coordinates for the resulting curves
        desmeared: the desmeared curve
        smoothed: the smoothed curve
        mat: the desmearing matrix
        params: the desmearing parameters
        smoothing: smoothing parameter
    """
    #default values
    dparams={'pixelmin':-pylab.inf,'pixelmax':pylab.inf,
             'beamnumh':1024,'beamnumv':0}
    dparams.update(params)
    params=dparams
    
    # calculate the matrix
    if params.has_key('matrix') and type(params['matrix'])==pylab.ndarray:
        A=params['matrix']
    else:
        A=smearingmatrix(params['pixelmin'],params['pixelmax'],
                         params['beamcenter'],params['pixelsize'],
                         params['lengthbaseh'],params['lengthtoph'],
                         params['lengthbasev'],params['lengthtopv'],
                         params['beamnumh'],params['beamnumv'])
        params['matrix']=A
    #x coordinates in pixels
    pixels=pylab.arange(len(data))
    def smooth_and_desmear(pixels,data,params,smoothing):
        # smoothing the dataset. Errors of the data are sqrt(data), weight will be therefore 1/data
        tck=scipy.interpolate.splrep(pixels,data,s=smoothing)
        data1=scipy.interpolate.splev(pixels,tck)
        indices=(pixels<=params['pixelmax']) & (pixels>=params['pixelmin'])
        data1=data1[indices]
        pixels=pixels[indices]
        print data1.shape
        print params['matrix'].shape
        ret=(pixels,pylab.solve(params['matrix'],data1.reshape(len(data1),1)),
             data1,params['matrix'],params,smoothing)
        return ret
    if type(smoothing)!=type({}):
        res=smooth_and_desmear(pixels,data,params,smoothing)
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
                                         pylab.log(smoothing['low']),
                                         pylab.log(smoothing['high']),
                                         pylab.log(smoothing['val']))
        elif smoothing['mode']=='lin':
            sl=matplotlib.widgets.Slider(axsl,'Smoothing',
                                         smoothing['low'],
                                         smoothing['high'],
                                         smoothing['val'])
        else:
            raise ValueError('Invalid value for smoothingmode: %s',
                             smoothing['mode'])
        def sliderfun(a=None,sl=sl,ax=ax,mode=smoothing['mode'],x=pixels,
                      y=data,p=params):
            print "sliderfun:",a,sl.val
            if mode=='lin':
                sm=sl.val
            else:
                sm=pylab.exp(sl.val)
            [x1,y1,ysm,A,par,sm]=smooth_and_desmear(x,y,p,sm)
            a=ax.axis()
            ax.cla()
            ax.semilogy(x,y,'.',label='Original')
            ax.semilogy(x1,ysm,'-',label='Smoothed (%lg)'%sm)
            ax.semilogy(x1,y1,'-',label='Desmeared')
            ax.legend(loc='best')
            ax.axis(a)
            pylab.gcf().show()
        sl.on_changed(sliderfun)
        [x1,y1,ysm,A,par,sm]=smooth_and_desmear(pixels,data,params,smoothing['val'])
        ax.semilogy(pixels,data,'.',label='Original')
        ax.semilogy(x1,ysm,'-',label='Smoothed (%lg)'%smoothing['val'])
        ax.semilogy(x1,y1,'-',label='Desmeared')
        ax.legend(loc='best')
        while not f.donedesmear:
            pylab.waitforbuttonpress()
        if smoothing['mode']=='lin':
            sm=sl.val
        elif smoothing['mode']=='log':
            sm=pylab.exp(sl.val)
        else:
            raise ValueError('Invalid value for smoothingmode: %s',
                             smoothing['mode'])
        res=smooth_and_desmear(pixels,data,params,sm)
        return res        
def readasa(basename):
    """Load SAXS/WAXS measurement files from ASA *.INF, *.P00 and *.E00 files.
    
    Input:
        basename: the basename (without extension) of the files
    
    Output:
        An ASA dictionary of the following fields:
            position: the counts for each pixel (numpy array)
            energy: the energy spectrum (numpy array)
            params: parameter dictionary. It has the following fields:
                Month: The month of the measurement
                Day: The day of the measurement
                Year: The year of the measurement
                Hour: The hour of the measurement
                Minute: The minute of the measurement
                Second: The second of the measurement
                Title: The title. If the user has written something to the
                    first line of the .INF file, it will be regarded as the
                    title. Otherwise the basename will be picked for this
                    field.
                Basename: The base name of the files (without the extension)
                Energywindow_Low: the lower value of the energy window
                Energywindow_High: the higher value of the energy window
                Stopcondition: stop condition in a string
                Realtime: real time in seconds
                Livetime: live time in seconds
            pixels: the pixel numbers.
    """
    try:
        p00=pylab.loadtxt('%s.P00' % basename)
    except IOError:
        try:
            p00=pylab.loadtxt('%s.p00' % basename)
        except:
            raise IOError('Cannot find %s.p00, neither %s.P00.' % (basename,basename))
    if p00 is not None:
        p00=p00[1:] # cut the leading -1
    try:
        e00=pylab.loadtxt('%s.E00' % basename)
    except IOError:
        try:
            e00=pylab.loadtxt('%s.e00' % basename)
        except:
            e00=None
    if e00 is not None:
        e00=e00[1:] # cut the leading -1
    try:
        inffile=open('%s.inf' % basename)
    except IOError:
        try:
            inffile=open('%s.Inf' % basename)
        except IOError:
            try:
                inffile=open('%s.INF' % basename)
            except:
                inffile=None
                params=None
    if inffile is not None:
        params={}
        l=inffile.readlines()
        def getdate(str):
            try:
                month=int(str.split()[0].split('-')[0])
                day=int(str.split()[0].split('-')[1])
                year=int(str.split()[0].split('-')[2])
                hour=int(str.split()[1].split(':')[0])
                minute=int(str.split()[1].split(':')[1])
                second=int(str.split()[1].split(':')[2])
            except:
                return None
            return {'Month':month,'Day':day,'Year':year,'Hour':hour,'Minute':minute,'Second':second}
        if getdate(l[0]) is None:
            params['Title']=l[0].strip()
            offset=1
        else:
            params['Title']=basename
            offset=0
        d=getdate(l[offset])
        params.update(d)
        for line in l:
            if line.strip().startswith('PSD1 Lower Limit'):
                params['Energywindow_Low']=float(line.strip().split(':')[1].replace(',','.'))
            elif line.strip().startswith('PSD1 Upper Limit'):
                params['Energywindow_High']=float(line.strip().split(':')[1].replace(',','.'))
            elif line.strip().startswith('Realtime'):
                params['Realtime']=float(line.strip().split(':')[1].split()[0].replace(',','.').replace('\xa0',''))
            elif line.strip().startswith('Lifetime'):
                params['Livetime']=float(line.strip().split(':')[1].split()[0].replace(',','.').replace('\xa0',''))
            elif line.strip().startswith('Lower Limit'):
                params['Energywindow_Low']=float(line.strip().split(':')[1].replace(',','.'))
            elif line.strip().startswith('Upper Limit'):
                params['Energywindow_High']=float(line.strip().split(':')[1].replace(',','.'))
            elif line.strip().startswith('Stop Condition'):
                params['Stopcondition']=line.strip().split(':')[1].strip().replace(',','.')
        params['basename']=basename.split(os.sep)[-1]
    return {'position':p00,'energy':e00,'params':params,'pixels':pylab.arange(len(p00))}
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
        tmp=findpeak(xdata,ydata,('Zoom to peak %d and press ENTER' % p),peakmode,scaling='log')
        pcoord.append(tmp)
    pcoord=pylab.array(pcoord)
    n=pylab.array(peaks)
    a=(n*wavelength)/(2*d)
    x=2*a*pylab.sqrt(1-a**2)/(1-2*a**2)
    LperH,xcent,LperHerr,xcenterr=linfit(x,pcoord)
    print 'pixelsize/dist:',1/LperH,'+/-',LperHerr/LperH**2
    print 'beam position:',xcent,'+/-',xcenterr
    b=(pylab.array(xdata)-xcent)/LperH
    if returnq:
        return 4*pylab.pi*pylab.sqrt(0.5*(b**2+1-pylab.sqrt(b**2+1))/(b**2+1))/wavelength
    else:
        return 1/LperH,xcent,LperHerr/LperH**2,xcenterr
def tripcalib(xdata,ydata,peakmode='Lorentz',wavelength=1.54,qvals=2*pylab.pi*pylab.array([0.21739,0.25641,0.27027]),returnq=True):
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
        tmp=findpeak(xdata,ydata,
                     ('Zoom to peak %d (q = %f) and press ENTER' % (p,qvals[p])),
                     peakmode,scaling='lin')
        pcoord.append(tmp)
    pcoord=pylab.array(pcoord)
    n=pylab.array(peaks)
    a,b,aerr,berr=linfit(pcoord,qvals)
    if returnq:
        return a*xdata+b
    else:
        return a,b,aerr,berr
#Macros for data processing
def addfsns(fileprefix,fsns,fileend,fieldinheader=None,valueoffield=None,dirs=[]):
    """
    """
    data,header=read2dB1data(fileprefix,fsns,fileend,dirs=dirs)
    
    dataout=None
    headerout=[]
    summed=[]
    for k in range(len(header)):
        h=header[k]
        if (abs(h['Energy']-header[0]['Energy'])<0.5) and \
            (h['Dist']==header[0]['Dist']) and \
            (h['Title']==header[0]['Title']):
                if(h['rot1']!=header[0]['rot1']) or  (h['rot2']!=header[0]['rot2']):
                    print "Warning! Rotation of sample in FSN %d (%s) is different from FSN %d (%s)." % (h['FSN'],h['Title'],header[0]['FSN'],header[0]['Title'])
                    shrubbery=raw_input('Do you still want to add the data? (y/n)   ')
                    if shrubbery.strip().upper()[0]!='Y':
                        return
                if(h['PosRef']!=header[0]['PosRef']):
                    print "Warning! Position of reference sample in FSN %d (%s) is different from FSN %d (%s)." % (h['FSN'],h['Title'],header[0]['FSN'],header[0]['Title'])
                    shrubbery=raw_input('Do you still want to add the data? (y/n)   ')
                    if shrubbery.strip().upper()[0]!='Y':
                        return
                if dataout is None:
                    dataout=data[k].copy()
                else:
                    dataout=dataout+data[k]
                headerout.append(h)
                summed.append(h['FSN'])
    return dataout,headerout,summed
def makesensitivity(fsn1,fsn2,fsnend,fsnDC,energymeas,energycalib,energyfluorescence,origx,origy):
    """Create matrix for detector sensitivity correction
    
    Inputs:
        fsn1: file sequence number for first measurement of foil at energy E1
        fsn2: file sequence number for first measurement of foil at energy E2
        fsnend: last FSN in the sensitivity measurement sequence
        fsnDC: a single number or a list of FSN-s for dark current
            measurements
        energymeas: apparent energies for energy calibration
        energycalib: calibrated energies for energy calibration
        energyfluorescence: energy of the fluorescence
        origx, origy: the centers of the beamstop.
    
    Outputs: sens,errorsens
        sens: the sensitivity matrix of the 2D detector, by which all
            measured data should be divided pointwise. The matrix is
            normalized to 1 on the average.
        errorsens: the error of the sensitivity matrix.
    """
    global _B1config
    
    pixelsize=_B1config['pixelsize']
    
    fsns=range(min(fsn1,fsn2),fsnend+1) # the fsn range of the sensitivity measurement
    
    #read in every measurement file
    data,header=read2dB1data(_B1config['2dfileprefix'],fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    E1header=[h for h in header if h['FSN']==fsn1 ][0] # the header of the first measurement at E1
    E1fsns=[h['FSN'] for h in header if (abs(h['Energy']-E1header['Energy'])<0.5) and (h['Title']==E1header['Title'])]
    E2header=[h for h in header if h['FSN']==fsn2 ][0] # the header of the first measurement at E2
    E2fsns=[h['FSN'] for h in header if (abs(h['Energy']-E2header['Energy'])<0.5) and (h['Title']==E2header['Title'])]
    dataE1,headerE1=read2dB1data(_B1config['2dfileprefix'],E1fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    dataE2,headerE2=read2dB1data(_B1config['2dfileprefix'],E2fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])

    ebE1fsns=unique([h['FSNempty'] for h in headerE1])
    ebE2fsns=unique([h['FSNempty'] for h in headerE2])
    dataebE1,headerebE1=read2dB1data(_B1config['2dfileprefix'],ebE1fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    dataebE2,headerebE2=read2dB1data(_B1config['2dfileprefix'],ebE2fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    datadc,headerdc=read2dB1data(_B1config['2dfileprefix'],fsnDC,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    #subtract background, and correct for transmission (and sensitivity :-): to override this correction, ones() and zeros() are given)
    A1,errA1=subdc(dataE1,headerE1,datadc,headerdc,pylab.ones(dataE1[0].shape),pylab.zeros(dataE1[0].shape))
    A2,errA2=subdc(dataE2,headerE2,datadc,headerdc,pylab.ones(dataE2[0].shape),pylab.zeros(dataE2[0].shape))
    eb1,erreb1=subdc(dataebE1,headerebE1,datadc,headerdc,pylab.ones(dataebE1[0].shape),pylab.zeros(dataebE1[0].shape))
    eb2,erreb2=subdc(dataebE2,headerebE2,datadc,headerdc,pylab.ones(dataebE2[0].shape),pylab.zeros(dataebE2[0].shape))
    
    #theta for pixels
    tth=pylab.arctan(calculateDmatrix(A1,pixelsize,origx,origy)/headerE1[0]['Dist'])
    #transmissions below and above the edge
    transm1=pylab.array([h['Transm'] for h in headerE1])
    transm2=pylab.array([h['Transm'] for h in headerE2])
    
    #angle-dependent absorption
    transmcorr1=absorptionangledependenttth(tth,transm1.mean())/transm1.mean()
    transmcorr2=absorptionangledependenttth(tth,transm2.mean())/transm2.mean()
    
    #subtract empty beam
    B1=(A1-eb1)*transmcorr1
    B2=(A2-eb2)*transmcorr2
    errB1=pylab.sqrt(errA1**2+erreb1**2)*transmcorr1
    errB2=pylab.sqrt(errA2**2+erreb2**2)*transmcorr2
    
    factor=1 #empirical compensation factor to rule out small-angle scattering completely
    if (E1header['Energy']>E2header['Energy']):
        C=B1-factor*B2
        Cerr=pylab.sqrt(errB1**2+factor**2*errB2**2)
    else:
        C=B2-factor*B1
        Cerr=pylab.sqrt(errB2**2+factor**2*errB1**2)
    C=C*gasabsorptioncorrectiontheta(energyfluorescence,tth)
    print "Please mask erroneous areas!"
    mask = makemask(pylab.ones(C.shape),C)
    print sum(mask)
    print sum(1-mask)
    C=(mask)*C
    cc=C.mean()
    sens=C/cc
    errorsens=Cerr/cc
    #taking care of zeros
    sens[sens==0]=1
    pylab.imshow(sens)
    pylab.colorbar()
    pylab.axis('equal')
    pylab.gcf().show()
    return sens,errorsens
def B1normint1(fsn1,thicknesses,orifsn,fsndc,sens,errorsens,mask,energymeas,energycalib,distminus=0,detshift=0,orig=[122,123.5]):
    """Integrate, normalize, do absolute calibration... on a sequence
    
    Inputs:
        fsn1: a single number or a list of FSNs.
        thicknesses: either one thickness in cm or a dictionary,
            containing thickness values for each sample. For example, if
            you have measured My_Samplename_1, Al2O3, DPPC_CdS with
            thicknesses 1, 1.2 and 0.5 mm, respectively, then you should
            give: thicknesses={'My_Samplename_1':0.1,'Al2O3':0.12,
                'DPPC_CdS':0.05}
        orifsn: which element of the sequence should be used for
            determining the origin. 0 means empty beam...
            Or you can give a tuple or a list of two: in this case these
            will be the coordinates of the origin and no auto-searching
            will be performed.
        fsndc: one FSN or a list of FSNS corresponding to the empty beam
            measurements
        sens: sensitivity matrix
        errorsens: error of the sensitivity matrix
        mask: mask matrix
        energymeas: list of apparent energies, for energy calibration
        energycalib: list of theoretical energies, corresponding to the
            apparent energies.
        distminus: this will be subtracted from the sample-to-detector
            distance read from the measurement files, but only for
            samples, not for references
        detshift: this will be subtracted from the sample-to-detector
            distance read from all measurement files, including
            references!
        orig: first guess for the origin. A list of two.
        
    Outputs:
        qs: q-scales of intensities
        ints: intensities
        errs: errors
        header: header dictionaries
    """
    try:
        ni=float(thicknesses)
        print "Using thickness %f cm for all samples except references" % (ni)
    except:
        pass
        
    if (len(energycalib)!=len(energymeas)) or len(energycalib)<2:
        print "Stopping. Variables energycalib and energymeas should contain equal amount of\npoints and at least two points to be able to make the energy calibration."
        return [],[],[],[]
    qs,ints,errs,areas,As,Aerrs,header=B1integrate(fsn1,fsndc,sens,errorsens,orifsn,mask,energymeas,energycalib,distminus,detshift,orig,transm)
    #find the reference measurement
    referencenumber=None
    for k in range(len(ints)):
        if header[k]['Title']=='Reference_on_GC_holder_before_sample_sequence':
            referencenumber=k
    if k is None:
        print "No reference measurements found! At least one file with title\n 'Reference_on_GC_holder_before_sample_sequence' should be part of the sequence!"
        return [],[],[],[]
    GCdata=None
    for r in _B1config['refdata']:
        if abs(r['pos']-header[referencenumber]['PosRef'])<_B1config['refposprecision']:
            GCdata=pylab.loadtxt("%s%s%s" % (_B1config['calibdir'],os.sep,r['data']))
            refthick=r['thick']
    if GCdata is None:
        print "No calibration data exists with ref. position %.2f +/- %.2f." % (header[referencenumber]['PosRef'],_B1config['refposprecision'])
        return [],[],[],[]
    print "FSN %d: Using GLASSY CARBON REFERENCE with nominal thickness %.f micrometers." % (header[referencenumber]['FSN'],refthick*1e4)
    
    #re-integrate GC measurement to the same q-bins
    print "Re-integrating GC data to the same bins at which the reference is defined"
    qGC,intGC,errGC,AGC=radint(As[referencenumber],
                               Aerrs[referencenumber],
                               header[referencenumber]['EnergyCalibrated'],
                               header[referencenumber]['Dist'],
                               header[referencenumber]['PixelSize'],
                               header[referencenumber]['BeamPosX'],
                               header[referencenumber]['BeamPosY'],
                               1-mask,
                               GCdata[:,0])
    print "Re-integration done."
    GCdata=GCdata[Agc>=GCareathreshold,:]
    intGC=intGC[Agc>=GCareathreshold]
    errGC=intGC[Agc>=GCareathreshold]
    qGC=intGC[Agc>=GCareathreshold]    
    
    intGC=intGC/refthick
    errGC=errGC/refthick
    
    mult,errmult=multfactor(qGC,GCdata[:,1],GCdata[:,2],intGC,errGC)
    
    print "Factor for GC normalization: %.2g +/- %.2 %%" % (mult,errmult/mult*100)
    pylab.clf()
    pylab.plot(qGC,intGC*mult,'.',label='Your reference (reintegrated)')
    pylab.plot(GCdata[:,0],GCdata[:,1],'.',label='Calibrated reference')
    pylab.plot(qs[referencenumber],ints[referencenumber]*mult/refthick,'-',label='Your reference (saved)')
    pylab.xlabel(u'q (1/%c)' % 197)
    pylab.ylabel('Scattering cross-section (1/cm)')
    pylab.title('Reference FSN %d multiplied by %.2e, error percenteage %.2f' %(header[referencenumber]['FSN'],mult,(errmult/multfactor*100)))
    pause()
    
    #normalize all data to 1/cm
    for k in range(len(ints)):
        if k!=referencenumber:
            thick=None
            try:
                thick=thicknesses[header[k]['Title']]
            except:
                pass
            if thick==None:
                try:
                    thick=float(thicknesses)
                except:
                    print "Did not find thickness for sample %s. Stopping." % header[k]['Title']
                    return qs,ints,errs,header
            print "Using thickness %f cm for sample %s" % (thick,header[k]['Title'])
            errs[k]=pylab.sqrt((mult*errs[k])**2+(errmult*ints[k])**2)/thick
            ints[k]=mult*ints[k]/thick
            Aerrs[k]=pylab.sqrt((mult*Aerrs[k])**2+(errmult*As[k])**2)/thick
            As[k]=mult*As[k]/thick
            if ((header[k]['Current1']>header[referencenumber]['Current2']) and (k>referencenumber)) or \
                ((header[k]['Current2']<header[referencenumber]['Current1']) and (k<referencenumber)):
                header[k]['injectionGC']='y'
            else:
                header[k]['injectionGC']='n'
            writelogfile(header[k],[header[k]['BeamPosX'],header[k]['BeamPosY']],thick,fsndc,
                         header[k]['EnergyCalibrated'],header[k]['Dist'],
                         mult,errmult,header[k][referencenumber]['FSN'],
                         refthick,header[k]['injectionGC'],header[k]['injectionEB'],
                         header[k]['PixelSize'],mode=_B1config.detector)
            writeintfile(qs[k],ints[k],errs[k],header[k],areas[k])
            write2dintfile(As[k],Aerrs[k],header[k])
def B1integrate(fsn1,fsndc,sens,errorsens,orifsn,mask,energymeas,energycalib,distminus=0,detshift=0,orig=[122,123.5],transm=None):
    """Integrate a sequence
    
    Inputs:
        fsn1: range of fsns. The first should be the empty beam
            measurement.
        fsndc: one FSN or a list of FSNS corresponding to the empty beam
            measurements
        sens: sensitivity matrix
        errorsens: error of the sensitivity matrix
        orifsn: which element of the sequence should be used for
            determining the origin. 0 means empty beam...
            Or you can give a tuple or a list of two: in this case these
            will be the coordinates of the origin and no auto-searching
            will be performed.
        mask: mask matrix
        energymeas: list of apparent energies, for energy calibration
        energycalib: list of theoretical energies, corresponding to the
            apparent energies.
        distminus: this will be subtracted from the sample-to-detector
            distance read from the measurement files, but only for
            samples, not for references
        detshift: this will be subtracted from the sample-to-detector
            distance read from all measurement files, including
            references!
        orig: first guess for the origin. A list of two.
        transm: you can give this if you know the transmission of the
            sample from another measurement. Leave it None to use the
            measured transmission.
            
    Outputs: qs,ints,errs,Areas,As,Aerrs,header
        qs: q-range. Automatically determined from the mask, the energy
            and the sample-to-detector distance
        ints: intensities corresponding to the q values
        errs: calculated absolute errors
        Areas: effective areas during integration
        As: list of corrected 2D data
        Aerrs: list of corrected 2D errors
        header: header dictionaries
    """
    global _B1config
    distancetoreference=_B1config['distancetoreference'] #mm. The references are nearer to the detector than the samples
    pixelsize=_B1config['pixelsize']
    # subtract the background and the dark current, normalise by sensitivity and transmission
    Asub,errAsub,header,injectionEB = subtractbg(fsn1,fsndc,sens,errorsens,transm)
    print "B1integrate: doing energy calibration and correction for reference distance"
    for k in range(len(Asub)):
        if header[k]['Title']=='Reference_on_GC_holder_before_sample_sequence':
            header[k]['Dist']=header[k]['Dist']-distancetoreference-detshift
            print "Corrected sample-detector distance for fsn %d (ref. before)." % header[k]['FSN']
        elif header[k]['Title']=='Reference_on_GC_holder_after_sample_sequence':
            header[k]['Dist']=header[k]['Dist']-distancetoreference-detshift
            print "Corrected sample-detector distance for fsn %d (ref. after)." % header[k]['FSN']
        else:
            header[k]['Dist']=header[k]['Dist']-distminus-detshift
        header[k]['EnergyCalibrated']=energycalibration(energymeas,energycalib,header[k]['Energy'])
        print "Calibrated energy for FSN %d (%s): %f -> %f" %(header[k]['FSN'],header[k]['Title'],header[k]['Energy'],header[k]['EnergyCalibrated'])
        
    #finding beamcenter
    
    try:
        lenorifsn=len(orifsn)
        if lenorifsn==2:
            orig=orifsn
        else:
            print "Malformed orifsn parameter for B1integrate: ",orifsn
            raise ValueError("Malformed orifsn parameter for B1integrate()")
    except TypeError:
        print "Determining origin from file FSN %d %s" %(header[orifsn]['FSN'],header[orifsn]['Title'])
        orig=findbeam(data[orifsn],orig,mask)
        print "Determined origin to be %.2f %.2f." % (orig[0],orig[1])
        testorigin(data[orifsn],orig,mask)
        pause()
        
    qs=[]
    ints=[]
    errs=[]
    Areas=[]
    As=[]
    Aerrs=[]
    orig=[]
    print "Integrating data. Press Return after inspecting the images."
    for k in range(len(Asub)):
        header[k]['BeamPosX']=orig[0]
        header[k]['BeamPosY']=orig[1]
        header[k]['PixelSize']=pixelsize
        D=calculateDmatrix(mask,pixelsize,orig[0],orig[1])
        tth=pylab.arctan(D/header[k].Dist)
        spatialcorr=geomcorrectiontheta(tth,header[k]['Dist'])
        absanglecorr=absorptionangledependenttth(tth,header[k]['Transm'])
        gasabsorptioncorr=gasabsorptioncorrectiontheta(header[k]['Energycalibrated'],tth)
        As.append(Asub[k]*spatialcorr*absanglecorr*gasabsorptioncorr)
        Aerrs.append(errAsub[k]*spatialcorr*absanglecorr*gasabsorptioncorr)
        pylab.clf()
        plot2dmatrix(Asub[k],None,mask,header[k],blacknegative=True)
        pylab.gcf().suptitle('FSN %d (%s) Corrected, log scale\nBlack: nonpositives; Faded: masked pixels' % (header[k]['FSN'],header[k]['Title']))
        pylab.gcf().show()
        #now do the integration
        print "Now integrating..."
        spam=time.time()
        q,I,e,A=radint(As,Aerrs,header[l]['EnergyCalibrated'],header[l]['Dist'],
                       header[l]['PixelSize'],header[l]['BeamPosX'],
                       header[l]['BeamPosY'],1-mask)
        qs.append(q)
        ints.append(I)
        errs.append(e)
        Areas.append(A)
        print "...done. Integration took %f seconds" % (time.time()-spam)
        pause() # we put pause here, so while the user checks the 2d data, the integration is carried out.
        pylab.clf()
        pylab.subplot(121)
        pylab.cla()
        pylab.errorbar(qs[-1],ints[-1],errs[-1])
        pylab.axis('tight')
        pylab.xlabel(u'q (1/%c)' % 197)
        pylab.ylabel('Intensity (arb. units)')
        pylab.title('FSN %d' % (header[l]['FSN']))
        pylab.subplot(122)
        pylab.cla()
        pylab.plot(qs[-1],Areas[-1])
        pylab.xlabel(u'q (1/%c)' %197)
        pylab.ylabel('Effective area (pixels)')
        pylab.title(header[l]['Title'])
        pylab.gcf().show()
        pause()
    return qs,ints,errs,Areas,As,Aerrs
def geomcorrectiontheta(tth,dist):
    return dist**2/(pylab.cos(tth)**3)
def absorptionangledependenttth(tth,transm):
    mud=-pylab.log(transm);
    cor=pylab.ones(tth.shape)
    cor[tth!=0]=transm/((1/(1-1/pylab.cos(tth[tth!=0]))/mud)*(pylab.exp(-mud/pylab.cos(tth[tth!=0]))-pylab.exp(-mud)))
    return cor
def gasabsorptioncorrectiontheta(energycalibrated,tth):
    
    global _B1config
    print "GAS ABSORPTION CORRECTION USED!!"
    
    #components is a list of dictionaries, describing the absorbing
    # components of the X-ray flight path. Each dict should have the
    # following fields:
    #   'name': name of the component (just for reference)
    #   'thick': thickness in mm
    #   'data': data file to find the transmission data.
    components=[{'name':'detector gas area','thick':50,'data':'TransmissionAr910Torr1mm298K.dat'},
                {'name':'air gap','thick':50,'data':'TransmissionAir760Torr1mm298K.dat'},
                {'name':'detector window','thick':0.1,'data':'TransmissionBe1mm.dat'},
                {'name':'flight tube window','thick':0.15,'data':'TransmissionPolyimide1mm.dat'}]
    cor=pylab.ones(tth.shape)
    for c in components:
        c['travel']=c['thick']/pylab.cos(tth)
        spam=pylab.loadtxt("%s%s%s" % (_B1config['calibdir'],os.sep,c['data']))
        if energycalibrated<spam[:,0].min():
            tr=spam[0,1]
        elif energycalibrated>spam[:,0].max():
            tr=spam[0,-1]
        else:
            tr=pylab.interp(energycalibrated,spam[:,0],spam[:,1])
        c['mu']=-pylab.log(tr) # in 1/mm
        cor=cor/pylab.exp(-c['travel']*c['mu'])
    return cor
def subtractbg(fsn1,fsndc,sens,senserr,transm=None):
    """Subtract dark current and empty beam from the measurements and
    carry out corrections for detector sensitivity, dead time and beam
    flux (monitor counter). subdc() is called...
    
    Inputs:
        fsn1: list of file sequence numbers (or just one number)
        fsndc: FSN for the dark current measurements. Can be a single
            integer number or a list of numbers. In the latter case the
            DC data are summed up.
        sens: sensitivity matrix
        senserr: error of the sensitivity matrix
        transm: if given, disregard the measured transmission of the
            sample.
    
    Outputs: Asub,errAsub,header,injectionEB
        Asub: the corrected matrix
        errAsub: the error of the corrected matrix
        header: header data
        injectionEB: 'y' if an injection between the empty beam and
            sample measurement occured. 'n' otherwise
    """
    global _B1config
    datadc,headerdc=read2dB1data(_B1config['2dfileprefix'],fsndc,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    data,header=read2dB1data(_B1config['2dfileprefix'],fsn1,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    Asub=[]
    errAsub=[]
    headerout=[]
    injectionEB=[]
    
    for k in range(len(data)): # for each measurement
        # read int empty beam measurement file
        [databg,headerbg]=read2dB1data(_B1config['2dfileprefix'],header[k]['FSNempty'],_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
        if len(databg)==0:
            print 'Cannot find all empty beam measurements.\nWhere is the empty FSN %d belonging to FSN %d? Stopping.'% (header[k]['FSNempty'],header[k]['FSN'])
            return Asub,errAsub,headerout,injectionEB
        # subtract dark current and normalize by sensitivity and transmission (1 in case of empty beam)
        Abg,Abgerr=subdc(databg,headerbg,datadc,headerdc,sens,senserr)
        # subtract dark current from scattering patterns and normalize by sensitivity and transmission
        A2,A2err=subdc([data[k]],[header[k]],datadc,headerdc,sens,senserr,transm)
        # subtract background, but first check if an injection occurred
        if header[k]['Current1']>headerbg[0]['Current2']:
            print "Possibly an injection between sample and its background:"
            getsamplenames(_B1config['2dfileprefix'],header[k]['FSN'],_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
            getsamplenames(_B1config['2dfileprefix'],header[k]['FSNempty'],_B1config['2dfilepostfix'],showtitles='no',dirs=_B1config['measdatadir'])
            print "Current in DORIS at the end of empty beam measurement %.2f." % headerbg[0]['Current2']
            print "Current in DORIS at the beginning of sample measurement %.2f." % header[k]['Current1']
            injectionEB.append['y']
        else:
            injectionEB.append['n']
        Asub.append(A2-Abg) # they were already normalised by the transmission
        errAsub.append(pylab.sqrt(A2err**2+Abgerr**2))
        header[k]['FSNdc']=fsndc
        headerout.append(header[k])
    return Asub, errAsub, headerout, injectionEB
def subdc(data,header,datadc,headerdc,sens,senserr,transm=None):
    """Carry out dead-time corrections on 2D data, from the Gabriel detector.
    
    Inputs:
        data: list of 2d scattering images, to be added
        header: list of header data, to be added
        datadc: list of 2d scattering images for dark current
            measurement, to be added
        headerdc: list of header data for dark current measurement, to
            be added up.
        sens: sensitivity matrix
        senserr: error matrix of the sensitivity data
        
    Outputs:
        the normalized data and its error matrix
    """
    # summarize transmission, anode, monitor and measurement time data
    # for measurement files
    if transm is None:
        transm=pylab.array([h['Transm'] for h in header])
        transmave=transm.mean() # average transmission
        transmerr=transm.std() # standard deviation of the transmission
    else:
        transmave=transm
        transmerr=0

    an1=sum([h['Anode'] for h in header])
    mo1=sum([h['Monitor'] for h in header])
    meastime1=sum([h['MeasTime'] for h in header])
    
    print "FSN %d\tTitle %s\tEnergy %.1f\tDistance %d" % (header[0]['FSN'],header[0]['Title'],header[0]['Energy'],header[0]['Dist'])
    print "Average transmission %.4f +/- %.4f" % (transmave,transmerr)
    
    # summarize transmission, anode, monitor and measurement time data
    # for dark current files
    andc=sum([h['Anode'] for h in headerdc])
    modc=sum([h['Monitor'] for h in headerdc])
    meastimedc=sum([h['MeasTime'] for h in headerdc])
    
    # correct monitor counts with its dark current
    monitor1corrected=mo1-modc*meastime1/meastimedc
    
    # add up scattering patterns
    A=sum(data) # do not use pylab.sum()
    Adc=sum(datadc)
    
    #subtract the dark current from the scattering pattern
    sumA2=(A-Adc*meastime1/meastimedc).sum()
    # error of sumA2, not sum of error of A2.
    sumA2err=pylab.sqrt((A+(meastime1/meastimedc)**2*Adc).sum())
    
    anA2=an1-andc*meastime1/meastimedc;
    anA2err=pylab.sqrt(an1+(meastime1/meastimedc)**2*andc)
    
    # summarized scattering pattern, subtracted the dark current,
    # normalized by the monitor counter and the sensitivity
    A2=(A-Adc*meastime1/meastimedc)/sens/monitor1corrected
    
    print "Sum/Total of dark current: %.2f. Counts/s %.1f." % (100*Adc.sum()/andc,andc/meastimedc)
    print "Sum/Total before dark current correction: %.2f. Counts on anode %.1f cps. Monitor %.1f cps." %(100*A.sum()/an1,an1/meastime1,monitor1corrected/meastime1)
    print "Sum/Total after dark current correction: %.2f." % (100*sumA2/anA2)
    errA=pylab.sqrt(A)
    errAdc=pylab.sqrt(Adc)
    errmonitor1corrected=mo1+modc*meastime1/meastimedc
    
    errA2=pylab.sqrt(1/(sens*monitor1corrected)**2*errA**2+
                     (meastime1/(meastimedc*sens*monitor1corrected))**2*errAdc**2+
                     (1/(monitor1corrected**2*sens)*(A-Adc*meastime1/meastimedc))**2*errmonitor1corrected**2+
                     (1/(monitor1corrected*sens**2*(A-Adc*meastime1/meastimedc))**2)*senserr**2)
    A3=A2*anA2/(sumA2*transmave)
    errA3=pylab.sqrt((anA2/(sumA2*transmave)*errA2)**2+
                     (A2/(sumA2*transmave)*anA2err)**2+
                     (A2*anA2/(sumA2**2*transmave)*sumA2err)**2+
                     (A2*anA2/(sumA2*transmave**2)*transmerr)**2)
    #normalize by beam size
    Bx=header[0]['XPixel']
    By=header[0]['YPixel']
    return A3/(Bx*By),errA3/(Bx*By)
#EXPERIMENTAL (DANGER ZONE)
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
def tweakplot2d(A,maxval=None,mask=None,header=None,qs=[],showqscale=True,pmin=0,pmax=1):
    """2d coloured plot of a matrix with tweaking in the colorscale.
    
    Inputs:
        A: the matrix
        maxval: maximal value, see plot2dmatrix()
        mask: mask matrix, see plot2dmatrix()
        header: header data, see plot2dmatrix()
        qs: qs see plot2dmatrix()
        showqscale: see plot2dmatrix()
        pmin: lower scaling limit (proportion, default=0)
        pmax: upper scaling limit (proportion, default=1)
    """
    f=pylab.figure()
    f.donetweakplot=False
    a2=pylab.axes((0.1,0.05,0.65,0.02))
    a1=pylab.axes((0.1,0.08,0.65,0.02))
    ab=pylab.axes((0.85,0.05,0.1,0.1))
    ax=pylab.axes((0.1,0.15,0.8,0.75))
    button=matplotlib.widgets.Button(ab,'OK')
    def finish(a=None,fig=f):
        f.donetweakplot=True
    button.on_clicked(finish)
    sl1=matplotlib.widgets.Slider(a1,'vmin',0,1,pmin)
    sl2=matplotlib.widgets.Slider(a2,'vmax',0,1,pmax)
    def redraw(tmp=None,ax=ax,sl1=sl1,sl2=sl2):
        ax.cla()
        plot2dmatrix(A,maxval,mask,header,qs,showqscale,pmin=sl1.val,pmax=sl2.val)
        pylab.gcf().show()    
    sl1.on_changed(redraw)
    sl2.on_changed(redraw)
    redraw()
    while not f.donetweakplot:
        f.waitforbuttonpress()
    pylab.close(f)
    print sl1.val,sl2.val
    return (sl1.val,sl2.val)
def plotasa(asadata):
    """Plot SAXS/WAXS measurement read by readasa().
    
    Input:
        asadata: ASA dictionary (see readasa()
    
    Output:
        none, a graph is plotted.
    """
    pylab.figure()
    pylab.subplot(211)
    pylab.plot(pylab.arange(len(asadata['position'])),asadata['position'],label='Intensity',color='black')
    pylab.xlabel('Channel number')
    pylab.ylabel('Counts')
    pylab.title('Scattering data')
    pylab.legend(loc='best')
    pylab.subplot(212)
    x=pylab.arange(len(asadata['energy']))
    e1=asadata['energy'][(x<asadata['params']['Energywindow_Low'])]
    x1=x[(x<asadata['params']['Energywindow_Low'])]
    e2=asadata['energy'][(x>=asadata['params']['Energywindow_Low']) &
                         (x<=asadata['params']['Energywindow_High'])]
    x2=x[(x>=asadata['params']['Energywindow_Low']) &
         (x<=asadata['params']['Energywindow_High'])]
    e3=asadata['energy'][(x>asadata['params']['Energywindow_High'])]
    x3=x[(x>asadata['params']['Energywindow_High'])]

    pylab.plot(x1,e1,label='excluded',color='red')
    pylab.plot(x2,e2,label='included',color='black')
    pylab.plot(x3,e3,color='red')
    pylab.xlabel('Energy channel number')
    pylab.ylabel('Counts')
    pylab.title('Energy (pulse-area) spectrum')
    pylab.legend(loc='best')
    pylab.suptitle(asadata['params']['Title'])
def uglyui():
    """Ugly but usable user interface for SAXS and WAXS data treatment
    """
    uiparams={'SAXS_bc':None,'SAXS_hperL':None,'WAXS_a':None,'WAXS_b':None,'wavelength':1.54}
    def menu(menutitle,menuitems,default=0):
        choice=-1
        while (choice<0) or (choice>=len(menuitems)):
            print menutitle
            for i in range(len(menuitems)):
                print "%d: ",menuitems[i]
                try:
                    choice=int(raw_input("Select a number:"))
                except:
                    choice=-1
        return choice
    def input_float(prompt='',low=-pylab.inf,high=pylab.inf):
        val=None
        while val is None:
            val=raw_input(prompt)
            try:
                val=float(val)
                if (val<low) or (val>high):
                    val=None
            except:
                val=None
    def input_caseinsensitiveword(prompt='',list=[]):
        word=None
        while word is None:
            word=raw_input(prompt)
            if len(list)==0:
                return word
            word=word.upper()
            for w in list:
                if word==w.upper():
                    return w
                word=None
    def do_desmear(asa):
        s=input_float('Smoothing parameter for desmearing (negative to select by hand):')
        if s <0:
            print "Setting up GUI for smoothing."
            smoothlow=input_float('Lowest smoothing value: ',0)
            smoothhigh=input_float('Highest smoothing value: ',0)
            smoothmode=input_casesensitiveword('Mode of the smoothing scale bar (lin or log): ',['lin','log'])
            s={'low':smoothlow,'high':smoothhigh,'mode':smoothmode,'val':0.5*(smoothlow+smoothhigh)}
        p={}
        p['pixelmin']=input_float('Lowest pixel to take into account (starting from 0):',0,len(asa['position']))
        p['pixelmax']=input_float('Highest pixel to take into account (starting from 0):',pixelmin,len(asa['position']))
        tmp=input_casesensitiveword('Do you have a desmearing matrix saved to a file (y or n):',['y','n'])
        if tmp=='y':
            fmatrix=raw_input('Please supply the file name:')
            try:
                p['matrix']=pylab.loadtxt(fmatrix)
            except:
                print "Could not load file. Falling back to manual selection"
                tmp='n'
                mat=None
        if tmp=='n':
            p['beamcenter']=input_float('Pixel coordinate of the beam center:')
            p['pixelsize']=input_float('Pixel size in micrometers:',0)
            p['lengthbaseh']=input_float('Length of the base of the horizontal beam trapezoid',0)
            p['lengthtoph']=input_float('Length of the top of the horizontal beam trapezoid',0)
            p['lengthbasev']=0
            p['lengthtopv']=0
        print "Desmearing..."
        pixels,desmeared,smoothed,mat,params,smoothing=directdesmear(asa['position'],s,p)
        x=pylab.arange(len(asa['position']))
        outname=raw_input('Output file name:')
        try:
            f=fopen(outname,'wt')
            f.write('# pixel\toriginal\tsmoothed\tdesmeared\n')
            for i in range(len(pixels)):
                f.write('%d\t%g\t%g\t%g\n' %(pixels[i],asa['position'][x==pixels[i]],smoothed[i],desmeared[i]))
            f.close()
        except:
            print "Could not write file %s" % outname
        tmp=input_caseinsensitiveword('Would you like to save the smearing matrix for later use (y or n):',['y','n'])
        if tmp=='y':
            outname=raw_input('File to save the matrix:')
            try:
                pylab.savetxt(outname,mat)
            except:
                print "Could not write file %s" % outname
        return pixels,desmeared,smoothed,mat,params,smoothing
    a=menu('SWAXS evaluation.',['Exit program','Do AgSt calibration',
                                               'Do Tripalmitine calibration','Desmear',
                                               'Plot original dataset'
                                               'q-calibration of SAXS data',
                                               'q-calibration of WAXS data',
                                               'Show parameters',
                                               'Set parameters'])
    if a==0:
        return
    elif a==1:
        fname=raw_input('AgSt measurement file basename (without .P00 extension but may contain path):')
        asa=readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        if raw_input('Desmear before picking peaks (y or n):').upper()=='Y':
            pixels,desmeared,smoothed,mat,params,smoothing=do_desmear(asa)
            x=pixels
            y=desmeared
        else:
            x=pylab.arange(len(asa['position']))
            y=asa['position']
        npeaks=input_float('How many AgSt peaks do you have (at least 2):',2)
        a,b,aerr,berr=agstcalib(x,y,pylab.arange(npeaks),returnq=False,wavelength=ui)
        uiparams['SAXS_bc']=b
        uiparams['SAXS_hperL']=a
    elif a==2:
        fname=raw_input('Tripalmitine measurement file basename (without .P00 extension but may contain path):')
        asa=readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        x=pylab.arange(len(asa['position']))
        y=asa['position']
        npeaks=input_float('How many AgSt peaks do you have (at least 2):',2)
        a,b,aerr,berr=tripcalib(x,y,returnq=False)
        uiparams['WAXS_b']=b
        uiparams['WAXS_a']=a
    elif a==3:
        fname=raw_input('Measurement file basename (without .P00 extension but may contain path):')
        asa=readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        pixels,desmeared,smoothed,mat,params,smoothing=do_desmear(asa)
    elif a==4:
        fname=raw_input('Measurement file basename (without .P00 extension but may contain path):')
        asa=readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        plotasa(asa)
    elif a==5:
        if ((uiparams['SAXS_bc'] is None) or (uiparams['SAXS_hperL'] is None) or 
            (uiparams['wavelength'] is None)):
            print """Parameters for SAXS calibration are not yet set. Please set them
                     via the "Set parameters" or "AgSt calibration" menu items!"""
            return
        fname=raw_input('Measurement file basename (without .P00 extension but may contain path):')
        asa=readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        x=4*pylab.pi*pylab.sin(0.5*pylab.arctan((pylab.arange(len(asa['position']))-uiparams['SAXS_bc'])*uiparams['SAXS_hperL']))/uiparams['wavelength']
        outfile=raw_input('Output filename:')
def asa_qcalib(asadata,a,b):
        pass
def tripcalib2(xdata,ydata,peakmode='Lorentz',wavelength=1.54,qvals=2*pylab.pi*pylab.array([0.21739,0.25641,0.27027]),returnq=True):
    pcoord=[]
    peaks=range(len(qvals))
    for p in peaks:
        tmp=findpeak(xdata,ydata,
                     ('Zoom to peak %d (q = %f) and press ENTER' % (p,qvals[p])),
                     peakmode,scaling='lin')
        pcoord.append(tmp)
    pcoord=pylab.array(pcoord)
    n=pylab.array(peaks)
    a,b,aerr,berr=linfit(pcoord,qvals)
    if returnq:
        return a*xdata+b
    else:
        return a,b,aerr,berr

    q=a*x+b
    bc=(0-b)/float(a)
    alpha=60*pylab.pi/180.0
    h=52e-3
    l=150
    def xtoq(x,bc,alpha,h,l,wavelength=wavelength):
        twotheta=pylab.arctan((x-bc)*h*pylab.sin(alpha)/(l-(x-bc)*h*pylab.cos(alpha)))
        return 4*pylab.pi*pylab.sin(0.5*twotheta)/wavelength
    def costfunc(p,x,y):
        pass

