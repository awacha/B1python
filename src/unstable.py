#-----------------------------------------------------------------------------
# Name:        unstable.py
# Purpose:     Unstable macros! Use with caution!
#
# Author:      Andras Wacha
#
# Created:     2010/02/22
# RCS-ID:      $Id: unstable.py $
# Copyright:   (c) 2010
# Licence:     GPLv2
#-----------------------------------------------------------------------------

import numpy as np
import pylab
import utils
import types
import scipy.stats.stats
import scipy.signal
import matplotlib.widgets
import B1io
import guitools
import asamacros
import fitting
from c_unstable import indirectdesmear
HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

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
    data['q']=np.vstack(tuple([t['q'].reshape(t['q'].size,1) for t in tup]))
    data['Intensity']=np.vstack(tuple([t['Intensity'].reshape(t['Intensity'].size,1) for t in tup]))
    data['Error']=np.vstack(tuple([t['Error'].reshape(t['Error'].size,1) for t in tup]))
    tmp=np.vstack((data['q'].reshape(1,data['q'].size),data['Intensity'].reshape(1,data['Intensity'].size),data['Error'].reshape(1,data['Error'].size)))
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
            The np.linalg.linalg.cond() function is used to determine this.  If
            the matrix is non-square (ie. rectangular), this type of condition
            number can still be determined from the singular value
            decomposition.
             
        """
        f1=np.interp(energies,f1f2[:,0],f1f2[:,1])
        f2=np.interp(energies,f1f2[:,0],f1f2[:,2])
        A=np.ones((len(energies),3));
        A[:,1]=2*(f1+atomicnumber);
        A[:,2]=(f1+atomicnumber)**2+f2**2;
        B=np.dot(np.inv(np.dot(A.T,A)),A.T);
        return np.linalg.linalg.cond(B)

    np.random.seed()
    f1f2=f1f2[f1f2[:,0]<=(energymax+100),:]
    f1f2=f1f2[f1f2[:,0]>=(energymin-100),:]
    energies=np.random.rand(Nenergies)*(energymax-energymin)+energymin
    c=matrixcond(f1f2,energies)
    ok=False
    oldenergies=energies.copy()
    oldc=c
    cs=np.zeros(NITERS)
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
            eidx=int(np.random.rand()*Nenergies)
            #modify energy in which direction?
            sign=2*np.floor(np.random.rand()*2)-1
            #modify energy
            energies[eidx]=energies[eidx]+sign*stepsize
            # if the modified energy is inside the bounds and the current
            # energies are different, go on.
            if energies.min()>=energymin and energies.max()<=energymax and len(utils.unique(energies,lambda a,b:(abs(a-b)<energydistance)))==Nenergies:
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
        except np.linalg.LinAlgError:
            energies=oldenergies
            c=oldc
            drops=drops+1
        else:
            if c>oldc: #if the condition number is larger than the old one,
                if np.random.rand()>(np.exp(c-oldc)/kT): # drop with some probability
                    energies=oldenergies
                    c=oldc
                    drops=drops+1
        cs[i]=c # save the current value for the condition number
        if np.mod(i,1000)==0: # printing is slow, only print every 1000th step
#            print i
            pass
        if c<condmin:
            condmin=c;
            energiesmin=energies.copy()
    energies=energiesmin
    f1end=np.interp(energies,f1f2[:,0],f1f2[:,1])
    f2end=np.interp(energies,f1f2[:,0],f1f2[:,2])
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
    Y,X=np.meshgrid(np.arange(data.shape[1]),np.arange(data.shape[0]));
    D=np.sqrt((res[0]*(X-bcx))**2+
                 (res[1]*(Y-bcy))**2)
    # Q-matrix is calculated from the D matrix
    q1=4*np.pi*np.sin(0.5*np.arctan(D/float(distance)))*energy/float(HC)
    # eliminating masked pixels
    data=data[mask==0]
    q1=q1[mask==0]
    q=np.array(q)
    q1=q1[np.isfinite(data)]
    data=data[np.isfinite(data)]
    # initialize the output matrix
    hist=np.zeros((len(I),len(q)))
    # set the bounds of the q-bins in qmin and qmax
    qmin=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmin.insert(0,q[0])
    qmin=np.array(qmin)
    qmax=map(lambda a,b:(a+b)/2.0,q[1:],q[:-1])
    qmax.append(q[-1])
    qmax=np.array(qmax)
    # go through every pixel
    for l in range(len(q)):
        indices=((q1<=qmax[l])&(q1>qmin[l])) # the indices of the pixels which belong to this q-bin
        hist[:,l]=scipy.stats.stats.histogram2(data[indices],I)/np.sum(indices.astype('float'))
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
        guitools.plot2dmatrix(A,maxval,mask,header,qs,showqscale,pmin=sl1.val,pmax=sl2.val)
        pylab.gcf().show()
        pylab.draw()
    sl1.on_changed(redraw)
    sl2.on_changed(redraw)
    redraw()
    while not f.donetweakplot:
        f.waitforbuttonpress()
    pylab.close(f)
    print sl1.val,sl2.val
    return (sl1.val,sl2.val)
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
    def input_float(prompt='',low=-np.inf,high=np.inf):
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
            smoothmode=input_caseinsensitiveword('Mode of the smoothing scale bar (lin or log): ',['lin','log'])
            s={'low':smoothlow,'high':smoothhigh,'mode':smoothmode,'val':0.5*(smoothlow+smoothhigh)}
        p={}
        p['pixelmin']=input_float('Lowest pixel to take into account (starting from 0):',0,len(asa['position']))
        p['pixelmax']=input_float('Highest pixel to take into account (starting from 0):',p['pixelmin'],len(asa['position']))
        tmp=input_caseinsensitiveword('Do you have a desmearing matrix saved to a file (y or n):',['y','n'])
        if tmp=='y':
            fmatrix=raw_input('Please supply the file name:')
            try:
                p['matrix']=np.loadtxt(fmatrix)
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
        pixels,desmeared,smoothed,mat,params,smoothing=asamacros.directdesmear(asa['position'],s,p)
        x=np.arange(len(asa['position']))
        outname=raw_input('Output file name:')
        try:
            f=open(outname,'wt')
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
                np.savetxt(outname,mat)
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
        asa=B1io.readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        if raw_input('Desmear before picking peaks (y or n):').upper()=='Y':
            pixels,desmeared,smoothed,mat,params,smoothing=do_desmear(asa)
            x=pixels
            y=desmeared
        else:
            x=np.arange(len(asa['position']))
            y=asa['position']
        npeaks=input_float('How many AgSt peaks do you have (at least 2):',2)
        a,b,aerr,berr=asamacros.agstcalib(x,y,np.arange(npeaks),returnq=False,wavelength=uiparams['wavelength'])
        uiparams['SAXS_bc']=b
        uiparams['SAXS_hperL']=a
    elif a==2:
        fname=raw_input('Tripalmitine measurement file basename (without .P00 extension but may contain path):')
        asa=B1io.readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        x=np.arange(len(asa['position']))
        y=asa['position']
        npeaks=input_float('How many AgSt peaks do you have (at least 2):',2)
        a,b,aerr,berr=asamacros.tripcalib(x,y,returnq=False)
        uiparams['WAXS_b']=b
        uiparams['WAXS_a']=a
    elif a==3:
        fname=raw_input('Measurement file basename (without .P00 extension but may contain path):')
        asa=B1io.readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        pixels,desmeared,smoothed,mat,params,smoothing=do_desmear(asa)
    elif a==4:
        fname=raw_input('Measurement file basename (without .P00 extension but may contain path):')
        asa=B1io.readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        guitools.plotasa(asa)
    elif a==5:
        if ((uiparams['SAXS_bc'] is None) or (uiparams['SAXS_hperL'] is None) or 
            (uiparams['wavelength'] is None)):
            print """Parameters for SAXS calibration are not yet set. Please set them
                     via the "Set parameters" or "AgSt calibration" menu items!"""
            return
        fname=raw_input('Measurement file basename (without .P00 extension but may contain path):')
        asa=B1io.readasa(fname)
        if asa is None:
            print "Cannot find file %s.{p00,e00,inf} (If on Linux, check the case)" % fname
            return
        x=4*np.pi*np.sin(0.5*np.arctan((np.arange(len(asa['position']))-uiparams['SAXS_bc'])*uiparams['SAXS_hperL']))/uiparams['wavelength']
        outfile=raw_input('Output filename:')
def asa_qcalib(asadata,a,b):
        pass
def tripcalib2(xdata,ydata,peakmode='Lorentz',wavelength=1.54,qvals=2*np.pi*np.array([0.21739,0.25641,0.27027]),returnq=True):
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

    q=a*xdata+b
    bc=(0-b)/float(a)
    alpha=60*np.pi/180.0
    h=52e-3
    l=150
    def xtoq(x,bc,alpha,h,l,wavelength=wavelength):
        twotheta=np.arctan((x-bc)*h*np.sin(alpha)/(l-(x-bc)*h*np.cos(alpha)))
        return 4*np.pi*np.sin(0.5*twotheta)/wavelength
    def costfunc(p,x,y):
        pass


def makesensitivity2(fsnrange,energypre,energypost,title,fsnDC,energymeas,energycalib,energyfluorescence,origx,origy,mask=None,savefile=None):
    """Create matrix for detector sensitivity correction
    
    Inputs:
        fsnrange: FSN range of the sensitivity measurement
        energypre: apparent (uncalibrated) energy of the pre-edge measurements
        energypost: apparent (uncalibrated) energy of the after-edge measurements
        title: title of the sensitivity foil measurements. If the measurements
            were recorded with more titles, give a list of strings.
        fsnDC: a single number or a list of FSN-s for dark current
            measurements
        energymeas: apparent energies for energy calibration
        energycalib: calibrated energies for energy calibration
        energyfluorescence: energy of the fluorescence
        origx, origy: the centers of the beamstop (row, column), starting from 1.
        mask: a mask to apply (1 for valid, 0 for masked pixels). If None,
            a makemask() window will pop up for user interaction.
        savefile: a .npz file to save results to. None to skip saving.
        
    Outputs: sensdict
        sens: a dictionary of the following fields:
            sens: the sensitivity matrix of the 2D detector, by which all
                measured data should be divided pointwise. The matrix is
                normalized to 1 on the average.
            errorsens: the calculated error of the sensitivity matrix
            chia, chim, alpha, beta, S1S: these values (4 matrices and
                a scalar) are needed for calculation of the correction
                terms (taking the dependence of a_0... and S into
                account)
    """
    # Watch out: this part is VERY UGLY. If you want to understand what
    # this does (and you better want it ;-)) please take a look at the
    # PDF file attached to the source
    global _B1config
    
    pixelsize=_B1config['pixelsize']

    # these are controls for non-physical tampering. Their theoretical
    # value is in [brackets]
    factor=1 # tune this if the scattering is not completely subtracted
                # from the fluorescence [1]
    t0scaling=1 # multiply the dark-current measurement time by this [1]
    hackDCsub=True # set nonpositive elements of DC-subtracted PSD counts to their smallest positive value. [False]
    transmerrors=True #if the error of the transmission should be accounted for [True]
    
    # some housekeeping...
    if energypost<energypre: # if the two energies were input as swapped
        tmp=energypost
        energypost=energypre
        energypre=tmp
    if type(title)==type(''): # make a list of "title"
        title=[title]
    #read in every measurement file
    print "makesensitivity: reading files"
    data,header=B1io.read2dB1data(_B1config['2dfileprefix'],fsnrange,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    dataDC,headerDC=B1io.read2dB1data(_B1config['2dfileprefix'],fsnDC,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
   
    EB1=[]; hEB1=[]
    EB2=[]; hEB2=[]
    F1=[]; hF1=[]
    F2=[]; hF2=[]
    print "makesensitivity: summarizing"
    # now summarize...
    print "list of sample titles:",title
    print "pre-edge energy:",energypre
    print "post-edge energy:",energypost
        
    for i in range(len(data)):
        if abs(header[i]['Energy']-energypost)<=_B1config['energyprecision']:
            if header[i]['Title']==_B1config['ebtitle']:
                EB1.append(data[i])
                hEB1.append(header[i])
            elif header[i]['Title'] in title:
                F1.append(data[i])
                hF1.append(header[i])
            else:
                print "Unknown title (neither empty beam, nor sample): ","*%s*" % header[i]['Title']
                print "(FSN: ",header[i]['FSN'],", Energy: ",header[i]['Energy'],")"
        elif abs(header[i]['Energy']-energypre)<=_B1config['energyprecision']:
            if header[i]['Title']==_B1config['ebtitle']:
                EB2.append(data[i])
                hEB2.append(header[i])
            elif header[i]['Title'] in title:
                F2.append(data[i])
                hF2.append(header[i])
            else:
                print "Unknown title (neither empty beam, nor sample): ","*%s*" % header[i]['Title']
                print "(FSN: ",header[i]['FSN'],", Energy: ",header[i]['Energy'],")"
        else:
            print "Unknown energy (neither preedge, nor post-edge): ",header[i]['Energy']
            print "(FSN: ",header[i]['FSN'],", Title: ",header[i]['Title'],")"
            print "Neglecting this file."
            # go on, this is not an error, just a warning.
    #print a short summary
    print "Empty beam measurements before the edge:",len(EB2)
    print "Empty beam measurements after the edge:",len(EB1)
    print "Foil measurements before the edge:",len(F2)
    print "Foil measurements after the edge:",len(F1)
    print "Energy precision:",_B1config['energyprecision']
    #summarize the scattering matrices
    D1=sum(F1) # the builtin sum function
    D2=sum(F2)
    E1=sum(EB1)
    E2=sum(EB2)
    D0=sum(dataDC)
    # summarize the measurement times
    t1=sum([h['MeasTime'] for h in hF1])
    t2=sum([h['MeasTime'] for h in hF2])
    te1=sum([h['MeasTime'] for h in hEB1])
    te2=sum([h['MeasTime'] for h in hEB2])
    t0=sum([h['MeasTime'] for h in headerDC])*t0scaling
    # summarize the anode counts
    a1=sum([h['Anode'] for h in hF1])
    a2=sum([h['Anode'] for h in hF2])
    ae1=sum([h['Anode'] for h in hEB1])
    ae2=sum([h['Anode'] for h in hEB2])
    a0=sum([h['Anode'] for h in headerDC])
    # summarize the monitor counts
    m1=sum([h['Monitor'] for h in hF1])
    m2=sum([h['Monitor'] for h in hF2])
    me1=sum([h['Monitor'] for h in hEB1])
    me2=sum([h['Monitor'] for h in hEB2])
    m0=sum([h['Monitor'] for h in headerDC])
    # calculate the transmissions
    T1=np.array([h['Transm'] for h in hF1]).mean()
    T2=np.array([h['Transm'] for h in hF2]).mean()
    if transmerrors:
        print "Taking error of transmission into account"
        dT1=np.array([h['Transm'] for h in hF1]).std()
        dT2=np.array([h['Transm'] for h in hF2]).std()
    else:
        print "NOT taking error of transmission into account"
        dT1=0
        dT2=0
    print "Transmission before the edge:",T2,"+/-",dT2
    print "Transmission after the edge:",T1,"+/-",dT1
    print "monitor_sample_above: ",m1
    print "monitor_sample_below: ",m2
    print "monitor_empty_above: ",me1
    print "monitor_empty_below: ",me2
    print "monitor_dark: ",m0

    print "anode_sample_above: ",a1
    print "anode_sample_below: ",a2
    print "anode_empty_above: ",ae1
    print "anode_empty_below: ",ae2
    print "anode_dark: ",a0

    print "time_sample_above: ",t1
    print "time_sample_below: ",t2
    print "time_empty_above: ",te1
    print "time_empty_below: ",te2
    print "time_dark: ",t0

    print "makesensitivity: calculating auxiliary values"
    # error values of anode counts
    da1=np.sqrt(a1);    da2=np.sqrt(a2);    dae1=np.sqrt(ae1);
    dae2=np.sqrt(ae2);    da0=np.sqrt(a0)

    # errors of monitor counts
    dm1=np.sqrt(m1);    dm2=np.sqrt(m2);    dme1=np.sqrt(me1);
    dme2=np.sqrt(me2);    dm0=np.sqrt(m0)

    # errors of 2d detector counts
    dD1=np.sqrt(D1);    dD2=np.sqrt(D2);    dE1=np.sqrt(E1);
    dE2=np.sqrt(E2);    dD0=np.sqrt(D0)
    
    # Dark current correction: abc -> abcx
    D1x=D1-t1/t0*D0 # scattering matrices...
    D2x=D2-t2/t0*D0
    E1x=E1-te1/t0*D0
    E2x=E2-te2/t0*D0
    a1x=a1-t1/t0*a0 # anode counts...
    a2x=a2-t2/t0*a0
    ae1x=ae1-te1/t0*a0
    ae2x=ae2-te2/t0*a0
    m1x=m1-t1/t0*m0 # monitor...
    m2x=m2-t2/t0*m0
    me1x=me1-te1/t0*m0
    me2x=me2-te2/t0*m0

    if hackDCsub:
        #In principle, correction for dark current should not make
        # scattering images negative. This hack corrects that problem.
        print "Tampering with dark current subtraction: setting negative values to positive!"
        D1x[D1x<=0]=np.nanmin(D1x[D1x>0])
        D2x[D2x<=0]=np.nanmin(D2x[D2x>0])
        E1x[E1x<=0]=np.nanmin(E1x[E1x>0])
        E2x[E2x<=0]=np.nanmin(E2x[E2x>0])
        print "Tampering done."
    
    print "Scattering images, corrected by dark current:"
    print utils.matrixsummary(D1x,"D1x")
    print utils.matrixsummary(E1x,"E1x")
    print utils.matrixsummary(D2x,"D2x")
    print utils.matrixsummary(E2x,"E2x")


    #two-theta for the pixels
    tth=np.arctan(utils2d.calculateDmatrix(D1,pixelsize,origx,origy)/hF1[0]['Dist'])

    # angle-dependent correction matrices
    C0=gasabsorptioncorrectiontheta(energyfluorescence,tth)
    C1,dC1=absorptionangledependenttth(tth,T1,diffaswell=True)
    C2,dC2=absorptionangledependenttth(tth,T2,diffaswell=True)
    print "Angle-dependent correction matrices:"
    print utils.matrixsummary(C1,"C1")
    print utils.matrixsummary(C2,"C2")
    print utils.matrixsummary(dC1,"dC1")
    print utils.matrixsummary(dC2,"dC2")
    
    # some auxiliary variables:
    P1=D1x*a1x/(T1*m1x*D1x.sum())*C0*C1
    Q1=-E1x*ae1x/(me1x*E1x.sum())*C0
    P2=-factor*D2x*a2x/(T2*m2x*D2x.sum())*C0*C2
    Q2=factor*E2x*ae2x/(me2x*E2x.sum())*C0

#    pylab.figure();    pylab.imshow(C0);    pylab.title('C0');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    pylab.imshow(C1);    pylab.title('C1');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    pylab.imshow(C2);    pylab.title('C2');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    pylab.imshow(P1);    pylab.title('P1');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    pylab.imshow(P2);    pylab.title('P2');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    pylab.imshow(Q1);    pylab.title('Q1');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    pylab.imshow(Q2);    pylab.title('Q2');    pylab.gcf().show(); pylab.colorbar()

    # the unnormalized, unmasked sensitivity matrix.
    S1=P1+Q1+P2+Q2
#    pylab.figure();    pylab.imshow(S1);    pylab.title('S1');    pylab.gcf().show(); pylab.colorbar()
    if mask is None:
        mask=np.ones(S1.shape)
    pylab.figure()
    print "makesensitivity: Please mask erroneous areas!"
    mask = guitools.makemask(mask,S1)

#    pylab.figure();    guitools.plot2dmatrix(D1x,mask=mask,blacknegative=True);    pylab.title('D1x');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    guitools.plot2dmatrix(E1x,mask=mask,blacknegative=True);    pylab.title('E1x');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    guitools.plot2dmatrix(D2x,mask=mask,blacknegative=True);    pylab.title('D2x');    pylab.gcf().show(); pylab.colorbar()
#    pylab.figure();    guitools.plot2dmatrix(E2x,mask=mask,blacknegative=True);    pylab.title('E2x');    pylab.gcf().show(); pylab.colorbar()

    P1=P1*mask
    Q1=Q1*mask
    P2=P2*mask
    Q2=Q2*mask
    # multiply the matrix by the mask: masked areas will be zeros.
    S1=S1*mask
    # summarize the matrix (masking was already taken into account)
    S1S=S1.sum()
    S=S1/S1S # normalize.

    # now the S matrix is ready.
    print "makesensitivity: calculating error terms"
    
    #we put together the error of S. The terms are denoted ET_<variable>,
    # each corresponding to (\frac{d S_{ij}}{d <variable>})^2*Delta^2 <variable>,
    # with exception of the matrices, where ET_<mx> corresponds to
    # \sum_{mn}(\frac{d S_{ij}}{d <mx>_{mn}})^2Delta^2<mx>_{mn}.

    diff=P1/a1x
    ET_a1=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*da1**2
    diff=P2/a2x
    ET_a2=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*da2**2
    diff=Q1/ae1x
    ET_ae1=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dae1**2
    diff=Q2/ae2x
    ET_ae2=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dae2**2

    diff=P1/m1x
    ET_m1=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dm1**2
    diff=P2/m2x
    ET_m2=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dm2**2
    diff=Q1/me1x
    ET_me1=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dme1**2
    diff=Q2/me2x
    ET_me2=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dme2**2

    diff=-(t1/t0*P1/a1x+te1/t0*Q1/ae1x+t2/t0*P2/a2x+te2/t0*Q2/ae2x)
    ET_a0=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*da0**2
    chia=-diff # save it for returning
    
    diff=(t1/t0*P1/m1x+te1/t0*Q1/me1x+t2/t0*P2/m2x+te2/t0*Q2/me2x)
    ET_m0=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dm0**2
    chim=diff # save it for returning

    # the error of the transmissions also have this form, only diff differs.
    diff=-P1/T1+P1/C1*dC1
    ET_T1=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dT1**2

    diff=-P2/T2+P2/C2*dC2
    ET_T2=1/S1S**2*(diff**2+S**2*diff.sum()**2-2*S*diff*diff.sum())*dT2**2

    #we should take extra care here. Some elements of D1x can be zero.
    # This shouldn't be a problem, as zero elements are invalid, so they
    # should be under the mask. Numpy signals float divisions by zero
    # by setting the result to NaN (not-a-number). Summarizing a matrix
    # however, if it contains NaN elements, renders the sum to be NaN
    # as well. Therefore we use np.nansum() instead of sum().
    alpha=P1/D1x
    beta=P1/np.nansum(D1x)**2
    dD=dD1
    ET_D1=1/S1S**2*(alpha**2-2*alpha*beta-2*S*alpha*(alpha-np.nansum(beta)))*dD**2+ \
          1/S1S**2*(beta**2+S**2*np.nansum(beta)**2-2*S*beta*np.nansum(beta))*np.nansum(dD**2) + \
          1/S1S**2*S*np.nansum(alpha**2*dD**2)+ \
          1/S1S**2*(2*S*beta-2*S**2*np.nansum(beta))*np.nansum(alpha*dD**2)
    alpha=Q1/E1x
    beta=Q1/np.nansum(E1x)**2
    dD=dE1
    ET_E1=1/S1S**2*(alpha**2-2*alpha*beta-2*S*alpha*(alpha-np.nansum(beta)))*dD**2+ \
          1/S1S**2*(beta**2+S**2*np.nansum(beta)**2-2*S*beta*np.nansum(beta))*np.nansum(dD**2) + \
          1/S1S**2*S*np.nansum(alpha**2*dD**2)+ \
          1/S1S**2*(2*S*beta-2*S**2*np.nansum(beta))*np.nansum(alpha*dD**2)
    alpha=P2/D2x
    beta=P2/np.nansum(D2x)**2
    dD=dD2
    ET_D2=1/S1S**2*(alpha**2-2*alpha*beta-2*S*alpha*(alpha-np.nansum(beta)))*dD**2+ \
          1/S1S**2*(beta**2+S**2*np.nansum(beta)**2-2*S*beta*np.nansum(beta))*np.nansum(dD**2) + \
          1/S1S**2*S*np.nansum(alpha**2*dD**2)+ \
          1/S1S**2*(2*S*beta-2*S**2*np.nansum(beta))*np.nansum(alpha*dD**2)
    alpha=Q2/E2x
    beta=Q2/np.nansum(E2x)**2
    dD=dE2
    ET_E2=1/S1S**2*(alpha**2-2*alpha*beta-2*S*alpha*(alpha-np.nansum(beta)))*dD**2+ \
          1/S1S**2*(beta**2+S**2*np.nansum(beta)**2-2*S*beta*np.nansum(beta))*np.nansum(dD**2) + \
          1/S1S**2*S*np.nansum(alpha**2*dD**2)+ \
          1/S1S**2*(2*S*beta-2*S**2*np.nansum(beta))*np.nansum(alpha*dD**2)

    alpha=P1/D1x+P2/D2x+Q1/E1x+Q2/E2x
    beta=P1/np.nansum(D1x)**2+P2/np.nansum(D2x)**2+Q1/np.nansum(E1x)**2+Q2/np.nansum(E2x)**2
    dD=dD0
    ET_D0=1/S1S**2*(alpha**2-2*alpha*beta-2*S*alpha*(alpha-np.nansum(beta)))*dD**2+ \
          1/S1S**2*(beta**2+S**2*np.nansum(beta)**2-2*S*beta*np.nansum(beta))*np.nansum(dD**2) + \
          1/S1S**2*S*np.nansum(alpha**2*dD**2)+ \
          1/S1S**2*(2*S*beta-2*S**2*np.nansum(beta))*np.nansum(alpha*dD**2)
    #the last alpha and beta are returned!!!

    # the error matrix
    dS=np.sqrt(ET_a1+ET_a2+ET_ae1+ET_ae2+ET_a0+ET_m1+ET_m2+ET_me1+ET_me2+ET_m0+ \
               ET_D1+ET_D2+ET_E1+ET_E2+ET_D0+ET_T1+ET_T2)
    print "The error terms:"
    print utils.matrixsummary(ET_a1,"ET_a1")
    pylab.hist(ET_a1.flatten(),bins=100,log=True)
    print utils.matrixsummary(ET_a2,"ET_a2")
    print utils.matrixsummary(ET_ae1,"ET_ae1")
    print utils.matrixsummary(ET_ae2,"ET_ae2")
    print utils.matrixsummary(ET_a0,"ET_a0")
    print utils.matrixsummary(ET_m1,"ET_m1")
    print utils.matrixsummary(ET_m2,"ET_m2")
    print utils.matrixsummary(ET_me1,"ET_me1")
    print utils.matrixsummary(ET_me2,"ET_me2")
    print utils.matrixsummary(ET_m0,"ET_m0")
    print utils.matrixsummary(ET_T1,"ET_T1")
    print utils.matrixsummary(ET_T2,"ET_T2")
    print utils.matrixsummary(ET_D1,"ET_D1")
    print utils.matrixsummary(ET_D2,"ET_D2")
    print utils.matrixsummary(ET_E1,"ET_E1")
    print utils.matrixsummary(ET_E2,"ET_E2")
    print utils.matrixsummary(ET_D0,"ET_D0")
    print "----------------"
    print utils.matrixsummary(dS**2,"dS^2")
    print utils.matrixsummary(dS,"dS")
    print utils.matrixsummary(S,"S")
    print utils.matrixsummary(dS/S,"dS/S")
    # set nans to zero
    pylab.figure()
    dS[np.isnan(dS)]=0
    guitools.plot2dmatrix(S,mask=mask,blacknegative=True)
    pylab.colorbar()
    pylab.title('Sensitivity')
    pylab.gcf().show()
    pylab.figure()
    guitools.plot2dmatrix(dS,mask=mask,blacknegative=True)
    pylab.colorbar()
    pylab.title('Error of sensitivity')
    pylab.gcf().show()
    pylab.figure()
    guitools.plot2dmatrix(dS/S,mask=mask,blacknegative=True)
    pylab.colorbar()
    pylab.title('Relative error of sensitivity')
    pylab.gcf().show()
    
    #take care of zeros (mask them and set them to 1 afterwards to avoid division by zero)
    mask[S<=0]=0
    S[S<=0]=1

    result={'sens':S,'errorsens':dS,'chia':chia,'chim':chim,'S1S':S1S,'alpha':alpha,'beta':beta,'mask':mask,'D0':D0,'a0':a0,'m0':m0,'t0':t0}
    if savefile is not None:
        if savefile[-4:].upper()=='.MAT':
            scipy.io.savemat(savefile,result,appendmat=False)
        else:
            np.savez(savefile,**result)
    return result
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
        origx, origy: the centers of the beamstop (row,column), starting from 1
    
    Outputs: sens,errorsens,mask
        sens: the sensitivity matrix of the 2D detector, by which all
            measured data should be divided pointwise. The matrix is
            normalized to 1 on the average.
        errorsens: the error of the sensitivity matrix.
        mask: mask matrix created by the user. This masks values where
            the sensitivity is invalid. This should be used as a base
            for further mask matrices.
    """
    global _B1config
    
    pixelsize=_B1config['pixelsize']
    
    fsns=range(min(fsn1,fsn2),fsnend+1) # the fsn range of the sensitivity measurement
    
    #read in every measurement file
    data,header=B1io.read2dB1data(_B1config['2dfileprefix'],fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    E1header=[h for h in header if h['FSN']==fsn1 ][0] # the header of the first measurement at E1
    # select the fsns for the measurements at E1
    E1fsns=[h['FSN'] for h in header if (abs(h['Energy']-E1header['Energy'])<0.5) and (h['Title']==E1header['Title'])]
    
    E2header=[h for h in header if h['FSN']==fsn2 ][0] # the header of the first measurement at E2
    # select the fsns for the measurements at E2
    E2fsns=[h['FSN'] for h in header if (abs(h['Energy']-E2header['Energy'])<0.5) and (h['Title']==E2header['Title'])]
    # read data and header structures for E1 and E2 measurements (not empty beams)
    dataE1,headerE1=B1io.read2dB1data(_B1config['2dfileprefix'],E1fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    dataE2,headerE2=B1io.read2dB1data(_B1config['2dfileprefix'],E2fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])

    # find all the empty beam FSNs for E1
    ebE1fsns=utils.unique([h['FSNempty'] for h in headerE1])
    # find all the empty beam FSNs for E2
    ebE2fsns=utils.unique([h['FSNempty'] for h in headerE2])
    # read data and header structures for empty beam measurements at E1 and E2
    dataebE1,headerebE1=B1io.read2dB1data(_B1config['2dfileprefix'],ebE1fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    dataebE2,headerebE2=B1io.read2dB1data(_B1config['2dfileprefix'],ebE2fsns,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    # read dark current measurement(s)
    datadc,headerdc=B1io.read2dB1data(_B1config['2dfileprefix'],fsnDC,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    #subtract background, and correct for transmission (and sensitivity :-): to override this correction, ones() and zeros() are given)
    A1,errA1=subdc(dataE1,headerE1,datadc,headerdc,np.ones(dataE1[0].shape),np.zeros(dataE1[0].shape))
    A2,errA2=subdc(dataE2,headerE2,datadc,headerdc,np.ones(dataE2[0].shape),np.zeros(dataE2[0].shape))
    eb1,erreb1=subdc(dataebE1,headerebE1,datadc,headerdc,np.ones(dataebE1[0].shape),np.zeros(dataebE1[0].shape))
    eb2,erreb2=subdc(dataebE2,headerebE2,datadc,headerdc,np.ones(dataebE2[0].shape),np.zeros(dataebE2[0].shape))
    
    #theta for pixels
    tth=np.arctan(utils2d.calculateDmatrix(A1,pixelsize,origx,origy)/headerE1[0]['Dist'])
    #transmissions below and above the edge
    transm1=np.array([h['Transm'] for h in headerE1])
    transm2=np.array([h['Transm'] for h in headerE2])
    
    #angle-dependent absorption
    transmcorr1=absorptionangledependenttth(tth,transm1.mean())/transm1.mean()
    transmcorr2=absorptionangledependenttth(tth,transm2.mean())/transm2.mean()
    
    #subtract empty beam
    B1=(A1/transm1.mean()-eb1)*transmcorr1
    B2=(A2/transm2.mean()-eb2)*transmcorr2
    errB1=np.sqrt(errA1**2/(transm1.mean())**2+erreb1**2)*transmcorr1
    errB2=np.sqrt(errA2**2/(transm2.mean())**2+erreb2**2)*transmcorr2
    
    factor=1 #empirical compensation factor to rule out small-angle scattering completely
    if (E1header['Energy']>E2header['Energy']):
        C=B1-factor*B2
        Cerr=np.sqrt(errB1**2+factor**2*errB2**2)
    else:
        C=B2-factor*B1
        Cerr=np.sqrt(errB2**2+factor**2*errB1**2)
    C=C*gasabsorptioncorrectiontheta(energyfluorescence,tth)
    print "Please mask erroneous areas!"
    mask = guitools.makemask(np.ones(C.shape),C)
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
    return sens,errorsens,mask
def B1normint1(fsn1,thicknesses,orifsn,fsndc,sens,errorsens,mask,energymeas,energycalib,distminus=0,detshift=0,orig=[122,123.5],transm=None):
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
            measurements. If sens is a sensitivity dict, this value will
            be disregarded.
        sens: sensitivity matrix, or a sensitivity dict.
        errorsens: error of the sensitivity matrix. If sens is a sensitivity
            dict, this value will be disregarded.
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
            GCdata=np.loadtxt("%s%s%s" % (_B1config['calibdir'],os.sep,r['data']))
            refthick=r['thick']
    if GCdata is None:
        print "No calibration data exists with ref. position %.2f +/- %.2f." % (header[referencenumber]['PosRef'],_B1config['refposprecision'])
        return [],[],[],[]
    print "FSN %d: Using GLASSY CARBON REFERENCE with nominal thickness %.f micrometers." % (header[referencenumber]['FSN'],refthick*1e4)
    
    #re-integrate GC measurement to the same q-bins
    print "Re-integrating GC data to the same bins at which the reference is defined"
    qGC,intGC,errGC,AGC=utils2d.radintC(As[referencenumber],
                               Aerrs[referencenumber],
                               header[referencenumber]['EnergyCalibrated'],
                               header[referencenumber]['Dist'],
                               header[referencenumber]['PixelSize'],
                               header[referencenumber]['BeamPosX'],
                               header[referencenumber]['BeamPosY'],
                               (1-mask).astype('uint8'),
                               GCdata[:,0])
    print "Re-integration done."
    GCdata=GCdata[(AGC>=_B1config['GCareathreshold']) & (intGC>=_B1config['GCintsthreshold']),:]
    errGC=errGC[(AGC>=_B1config['GCareathreshold']) & (intGC>=_B1config['GCintsthreshold'])]
    qGC=qGC[(AGC>=_B1config['GCareathreshold']) & (intGC>=_B1config['GCintsthreshold'])]
    intGC=intGC[(AGC>=_B1config['GCareathreshold']) & (intGC>=_B1config['GCintsthreshold'])]
    
    if len(intGC)<2:
        print "ERROR: re-integrated reference does not have enough points! Saving not normalized!"
        mult=1
        errmult=0
    else:
        intGC=intGC/refthick
        errGC=errGC/refthick
    
        mult,errmult=utils.multfactor(qGC,GCdata[:,1],GCdata[:,2],intGC,errGC)
    
        print "Factor for GC normalization: %.2g +/- %.2f %%" % (mult,errmult/mult*100)
        pylab.clf()
        pylab.plot(qGC,intGC*mult,'.',label='Your reference (reintegrated)')
        pylab.plot(GCdata[:,0],GCdata[:,1],'.',label='Calibrated reference')
        pylab.plot(qs[referencenumber],ints[referencenumber]*mult/refthick,'-',label='Your reference (saved)')
        pylab.xlabel('q (1/Angstrom)')
        pylab.ylabel('Scattering cross-section (1/cm)')
        pylab.title('Reference FSN %d multiplied by %.2e, error percentage %.2f' %(header[referencenumber]['FSN'],mult,(errmult/mult*100)))
        #pylab.xscale('log')
        pylab.yscale('log')
        #pause('on')
        utils.pause()
    
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
            errs[k]=np.sqrt((mult*errs[k])**2+(errmult*ints[k])**2)/thick
            ints[k]=mult*ints[k]/thick
            Aerrs[k]=np.sqrt((mult*Aerrs[k])**2+(errmult*As[k])**2)/thick
            As[k]=mult*As[k]/thick
            if ((header[k]['Current1']>header[referencenumber]['Current2']) and (k>referencenumber)) or \
                ((header[k]['Current2']<header[referencenumber]['Current1']) and (k<referencenumber)):
                header[k]['injectionGC']='y'
            else:
                header[k]['injectionGC']='n'
            if len(intGC)<2:
                norm=False
            else:
                norm=True
            B1io.writelogfile(header[k],[header[k]['BeamPosX'],header[k]['BeamPosY']],thick,fsndc,
                         header[k]['EnergyCalibrated'],header[k]['Dist'],
                         mult,errmult,header[referencenumber]['FSN'],
                         refthick,header[k]['injectionGC'],header[k]['injectionEB'],
                         header[k]['PixelSize'],mode=_B1config['detector'],norm=norm)
            if norm:
                B1io.writeintfile(qs[k],ints[k],errs[k],header[k],areas[k],filetype='intnorm')
            else:
                B1io.writeintfile(qs[k],ints[k],errs[k],header[k],areas[k],filetype='intarb')
            B1io.write2dintfile(As[k],Aerrs[k],header[k],norm=norm)

def B1integrate(fsn1,fsndc,sens,errorsens,orifsn,mask,energymeas,energycalib,distminus=0,detshift=0,orig=[122,123.5],transm=None):
    """Integrate a sequence
    
    Inputs:
        fsn1: range of fsns. The first should be the empty beam
            measurement.
        fsndc: one FSN or a list of FSNS corresponding to the empty beam
            measurements. If sens is a sensitivity dict, this value will
            be disregarded.
        sens: sensitivity matrix, or a sensitivity dict.
        errorsens: error of the sensitivity matrix. If sens is a
            sensitivity dict, this value will be disregarded.
        orifsn: which element of the sequence should be used for
            determining the origin. 0 means empty beam...
            Or you can give a tuple or a list of two: in this case these
            will be the coordinates of the origin and no auto-searching
            will be performed.
        mask: mask matrix. 1 for non-masked, 0 for masked
        energymeas: list of apparent energies, for energy calibration
        energycalib: list of theoretical energies, corresponding to the
            apparent energies.
        distminus: this will be subtracted from the sample-to-detector
            distance read from the measurement files, but only for
            samples, not for references
        detshift: this will be subtracted from the sample-to-detector
            distance read from all measurement files, including
            references!
        orig: helper data for the beam finding procedures. You have
            several possibilities:
            A) a vector/list/tuple of two: findbeam_sector() will be
                tried. In this case this is the initial value of the
                beam center
            B) a vector/list/tuple of four: xmin,xmax,ymin,ymax:
                the borders of the rectangle, around the beam, if a
                semitransparent beam-stop was used. In this case
                findbeam_semitransparent() will be tried, and the beam
                center will be determined for each measurement,
                independently (disregarding orifsn).
            C) a vector/list/tuple of five: Ntheta,dmin,dmax,bcx,bcy:
                findbeam_azimuthal will be used. Ntheta, dmin and dmax
                are the respective parameters for azimintpix(), while
                bcx and bcy are the x and y coordinates for the origin
                at the first guess.
            D) a mask matrix (1 means nonmasked, 0 means masked), the
                same size as that of the measurement data. In this
                case findbeam_gravity() will be used.
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

    #load measurement files
    data,header=B1io.read2dB1data(_B1config['2dfileprefix'],fsn1,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])

    #finding beamcenter
    B1findbeam(data,header,orifsn,orig,mask)

    print "B1integrate: doing energy calibration and correction for reference distance"
    for k in range(len(data)):
        if header[k]['Title']=='Reference_on_GC_holder_before_sample_sequence':
            header[k]['Dist']=header[k]['Dist']-distancetoreference-detshift
            print "Corrected sample-detector distance for fsn %d (ref. before)." % header[k]['FSN']
        elif header[k]['Title']=='Reference_on_GC_holder_after_sample_sequence':
            header[k]['Dist']=header[k]['Dist']-distancetoreference-detshift
            print "Corrected sample-detector distance for fsn %d (ref. after)." % header[k]['FSN']
        else:
            header[k]['Dist']=header[k]['Dist']-distminus-detshift
        header[k]['EnergyCalibrated']=fitting.energycalibration(energymeas,energycalib,header[k]['Energy'])
        print "Calibrated energy for FSN %d (%s): %f -> %f" %(header[k]['FSN'],header[k]['Title'],header[k]['Energy'],header[k]['EnergyCalibrated'])
        header[k]['XPixel']=pixelsize
        header[k]['YPixel']=pixelsize
    # subtract the background and the dark current, normalise by sensitivity and transmission
    Asub,errAsub,header,injectionEB = subtractbg(data,header,fsndc,sens,errorsens,transm)
    
    qs=[]
    ints=[]
    errs=[]
    Areas=[]
    As=[]
    Aerrs=[]
    headerout=[]
    print "Integrating data. Press Return after inspecting the images."
    for k in range(len(Asub)):
        if header[k]['Title']==_B1config['ebtitle']:
            continue
        header[k]['PixelSize']=pixelsize
        As.append(Asub[k])
        Aerrs.append(errAsub[k])
        pylab.clf()
        guitools.plot2dmatrix(Asub[k],None,mask,header[k],blacknegative=True)
        pylab.gcf().suptitle('FSN %d (%s) Corrected, log scale\nBlack: nonpositives; Faded: masked pixels' % (header[k]['FSN'],header[k]['Title']))
        #pylab.gcf().show()
        pylab.draw()
        #now do the integration
        print "Now integrating..."
        spam=time.time()
        q,I,e,A=utils2d.radintC(As[-1],Aerrs[-1],header[k]['EnergyCalibrated'],header[k]['Dist'],
                       header[k]['PixelSize'],header[k]['BeamPosX'],
                       header[k]['BeamPosY'],(1-mask).astype('uint8'))
        qs.append(q)
        ints.append(I)
        errs.append(e)
        Areas.append(A)
        headerout.append(header[k])
        print "...done. Integration took %f seconds" % (time.time()-spam)
        utils.pause() # we put pause here, so while the user checks the 2d data, the integration is carried out.
        pylab.clf()
        pylab.subplot(121)
        pylab.cla()
        #print qs[-1]
        #print ints[-1]
        valid=np.isfinite(ints[-1])&np.isfinite(errs[-1])
        pylab.errorbar(qs[-1][valid],ints[-1][valid],errs[-1][valid])
        pylab.axis('tight')
        pylab.xlabel('q (1/Angstrom)')
        pylab.ylabel('Intensity (arb. units)')
        pylab.xscale('log')
        pylab.yscale('log')
        pylab.title('FSN %d' % (header[k]['FSN']))
        pylab.subplot(122)
        pylab.cla()
        pylab.plot(qs[-1],Areas[-1],'.')
        pylab.xlabel('q (1/Angstrom)')
        pylab.ylabel('Effective area (pixels)')
        pylab.title(header[k]['Title'])
        pylab.gcf().show()
        utils.pause()
    return qs,ints,errs,Areas,As,Aerrs,headerout
def subtractbg(data,header,fsndc,sens,senserr,transm=None,oldversion=False):
    """Subtract dark current and empty beam from the measurements and
    carry out corrections for detector sensitivity, dead time and beam
    flux (monitor counter).
    
    Inputs:
        data: list of scattering image matrices
        header: list of header dictionaries
        fsndc: FSN for the dark current measurements. Can be a single
            integer number or a list of numbers. In the latter case the
            DC data are summed up. However, if sens is a sensitivity
            dict, this value will be disregarded.
        sens: sensitivity matrix, or a sensitivity dict.
        senserr: error of the sensitivity matrix. If sens is a
            sensitivity dict, this value will be disregarded.
        transm: if given, disregard the measured transmission of the
            sample.
        oldversion: do an old-style error propagation (an approximation
            only, not a true and mathematically exact error propagation)
    Outputs: Asub,errAsub,header,injectionEB
        Asub: the corrected matrix
        errAsub: the error of the corrected matrix
        header: header data
        injectionEB: 'y' if an injection between the empty beam and
            sample measurement occured. 'n' otherwise
    """
    global _B1config
    hackDCsub=1
    
    if type(sens)==type({}): # if sens is a new-type sensitivity dict
        ad=sens['a0']
        md=sens['m0']
        td=sens['t0']
        D=sens['D0']
        S=sens['sens']
        senserr=sens['errorsens']
    else: # the original way
        # load DC measurement files
        datadc,headerdc=B1io.read2dB1data(_B1config['2dfileprefix'],fsndc,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
        #sum up darkcurrent measurements, if more.
        # summarize transmission, anode, monitor and measurement time data
        # for dark current files
        ad=sum([h['Anode'] for h in headerdc])
        md=sum([h['Monitor'] for h in headerdc])
        td=sum([h['MeasTime'] for h in headerdc])    
        D=sum(datadc)
        S=sens
    dad=np.sqrt(ad)
    dmd=np.sqrt(md)
    dD=np.sqrt(D)
    # initialize the result lists to emptys
    Asub=[]
    errAsub=[]
    headerout=[]
    injectionEB=[]
    
    for k in range(len(data)): # for each measurement
        if header[k]['Title']==_B1config['ebtitle']: # if this is an empty beam measurement, skip.
            continue
        ebindex=None
        for i in range(len(header)): # find empty beam in sequence
            if header[i]['FSN']==header[k]['FSNempty']:
                ebindex=i
        if ebindex is None:
            print 'Cannot find all empty beam measurements.\nWhere is the empty FSN %d belonging to FSN %d? Ignoring.'% (header[k]['FSNempty'],header[k]['FSN'])
            continue

        if oldversion:
            # subtract dark current and normalize by sensitivity and transmission (1 in case of empty beam)
            Abg,Abgerr=subdc(databg,headerbg,datadc,headerdc,sens,senserr)
            # subtract dark current from scattering patterns and normalize by sensitivity and transmission
            A2,A2err=subdc([data[k]],[header[k]],datadc,headerdc,sens,senserr,transm)

            #two-theta
            tth=np.arctan(utils2d.calculateDmatrix(Ax,(header[k]['XPixel'],header[k]['YPixel']),header[k]['BeamPosX'],header[k]['BeamPosY'])/header[k]['Dist'])
            #angle-dependent corrections:
            C0=gasabsorptioncorrectiontheta(header[k]['EnergyCalibrated'],tth)*geomcorrectiontheta(tth,header[k]['Dist'])
            Ca=absorptionangledependenttth(tth,Ta)
            # subtract background, but first check if an injection occurred
            K=A2*C0*Ca-Abg*C0
            dK=np.sqrt(A2err**2*C0**2*Ca**2+Abgerr**2*C0**2)
        else: # new version
            A=data[k]
            dA=np.sqrt(A)
            B=data[ebindex]
            dB=np.sqrt(B)
            ta=header[k]['MeasTime']
            tb=header[ebindex]['MeasTime']
            aa=header[k]['Anode']
            daa=np.sqrt(aa)
            ab=header[ebindex]['Anode']
            dab=np.sqrt(ab)
            ma=header[k]['Monitor']
            dma=np.sqrt(ma)
            mb=header[ebindex]['Monitor']
            dmb=np.sqrt(mb)
            Ta=header[k]['Transm']
            dTa=0 # if the error of the transmission is not zero, please set it here.
            if (Ta<=0): # older measurements on the glassy carbon did not save the transmission.
                print ""
                print ""
                print "---------------------------------------------------------------------"
                print "VERY BIG, VERY FAT WARNING!!!"
                print "The transmission of this sample (",header[k]['Title'],") is nonpositive!"
                print "ASSUMING IT TO BE %f." % (_B1config['GCtransmission'])
                print "Note that this may foul the calibration into absolute intensity units!"
                print "----------------------------------------------------------------------"
                print ""
                utils.pause()
                Ta=_B1config['GCtransmission']
                header[k]['Transm']=Ta
            # <anything>x will be the DC corrected version of <anything>
            Ax=A-D*ta/td
            max=ma-md*ta/td 
            aax=aa-ad*ta/td
            Bx=B-D*tb/td
            mbx=mb-md*tb/td
            abx=ab-ad*tb/td
            if hackDCsub:
                print "Tampering with DC subtraction!"
                Ax[Ax<=0]=np.nanmin(Ax[Ax>0])
                Bx[Bx<=0]=np.nanmin(Bx[Bx>0])

#            print "DC corrected counts:"
#            print "ma:",max
#            print "mb:",mbx
#            print "aa:",aax
#            print "ab:",abx
#            print utils.matrixsummary(Ax,"A")
#            print utils.matrixsummary(Bx,"B")
            #two-theta for the pixels
            tth=np.arctan(utils2d.calculateDmatrix(Ax,(header[k]['XPixel'],header[k]['YPixel']),header[k]['BeamPosX'],header[k]['BeamPosY'])/header[k]['Dist'])
    
            #angle-dependent corrections:
            C0=gasabsorptioncorrectiontheta(header[k]['EnergyCalibrated'],tth)*geomcorrectiontheta(tth,header[k]['Dist'])
            Ca,dCa=absorptionangledependenttth(tth,Ta,diffaswell=True)
            
            #auxiliary variables:
            P=Ax/(Ta*max*S)*aax/np.nansum(Ax)*C0*Ca
            Q=-Bx/(mbx*S)*abx/np.nansum(Bx)*C0
#            print "Auxiliary matrices:"
#            print utils.matrixsummary(P,"P")
#            print utils.matrixsummary(Q,"Q")
            # K is the resulting matrix (corrected for dark current, 
            # lost anode counts (dead time), sensitivity, monitor,
            # transmission, background, and various angle-dependent errors)
            K=P+Q
            #now calculate the different error terms: ET_x is the contribution
            # to the error of K from x.
            ET_Ta=(-P/Ta+dCa/Ca)**2*dTa**2
            ET_ma=(P/max)**2*dma**2
            ET_mb=(Q/mbx)**2*dmb**2
            ET_md=(ta/td*P/max+tb/td*Q/mbx)**2*dmd**2
            ET_aa=(P/aax)**2*daa**2
            ET_ab=(Q/abx)**2*dab**2
            ET_ad=(ta/td*P/aax+tb/td*Q/abx)**2*dad**2
            ET_S=(K/S)**2*senserr**2

            ET_A=(P**2/Ax**2-2*P**2/(Ax*np.nansum(Ax)))*dA**2+P**2/np.nansum(Ax)**2*np.nansum(dA**2)
            ET_B=(Q**2/Bx**2-2*Q**2/(Bx*np.nansum(Bx)))*dB**2+Q**2/np.nansum(Bx)**2*np.nansum(dB**2)
            alpha=(ta/td*P/Ax+tb/td*Q/Bx)
            beta=(ta/td*P/np.nansum(Ax)+tb/td*Q/np.nansum(Bx))
           
#            print utils.matrixsummary(alpha,"alpha_dark")
#            print utils.matrixsummary(beta,"beta_dark")
            ET_D=alpha**2*dD**2+beta**2*np.nansum(dD**2)-2*alpha*beta*dD**2

            print "error analysis for sample %s" %header[k]['Title']
            print utils.matrixsummary(ET_Ta,"Transmission             ")
            print utils.matrixsummary(ET_ma,"Monitor (sample)         ")
            print utils.matrixsummary(ET_mb,"Monitor (empty beam)     ")
            print utils.matrixsummary(ET_md,"Monitor (dark current)   ")
            print utils.matrixsummary(ET_aa,"Anode (sample)           ")
            print utils.matrixsummary(ET_ab,"Anode (empty beam)       ")
            print utils.matrixsummary(ET_ad,"Anode (dark current)     ")
            print utils.matrixsummary(ET_S,"Sensitivity              ")
            print utils.matrixsummary(ET_A,"Scattering (sample)      ")
            print utils.matrixsummary(ET_B,"Scattering (empty beam)  ")
            print utils.matrixsummary(ET_D,"Scattering (dark current)")
            dK=np.sqrt(ET_Ta+ET_ma+ET_mb+ET_md+ET_aa+ET_ab+ET_ad+ET_S+ET_A+ET_B+ET_D)
            print utils.matrixsummary(K,"Corrected matrix         ")
            print utils.matrixsummary(dK,"Total error              ")
            print utils.matrixsummary(dK**2,"Squared total error      ")
            print utils.matrixsummary(dK/K,"Relative error           ")
            if type(sens)==type({}): # if sens is a new-type sensitivity dict
                #correction terms, accounting for the dependence of S and md,ad,D
                CT_md=-2*K/S*(ta/t0*P/max+tb/t0*Q/mbx)*1/sens['S1S']* \
                    (sens['chim']-S*np.nansum(sens['chim']))*dmd**2
                CT_ad=-2*K/S*(ta/t0*P/aax+tb/t0*Q/abx)*1/sens['S1S']* \
                    (sens['chia']-S*np.nansum(sens['chia']))*dad**2
                CT_D=2*K/(S*sens['S1S'])*(ta/t0*P/Ax+tb/t0*Q/Bx)*(S*np.nansum(sens['beta'])-sens['beta']+(1-S+ta/t0*P/np.nansum(Ax)+tb/t0*Q/np.nansum(Bx))*sens['alpha'])*dD**2-\
                    2*K/(S*sens['S1S'])*(ta/t0*P/np.nansum(Ax)+tb/t0*Q/np.nansum(Bx))*(S*np.nansum(sens['beta'])-sens['beta'])*np.nansum(dD**2) + \
                    2*K/(S*sens['S1S'])*(ta/t0*P/np.nansum(Ax)+tb/t0*Q/np.nansum(Bx))*S*np.nansum(sens['alpha']*dD**2)
                print "Correction terms:"
                print utils.matrixsummary(CT_ad,"Anode (dark current)     ")
                print utils.matrixsummary(CT_md,"Monitor (dark current)   ")
                print utils.matrixsummary(CT_D,"Scattering (dark current)")
                dK=np.sqrt(ET_Ta+ET_ma+ET_mb+ET_md+ET_aa+ET_ab+ET_ad+ET_S+ET_A+ET_B+ET_D+ CT_ad+CT_md+CT_D)
                print utils.matrixsummary(dK,"Total error")
                print utils.matrixsummmary(dK**2,"Squared total error")

            #normalize by beam size
            Bx=header[k]['XPixel'] # the empty beam should be measured with the same settings...
            By=header[k]['YPixel']
            K=K/(Bx*By)
            dK=dK/(Bx*By)
        # now K and dK are ready, either using the old or the new version.

        # check if injection occurred between the empty beam and the dark current measurement
        if header[k]['Current1']>header[ebindex]['Current2']:
            print "Possibly an injection between sample and its background:"
            B1io.getsamplenames(_B1config['2dfileprefix'],header[k]['FSN'],_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
            B1io.getsamplenames(_B1config['2dfileprefix'],header[k]['FSNempty'],_B1config['2dfilepostfix'],showtitles='no',dirs=_B1config['measdatadir'])
            print "Current in DORIS at the end of empty beam measurement %.2f." % header[ebindex]['Current2']
            print "Current in DORIS at the beginning of sample measurement %.2f." % header[k]['Current1']
            injectionEB.append('y')
        else:
            injectionEB.append('n')
        header[k]['injectionEB']=injectionEB[-1]
        Asub.append(K)
        errAsub.append(dK)
        header[k]['FSNdc']=fsndc
        headerout.append(header[k])
    return Asub, errAsub, headerout, injectionEB
def subdc(data,header,datadc,headerdc,sens,senserr,transm=None):
    """Carry out dead-time corrections (anode counts/sum) on 2D
        scattering data, subtract dark current, normalize by monitor
        counts and sensitivity, and finally by pixel size.
    
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
    if type(data)!=type([]):
        raise TypeError("Parameter <data> should be a list!")
    if type(header)!=type([]):
        raise TypeError("Parameter <header> should be a list!")
    if type(datadc)!=type([]):
        raise TypeError("Parameter <datadc> should be a list!")
    if type(headerdc)!=type([]):
        raise TypeError("Parameter <headerdc> should be a list!")
    
    if transm is None:
        transm=np.array([h['Transm'] for h in header])
        transmave=transm.mean() # average transmission
        transmerr=transm.std() # standard deviation of the transmission
    else:
        transmave=transm
        transmerr=0
    if transmave<=0:
        print ""
        print ""
        print "---------------------------------------------------------------------"
        print "VERY BIG, VERY FAT WARNING!!!"
        print "The transmission of this sample (",header[0]['Title'],") is nonpositive!"
        print "ASSUMING IT TO BE 0.5."
        print "Note that this may foul the calibration into absolute intensity units!"
        print ""
        print ""
        utils.pause()
        transmave=0.5
        for h in header:
            h['Transm']=transmave
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

    # add up scattering patterns
    A1=sum(data) # do not use np.sum()
    Adc=sum(datadc)
    
    # correct monitor counts with its dark current
    mo2=mo1-modc*meastime1/meastimedc
    an2=an1-andc*meastime1/meastimedc
    A2=A-Adc*meastime1/meastimedc

    mo2err=np.sqrt(mo1+modc*(meastime1/meastimedc)**2)
    an2err=np.sqrt(an1+andc*(meastime1/meastimedc)**2)
    A2err=np.sqrt(A1+Adc*(meastime1/meastimedc)**2)

    sumA2=A2.sum()
    # error of sumA2, not sum of error of A2.
    sumA2err=np.sqrt((A2err**2).sum())
    
    # summarized scattering pattern, subtracted the dark current,
    # normalized by the monitor counter and the sensitivity
    A3=A2/(sens*mo2)*an2/sumA2/transmave
    
    print "Sum/Total of dark current: %.2f %. Counts/s %.1f." % (100*Adc.sum()/andc,andc/meastimedc)
    print "Sum/Total before dark current correction: %.2f %. Counts on anode %.1f cps. Monitor %.1f cps." %(100*A.sum()/an1,an1/meastime1,mo2/meastime1)
    print "Sum/Total after dark current correction: %.2f %." % (100*sumA2/anA2)

    A3err=np.sqrt((A2err/(sens*mo2*transmave)*an2/sumA2)**2+ \
                  (senserr*A2/(sens**2*mo2*transmave)*an2/sumA2)**2+ \
                  (mo2err*A2/(sens*transmave*mo2**2)*an2/sumA2)**2+ \
                  (an2err*A2/(sens*mo2*transmave)/sumA2)**2+ \
                  (sumA2err*A2/(sens*mo2*transmave)*an2/sumA2**2)**2+ \
                  (transmerr*A2/(sens*mo2*transmave**2)*an2/sumA2)**2)
                     
    #normalize by beam size
    Bx=header[0]['XPixel']
    By=header[0]['YPixel']
    return A3/(Bx*By),errA3/(Bx*By)
