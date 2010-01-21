#unstable.py
import numpy as np
import pylab
import utils
import types
import scipy.stats.stats
import matplotlib.widgets
import B1io
import guitools
import asamacros
import fitting

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

