#-----------------------------------------------------------------------------
# Name:        B1macros.py
# Purpose:     Macros for data processing
#
# Author:      Andras Wacha
#
# Created:     2010/02/22
# RCS-ID:      $Id: B1macros.py $
# Copyright:   (c) 2010
# Licence:     GPLv2
#-----------------------------------------------------------------------------

import numpy as np
import pylab
import types
import B1io
import utils2d
import guitools
import utils
import os
import time
import fitting
import scipy.io

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

_B1config={'measdatadir':'.',
           'evaldatadir':'.',
           'calibdir':'.',
           'distancetoreference':219,
           'pixelsize':0.798,
           'detector':'Gabriel',
           '2dfileprefix':'ORG',
           '2dfilepostfix':'.DAT',
           'GCareathreshold':10,
           'GCintsthreshold':1,
           'detshift':0,
           'refdata':[{'thick':143e-4,'pos':129,'data':'GC155.dat'},
                      {'thick':508e-4,'pos':139,'data':'GC500.dat'},
                      {'thick':992e-4,'pos':159,'data':'GC1000.dat'},
                      {'thick':500e-4,'pos':152.30,'data':'GC500Guenter_invcm.dat'}],
            'refposprecision':0.5,
            'ebtitle':'Empty_beam',
            'GCtransmission':None,
            'energyprecision':1
           }


def setconfig(config):
    """Set _B1config dict.
    
    Input:
        config: B1config dict. For details, look at the source of B1macros.py
    """
    global _B1config
    _B1config=config
def getconfig():
    """Get _B1config dict.
    
    Output:
        the _B1config dictionary. For details, look at the source of B1macros.py
    """
    global _B1config
    return _B1config
def addfsns(fileprefix,fsns,fileend,fieldinheader=None,valueoffield=None,dirs=[]):
    """
    """
    data,header=B1io.read2dB1data(fileprefix,fsns,fileend,dirs=dirs)
    
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
        origx, origy: the centers of the beamstop.
        mask: a mask to apply (1 for valid, 0 for masked pixels). If None,
            a makemask() window will pop up for user interaction.
        savefile: a .npz file to save results to. None to skip saving.
        
    Outputs: sensdict,mask
        sens: a dictionary of the following fields:
            sens: the sensitivity matrix of the 2D detector, by which all
                measured data should be divided pointwise. The matrix is
                normalized to 1 on the average.
            errorsens: the calculated error of the sensitivity matrix
            chia, chim, alpha, beta, S1S: these values (4 matrices and
                a scalar) are needed for calculation of the correction
                terms (taking the dependence of a_0... and S into
                account)
        mask: mask matrix created by the user. This masks values where
            the sensitivity is invalid. This should be used as a base
            for further mask matrices.
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
    chia=diff # save it for returning
    
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
        origx, origy: the centers of the beamstop.
    
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
    qGC,intGC,errGC,AGC=utils2d.radint(As[referencenumber],
                               Aerrs[referencenumber],
                               header[referencenumber]['EnergyCalibrated'],
                               header[referencenumber]['Dist'],
                               header[referencenumber]['PixelSize'],
                               header[referencenumber]['BeamPosX'],
                               header[referencenumber]['BeamPosY'],
                               1-mask,
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
        pylab.xlabel(u'q (1/%c)' % 197)
        pylab.ylabel('Scattering cross-section (1/cm)')
        pylab.title('Reference FSN %d multiplied by %.2e, error percentage %.2f' %(header[referencenumber]['FSN'],mult,(errmult/mult*100)))
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
        orig1=None
        try:
            print "Finding beam, len(orig)=",len(orig)
            if len(orig)==2:
                print "Determining origin (by the 'slices' method) from file FSN %d %s" %(header[orifsn-1]['FSN'],header[orifsn-1]['Title'])
                orig1=utils2d.findbeam_slices(Asub[orifsn-1],orig,mask)
                print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                guitools.testorigin(Asub[orifsn-1],orig1,mask)
                utils.pause()
            elif len(orig)==5:
                print "Determining origin (by the 'azimuthal' method) from file FSN %d %s" %(header[orifsn-1]['FSN'],header[orifsn-1]['Title'])
                orig1=utils2d.findbeam_azimuthal(Asub[orifsn-1],orig[3:5],mask,Ntheta=orig[0],dmin=orig[1],dmax=orig[2])
                print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                guitools.testorigin(Asub[orifsn-1],orig1,mask,dmin=orig[1],dmax=orig[2])
                utils.pause()
            elif len(orig)==4:
                for k in range(len(Asub)):
                    print "Determining origin (by the 'semitransparent' method) for file FSN %d %s" %(header[k]['FSN'],header[k]['Title'])
                    orig1=utils2d.findbeam_semitransparent(Asub[k],orig)
                    print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                    header[k]['BeamPosX']=orig1[0]
                    header[k]['BeamPosY']=orig1[1]
                    guitools.testorigin(Asub[orifsn],orig1,mask)
                    utils.pause()
            elif orig.shape==Asub[orifsn-1].shape:
                print "Determining origin (by the 'gravity' method) from file FSN %d %s" %(header[orifsn-1]['FSN'],header[orifsn-1]['Title'])
                orig1=utils2d.findbeam_gravity(Asub[orifsn-1],orig)
                print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                guitools.testorigin(Asub[orifsn-1],orig1,mask)
                utils.pause()
        except:
            print "Finding the origin did not succeed"
            return [],[],[],[],[],[],[]
        
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
        if len(orig)!=4:
            header[k]['BeamPosX']=orig1[0]
            header[k]['BeamPosY']=orig1[1]
        header[k]['PixelSize']=pixelsize
        D=utils2d.calculateDmatrix(mask,pixelsize,header[k]['BeamPosX'],header[k]['BeamPosY'])
        tth=np.arctan(D/header[k]['Dist'])
        spatialcorr=geomcorrectiontheta(tth,header[k]['Dist'])
        absanglecorr=absorptionangledependenttth(tth,header[k]['Transm'])
        gasabsorptioncorr=gasabsorptioncorrectiontheta(header[k]['EnergyCalibrated'],tth)
        As.append(Asub[k]*spatialcorr*absanglecorr*gasabsorptioncorr)
        Aerrs.append(errAsub[k]*spatialcorr*absanglecorr*gasabsorptioncorr)
        pylab.clf()
        guitools.plot2dmatrix(Asub[k],None,mask,header[k],blacknegative=True)
        pylab.gcf().suptitle('FSN %d (%s) Corrected, log scale\nBlack: nonpositives; Faded: masked pixels' % (header[k]['FSN'],header[k]['Title']))
        #pylab.gcf().show()
        pylab.draw()
        #now do the integration
        print "Now integrating..."
        spam=time.time()
        q,I,e,A=utils2d.radint(As[-1],Aerrs[-1],header[k]['EnergyCalibrated'],header[k]['Dist'],
                       header[k]['PixelSize'],header[k]['BeamPosX'],
                       header[k]['BeamPosY'],1-mask)
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
        pylab.errorbar(qs[-1],ints[-1],errs[-1])
        pylab.axis('tight')
        pylab.xlabel(u'q (1/%c)' % 197)
        pylab.ylabel('Intensity (arb. units)')
        pylab.xscale('log')
        pylab.yscale('log')
        pylab.title('FSN %d' % (header[k]['FSN']))
        pylab.subplot(122)
        pylab.cla()
        pylab.plot(qs[-1],Areas[-1],'.')
        pylab.xlabel(u'q (1/%c)' %197)
        pylab.ylabel('Effective area (pixels)')
        pylab.title(header[k]['Title'])
        pylab.gcf().show()
        utils.pause()
    return qs,ints,errs,Areas,As,Aerrs,headerout
def geomcorrectiontheta(tth,dist):
    return dist**2/(np.cos(tth)**3)
def absorptionangledependenttth(tth,transm,diffaswell=False):
    """Create matrix for correction by angle-dependent X-ray absorption
    
    Inputs:
        tth: two-theta values
        transm: transmission (e^(-mu*d))
        diffaswell: set True if you want to calculate the derivative (with
            respect to trasm) as well
    Output: C, [dC]
        C: a matrix of the sape of tth, containing the correction factors for
        angle-dependent absorption. The scattering data should be multiplied
        by this.
        dC: the derivative of C, with respect of transm. Only returned if
            diffaswell was True.
    """
    mud=-np.log(transm);
    cor=np.ones(tth.shape)
    
    #cor[tth>0]=transm/((1/(1-1/np.cos(tth[tth>0]))/mud)*(np.exp(-mud/np.cos(tth[tth>0]))-np.exp(-mud)))
    cor[tth>0]=transm*mud*(1-1/np.cos(tth[tth>0]))/(np.exp(-mud/np.cos(tth[tth>0]))-np.exp(-mud))
    if diffaswell:
        K=1/np.cos(tth)
        dcor=np.zeros(tth.shape)
        dcor[tth>0]=(K[tth>0]-1)/(transm**K[tth>0]-transm)**2*(transm**K[tth>0]*np.log(transm)*(1-K[tth>0])+(transm**K[tth>0]-transm))
        return cor,dcor
    else:
        return cor
def gasabsorptioncorrectiontheta(energycalibrated,tth,components=None):
    """Create matrix for correction by absorption of the various elements of the
    X-ray scattering set-up
    
    Inputs:
        energycalibrated: the calibrated energy, in eV-s
        tth: matrix of two-theta values
        components: components of the flight path. Give None for default, which
            corresponds to the set-up of beamline B1 at Hasylab/DESY, with the
            Gabriel detector. You can supply a list of dictionaries. In each
            dictionary the following fields should be given:
            'name': the name of the element
            'thick' : the thickness, in mm
            'data' : the data file name (without the pathname) where the
                transmission data can be found. They are text files with two
                columns: first the energy, second the transmission.

    Outputs:
        the correction matrix of the same shape as tth. The scattering data
        should be multiplied by this one. If you are interested how this is
        accomplished, please look at the source code...
    """
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
    cor=np.ones(tth.shape)
    for c in components:
        c['travel']=c['thick']/np.cos(tth)
        spam=np.loadtxt("%s%s%s" % (_B1config['calibdir'],os.sep,c['data']))
        if energycalibrated<spam[:,0].min():
            tr=spam[0,1]
        elif energycalibrated>spam[:,0].max():
            tr=spam[0,-1]
        else:
            tr=np.interp(energycalibrated,spam[:,0],spam[:,1])
        c['mu']=-np.log(tr) # in 1/mm
        cor=cor/np.exp(-c['travel']*c['mu'])
    return cor
def subtractbg_old(fsn1,fsndc,sens,senserr,transm=None):
    """Subtract dark current and empty beam from the measurements and
    carry out corrections for detector sensitivity, dead time and beam
    flux (monitor counter). subdc() is called... NOTE: the error propagation
    of this is not very good, please consider using subtractbg().
    
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
    datadc,headerdc=B1io.read2dB1data(_B1config['2dfileprefix'],fsndc,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    data,header=B1io.read2dB1data(_B1config['2dfileprefix'],fsn1,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    Asub=[]
    errAsub=[]
    headerout=[]
    injectionEB=[]
    
    for k in range(len(data)): # for each measurement
        # read int empty beam measurement file
        [databg,headerbg]=B1io.read2dB1data(_B1config['2dfileprefix'],header[k]['FSNempty'],_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
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
            B1io.getsamplenames(_B1config['2dfileprefix'],header[k]['FSN'],_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
            B1io.getsamplenames(_B1config['2dfileprefix'],header[k]['FSNempty'],_B1config['2dfilepostfix'],showtitles='no',dirs=_B1config['measdatadir'])
            print "Current in DORIS at the end of empty beam measurement %.2f." % headerbg[0]['Current2']
            print "Current in DORIS at the beginning of sample measurement %.2f." % header[k]['Current1']
            injectionEB.append('y')
        else:
            injectionEB.append('n')
        header[k]['injectionEB']=injectionEB[-1]
        Asub.append(A2-Abg) # they were already normalised by the transmission
        errAsub.append(np.sqrt(A2err**2+Abgerr**2))
        header[k]['FSNdc']=fsndc
        headerout.append(header[k])
    return Asub, errAsub, headerout, injectionEB
def subtractbg(fsn1,fsndc,sens,senserr,transm=None):
    """Subtract dark current and empty beam from the measurements and
    carry out corrections for detector sensitivity, dead time and beam
    flux (monitor counter). This is a newer version, which does all the
    error propagation more correctly.
    
    Inputs:
        fsn1: list of file sequence numbers (or just one number)
        fsndc: FSN for the dark current measurements. Can be a single
            integer number or a list of numbers. In the latter case the
            DC data are summed up. However, if sens is a sensitivity
            dict, this value will be disregarded.
        sens: sensitivity matrix, or a sensitivity dict.
        senserr: error of the sensitivity matrix. If sens is a
            sensitivity dict, this value will be disregarded.
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
    datadc,headerdc=B1io.read2dB1data(_B1config['2dfileprefix'],fsndc,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    data,header=B1io.read2dB1data(_B1config['2dfileprefix'],fsn1,_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
    
    #sum up darkcurrent measurements, if more.
    # summarize transmission, anode, monitor and measurement time data
    # for dark current files
    if type(sens)==type({}):
        ad=sens['a0']
        md=sens['m0']
        td=sens['t0']
        D=sens['D0']
        S=sens['sens']
        senserr=sens['errorsens']
    else:
        ad=sum([h['Anode'] for h in headerdc])
        md=sum([h['Monitor'] for h in headerdc])
        td=sum([h['MeasTime'] for h in headerdc])    
        D=sum(datadc)
        S=sens

    Asub=[]
    errAsub=[]
    headerout=[]
    injectionEB=[]
    
    for k in range(len(data)): # for each measurement
        # read int empty beam measurement file
        if header[k]['Title']==_B1config['ebtitle']:
            continue
        [databg,headerbg]=B1io.read2dB1data(_B1config['2dfileprefix'],header[k]['FSNempty'],_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
        if len(databg)==0:
            print 'Cannot find all empty beam measurements.\nWhere is the empty FSN %d belonging to FSN %d? Ignoring.'% (header[k]['FSNempty'],header[k]['FSN'])
            continue    
        A=data[k]
        B=databg[0]
        ta=header[k]['MeasTime']
        tb=headerbg[0]['MeasTime']
        aa=header[k]['Anode']
        ab=headerbg[0]['Anode']
        ma=header[k]['Monitor']
        mb=headerbg[0]['Monitor']
        Ta=header[k]['Transm']
        dTa=0 # if it is not zero, please set it here.
        if (Ta<=0):
            print ""
            print ""
            print "---------------------------------------------------------------------"
            print "VERY BIG, VERY FAT WARNING!!!"
            print "The transmission of this sample (",header[k]['Title'],") is nonpositive!"
            print "ASSUMING IT TO BE %f." % (_B1config['GCtransmission'])
            print "Note that this may foul the calibration into absolute intensity units!"
            print ""
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

        #angle-dependent corrections:
        C0=gasabsorptioncorrectiontheta(energycalibrated,tth)*geomcorrectiontheta(tth,dist)
        
        #auxiliary variables:
        P=Ax/(Ta*max*S)*aa/np.nansum(Ax)*C0*Ca
        
        # K is the resulting matrix (corrected for dark current, 
        # lost anode counts (dead time), sensitivity, monitor,
        # transmission, background, and various angle-dependent errors)
        K=A1/(Ta*S*ma1)*(aa1/A1.sum())-B1/(S*mb1)*(ab1/B1.sum())
        # for the error propagation, calculate various derivatives. The
        # names of the variables might seem a bit cryptic, but they
        # aren't: dCdTa means simply (dC/dTa), a matrix of the same size
        # that of C
        dCdTa=-A1/(Ta*Ta*S*ma1)*aa1/A1.sum()
        dCdma=-A1/(Ta*S*ma1*ma1)*aa1/A1.sum()
        dCdmb=B1/(S*mb1*mb1)*ab1/B1.sum()
        dCdmd=A1/(Ta*S*ma1*ma1)*ta/td*aa1/A1.sum()-B1/(S*mb1*mb1)*tb/td*ab1/B1.sum()
        dCdaa=A1/(Ta*S*ma1)/A1.sum()
        dCdab=-B1/(S*mb1)/B1.sum()
        dCdad=-A1/(Ta*S*ma1)*(ta/td)/A1.sum()+B1/(S*mb1)*(tb/td)/B1.sum()
        dCdS=-A1/(Ta*S*S*ma1)*aa1/A1.sum()+B1/(S*S*mb1)*ab1/B1.sum()
        # the dependence of the error of C on DA, DB, DD is not trivial.
        # They can be calculated as:
        # DC_ij=sqrt(dC_ijdTa**2 + ... + sum_mn (dC_ijdA_mn)**2*DA_mn**2 + ... )
        # dCdA_mn = delta(i,m)*delta(j,n)/(Ta*S_ij*ma1)*aa1/A1.sum() - A1/(Ta*S_ij*ma1)*aa1/A1.sum()**2 =
        #         = aa1 / (Ta*S_ij*ma1*A1.sum())* (delta(i,m)*delta(j,n)-A1/A1.sum())
        # so sum_mn (dCdA_mn**2*dA_mn**2) = sum_mn(dCdA_mn**2*A_mn) = ... (see next line in code)
        dCdA2DA2=(aa1/(Ta*S*ma1*A1.sum()))**2*(A.sum()*(A1/A1.sum())**2-2*A1/A1.sum()*A+A) # the name means now (dC/dA)^2*DA*2
        # the error propagation of B is similar:
        dCdB2DB2=(ab1/(S*mb1*B1.sum()))**2*(B.sum()*(B1/B1.sum())**2-2*B1/B1.sum()*B+B)
        # the error propagation of D is just a little trickier:
        alpha=aa1*ta/(Ta*ma1*A1.sum())
        beta=ab1*tb/(mb1*B1.sum())
        
        dCdD2DD2=1/(td*S)**2*( (alpha*A1/A1.sum()-beta*B1/B1.sum())**2*D.sum() 
                                -2*(alpha*A1/A1.sum()-beta*B1/B1.sum())*(alpha-beta)*D +
                                (alpha-beta)**2 *D)
        print "error analysis for sample %s" %header[k]['Title']
        print "Transmission: %g +/- %g" %((dCdTa**2*dTa**2).mean(),(dCdTa**2*dTa**2).std())
        print "Monitor(Sample): %g +/- %g" %((dCdma**2*ma).mean(),(dCdma**2*ma).std())
        print "Monitor(Empty beam): %g +/- %g" %((dCdmb**2*mb).mean(),(dCdmb**2*mb).std())
        print "Monitor(Dark current): %g +/- %g" %((dCdmd**2*md).mean(),(dCdmd**2*md).std())
        print "Anode(Sample): %g +/- %g" %((dCdaa**2*aa).mean(),(dCdaa**2*aa).std())
        print "Anode(Empty beam): %g +/- %g" %((dCdab**2*ab).mean(),(dCdab**2*ab).std())
        print "Anode(Dark current): %g +/- %g" %((dCdad**2*ad).mean(),(dCdad**2*ad).std())
        print "Sensitivity: %g +/- %g" %((dCdS**2*senserr**2).mean(),(dCdS**2*senserr**2).std())
        print "Sample: %g +/- %g" %((dCdA2DA2).mean(),dCdA2DA2.std())
        print "Empty beam: %g +/- %g" %((dCdB2DB2).mean(),(dCdB2DB2).std())
        print "Dark current: %g +/- %g" %((dCdD2DD2).mean(),(dCdD2DD2).std())
        dC=np.sqrt(dCdTa**2*dTa**2 + dCdma**2*ma + dCdmb**2*mb + dCdmd**2*md + dCdS**2*senserr**2 +
                      dCdaa**2*aa + dCdab**2*ab + dCdad**2*ad + dCdA2DA2 + dCdB2DB2 + dCdD2DD2)
        print "Total error: %g +/- %g" % ((dC**2).mean(),(dC**2).std())
        #normalize by beam size
        Bx=header[k]['XPixel'] # the empty beam should be measured with the same settings...
        By=header[k]['YPixel']
        C=C/(Bx*By)
        dC=dC/(Bx*By)
        
        if header[k]['Current1']>headerbg[0]['Current2']:
            print "Possibly an injection between sample and its background:"
            B1io.getsamplenames(_B1config['2dfileprefix'],header[k]['FSN'],_B1config['2dfilepostfix'],dirs=_B1config['measdatadir'])
            B1io.getsamplenames(_B1config['2dfileprefix'],header[k]['FSNempty'],_B1config['2dfilepostfix'],showtitles='no',dirs=_B1config['measdatadir'])
            print "Current in DORIS at the end of empty beam measurement %.2f." % headerbg[0]['Current2']
            print "Current in DORIS at the beginning of sample measurement %.2f." % header[k]['Current1']
            injectionEB.append('y')
        else:
            injectionEB.append('n')
        header[k]['injectionEB']=injectionEB[-1]
        Asub.append(C)
        errAsub.append(dC)
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
        print "(sleeping for 5 seconds)"
        time.sleep(5)
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
    
    # correct monitor counts with its dark current
    monitor1corrected=mo1-modc*meastime1/meastimedc
    
    # add up scattering patterns
    A=sum(data) # do not use np.sum()
    Adc=sum(datadc)
    
    #subtract the dark current from the scattering pattern
    sumA2=(A-Adc*meastime1/meastimedc).sum()
    # error of sumA2, not sum of error of A2.
    sumA2err=np.sqrt((A+(meastime1/meastimedc)**2*Adc).sum())
    
    anA2=an1-andc*meastime1/meastimedc;
    anA2err=np.sqrt(an1+(meastime1/meastimedc)**2*andc)
    
    # summarized scattering pattern, subtracted the dark current,
    # normalized by the monitor counter and the sensitivity
    A2=(A-Adc*meastime1/meastimedc)/sens/monitor1corrected
    
    print "Sum/Total of dark current: %.2f. Counts/s %.1f." % (100*Adc.sum()/andc,andc/meastimedc)
    print "Sum/Total before dark current correction: %.2f. Counts on anode %.1f cps. Monitor %.1f cps." %(100*A.sum()/an1,an1/meastime1,monitor1corrected/meastime1)
    print "Sum/Total after dark current correction: %.2f." % (100*sumA2/anA2)
    errA=np.sqrt(A)
    errAdc=np.sqrt(Adc)
    errmonitor1corrected=mo1+modc*meastime1/meastimedc
    errA2=np.sqrt(1/(sens*monitor1corrected)**2*errA**2+
                     (meastime1/(meastimedc*sens*monitor1corrected))**2*errAdc**2+
                     (1/(monitor1corrected**2*sens)*(A-Adc*meastime1/meastimedc))**2*errmonitor1corrected**2+
                     (1/(monitor1corrected*sens**2)*(A-Adc*meastime1/meastimedc))**2*senserr**2)
                     
    A3=A2*anA2/(sumA2*transmave)
    errA3=np.sqrt((anA2/(sumA2*transmave)*errA2)**2+
                     (A2/(sumA2*transmave)*anA2err)**2+
                     (A2*anA2/(sumA2**2*transmave)*sumA2err)**2+
                     (A2*anA2/(sumA2*transmave**2)*transmerr)**2)
    #normalize by beam size
    Bx=header[0]['XPixel']
    By=header[0]['YPixel']
    return A3/(Bx*By),errA3/(Bx*By)
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
    a,b,aerr,berr=fitting.linfit(energymeas,energycalib)
    if type(energy1)==np.ndarray:
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
    qrange=np.array(qrange)
    if type(data)!=types.ListType:
        data=[data]
    data2=[];
    counter=0;
    for d in data:
        #print counter
        counter=counter+1
        tmp={};
        tmp['q']=qrange
        tmp['Intensity']=np.interp(qrange,d['q'],d['Intensity'])
        tmp['Error']=np.interp(qrange,d['q'],d['Error'])
        data2.append(tmp)
    return data2;
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
        A,Aerr,param=B1io.read2dintfile(fsn)
        if len(A)<1:
            continue
        waxsdata=B1io.readwaxscor(fsn)
        if len(waxsdata)<1:
            continue
        D=utils2d.calculateDmatrix(mask2d,param[0]['PixelSize'],param[0]['BeamPosX']-1,
                           param[0]['BeamPosY']-1)
        Dmax=D[mask2d!=0].max()
        qmax=4*np.pi*np.sin(0.5*np.arctan(Dmax/float(param[0]['Dist'])))*param[0]['EnergyCalibrated']/float(HC)
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
        [q,I,E,Area]=utils2d.radintC(A[0],Aerr[0],param[0]['EnergyCalibrated'],param[0]['Dist'],
                         param[0]['PixelSize'],param[0]['BeamPosX']-1,
                         param[0]['BeamPosY']-1,1-mask2d,qrange)
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
        mult,errmult=utils.multfactor(q,I,E,Iw,Ew)
        mult1=param[0]['NormFactor']
        errmult1=param[0]['NormFactorRelativeError']*mult1*0.01
        waxsdata[0]['Error']=np.sqrt((waxsdata[0]['Error']*mult)**2+
                                     (errmult*waxsdata[0]['Intensity'])**2)
        waxsdata[0]['Intensity']=waxsdata[0]['Intensity']*mult
        print 'mult: ',mult,'+/-',errmult
#        print 'mult1: ',mult1,'+/-',errmult1
        B1io.writeintfile(waxsdata[0]['q'],waxsdata[0]['Intensity'],waxsdata[0]['Error'],param[0],filetype='waxsscaled')
        [q,I,E,Area]=utils2d.radintC(A[0],Aerr[0],param[0]['EnergyCalibrated'],param[0]['Dist'],
                            param[0]['PixelSize'],param[0]['BeamPosX']-1,
                            param[0]['BeamPosY']-1,1-mask2d,q=np.linspace(0,qmax,np.sqrt(mask2d.shape[0]**2+mask2d.shape[1]**2)))
        pylab.figure()
        pylab.subplot(1,1,1)
        pylab.loglog(q,I,label='SAXS')
        pylab.loglog(waxsdata[0]['q'],waxsdata[0]['Intensity'],label='WAXS')
        pylab.legend()
        pylab.title('FSN %d: %s' % (param[0]['FSN'], param[0]['Title']))
        pylab.xlabel(u'q (1/%c)' % 197)
        pylab.ylabel('Scattering cross-section (1/cm)')
        pylab.savefig('scalewaxs%d.png' % param[0]['FSN'],dpi=300,transparent='True',format='png')
        pylab.close(pylab.gcf())
def reintegrateB1(fsnrange,mask,qrange=None,samples=None,savefiletype='intbinned',dirs=[]):
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
        savefiletype: the first part of the files to be saved. Default is
            'intbinned'
        dirs: directories for searching input files.
        
    Outputs:
        <savefiletype>*.dat files are saved to the disk.
        
    Note:
        the int2dnorm*.mat files along with the respective intnorm*.log files
        should reside in the current directory or one of the directories in 
        <dirs>
    """
    if qrange is not None:
        if type(qrange)!=types.ListType and type(qrange)!=np.ndarray:
            qrange=[qrange]
        qrange=np.array(qrange)
        original_qrange=qrange.copy(); # take a copy of it
    else:
        original_qrange=None
    if type(fsnrange)!=types.ListType:
        fsnrange=[fsnrange];
    params=B1io.readlogfile(fsnrange,dirs=dirs);
    if len(params)<1:
        return
    if samples is None:
        samples=utils.unique([p['Title'] for p in params]);
    if type(samples)!=types.ListType:
        samples=[samples]
    for s in samples:
        print 'Reintegrating measurement files for sample %s' % s
        sparams=[p for p in params if p['Title']==s];
        if len(sparams)<1:
            print 'No measurements of %s in the current sequence.' % s
            continue # with the next sample
        dists=utils.unique([p['Dist'] for p in sparams]);
        for d in dists:
            if original_qrange is None:
                qrange=None
            else:
                qrange=original_qrange[:];
            sdparams=[p for p in sparams if p['Dist']==d];
            print 'Evaluating measurements with distance %f' %d
            if qrange is not None:
                if (type(qrange) != types.ListType) and (type(qrange) != np.ndarray):
                    qrange=[qrange];
            if (qrange is None) or (len(qrange)<2) :
                print 'Generating common q-range'
                energymin=min([p['EnergyCalibrated'] for p in sdparams])
                energymax=max([p['EnergyCalibrated'] for p in sdparams])
                Y,X=np.meshgrid(np.arange(mask.shape[1]),np.arange(mask.shape[0]));
                D=np.sqrt((sdparams[0]['PixelSize']*(X-sdparams[0]['BeamPosX']-1))**2+
                            (sdparams[0]['PixelSize']*(Y-sdparams[0]['BeamPosY']-1))**2)
                Dlin=D[mask!=0]
                qmin=4*np.pi*np.sin(0.5*np.arctan(Dlin.min()/d))*energymax/HC;
                qmax=4*np.pi*np.sin(0.5*np.arctan(Dlin.max()/d))*energymin/HC;
                print 'Auto-determined qmin:',qmin
                print 'Auto-determined qmax:',qmax
                print 'qmin=4pi*sin(0.5*atan(Rmin/L))*energymax/HC'
                print 'qmax=4pi*sin(0.5*atan(Rmax/L))*energymin/HC'
                if qrange is None:
                    NQ=np.ceil((Dlin.max()-Dlin.min())/sdparams[0]['PixelSize']*2)
                    print 'Auto-determined number of q-bins:',NQ
                else:
                    NQ=qrange[0];
                    print 'Number of q-bins (as given by the user):',NQ
                qrange=np.linspace(qmin,qmax,NQ)
            for p in sdparams:
                print 'Loading 2d intensity for FSN %d' % p['FSN']
                data,dataerr,tmp=B1io.read2dintfile(p['FSN'],dirs=dirs);
                if len(data)<1:
                    continue
                print 'Re-integrating...'
                qs,ints,errs,areas=utils2d.radintC(data[0],dataerr[0],p['EnergyCalibrated'],
                                        p['Dist'],p['PixelSize'],p['BeamPosX']-1,
                                        p['BeamPosY']-1,1-mask,qrange);
                B1io.writeintfile(qs,ints,errs,p,areas,filetype=savefiletype)
                print 'done.'
                del data
                del dataerr
                del qs
                del ints
                del errs
                del areas
def sumfsns(fsns,samples=None,filetype='intnorm',waxsfiletype='waxsscaled',dirs=[]):
    """Summarize scattering data.
    
    Inputs:
        fsns: FSN range
        samples: samples to evaluate. Leave it None to auto-determine
        filetype: 1D SAXS filetypes (ie. the beginning of the file) to summarize. 
        waxsfiletype: WAXS filetypes (ie. the beginning of the file) to summarize.
        dirs: directories for searching input files.
    """
    if type(fsns)!=types.ListType:
        fsns=[fsns]
    params=B1io.readlogfile(fsns,dirs=dirs)
    if samples is None:
        samples=utils.unique([p['Title'] for p in params])
    if type(samples)!=types.ListType:
        samples=[samples]
    for s in samples:
        print 'Summing measurements for sample %s' % s
        sparams=[p for p in params if p['Title']==s]
        energies=utils.unique([p['Energy'] for p in sparams],lambda a,b:abs(a-b)<2)
        for e in energies:
            print 'Processing energy %f for sample %s' % (e,s)
            esparams=[p for p in sparams if abs(p['Energy']-e)<2]
            dists=utils.unique([p['Dist'] for p in esparams])
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
                    intdata=B1io.readintfile(filename,dirs=dirs)
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
                        if np.sum(q-intdata['q'])!=0:
                            print 'q-range of file %s differs from the others read before. Skipping.' % filename
                            continue
                        Isum=Isum+intdata['Intensity']/(intdata['Error']**2)
                        w=w+1/(intdata['Error']**2)
                    counter=counter+1
                if counter>0:
                    Esum=1/w
                    Esum[np.isnan(Esum)]=0
                    Isum=Isum/w
                    Isum[np.isnan(Isum)]=0
                    B1io.writeintfile(q,Isum,Esum,edsparams[0],filetype='summed')
                else:
                    print 'No files were found for summing.'
            waxscounter=0
            qwaxs=None
            Iwaxs=None
            wwaxs=None
            print 'Processing waxs files for energy %f for sample %s' % (e,s)
            for p in esparams:
                waxsfilename='%s%d.dat' % (waxsfiletype,p['FSN'])
                waxsdata=B1io.readintfile(waxsfilename,dirs=dirs)
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
                    if np.sum(qwaxs-waxsdata['q'])!=0:
                        print 'q-range of file %s differs from the others read before. Skipping.' % waxsfilename
                        continue
                    Iwaxs=Iwaxs+waxsdata['Intensity']/(waxsdata['Error']**2)
                    wwaxs=wwaxs+1/(waxsdata['Error']**2)
                waxscounter=waxscounter+1
            if waxscounter>0:
                Ewaxs=1/wwaxs
                Ewaxs[np.isnan(Ewaxs)]=0
                Iwaxs=Iwaxs/wwaxs
                Iwaxs[np.isnan(Iwaxs)]=0
                B1io.writeintfile(qwaxs,Iwaxs,Ewaxs,esparams[0],filetype='waxssummed')
            else:
                print 'No waxs file was found'