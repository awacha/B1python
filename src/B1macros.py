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
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#       
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#       
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

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
import re
import warnings
import sys
import string

try:
    import xlwt
except ImportError:
    pass

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

def normintBessy(fsn,filenameformat,mask,thicknesses,referencethickness,referenceindex,doffset,step2cm,resol,energyapp,energytheor,ref_qrange=None,qphirange=None,center_override=None,inttype='radial',int_aux=None,save_with_extension=None,noplot=False):
    """Normalize Bessy data according to glassy carbon measurements
    
    Inputs:
        fsn: range of file sequence numbers. Only give one ASAXS
            sequence at a time! The first fsn should be the first frame
            of the sequence.
        filenameformat: C-style format string for the filename without
            extension, e.g. 's%07u_000'
        mask: mask matrix. 0 is masked, 1 is unmasked
        thicknesses: a single thickness value for all measurements OR a
            dictionary, the keys being the sample names. In cm-s.
        referencethickness: thickness of the reference sample, in microns
        referenceindex: the index of the reference measurement in the
            sequence. 0 is the first measurement (usually empty beam).
        doffset: additive correction to the detector position read from
            the BDF file. In cm-s.
        step2cm: multiplicative correction to the detector position read
            from the BDF file. The true sample-to-detector distance is
            calculated: SD [cm] = detx*step2cm+doffset
        resol: resolution of the detector (pixel size), in mm/pixel
        energyapp: list of apparent energies
        energytheor: list of theoretical energies, corresponding to the
            apparent energies
        ref_qrange: q-range to be taken into account for the reference.
            Only the first and the last value of this will be used. Set
            it to None if the complete available reference dataset is to
            be used.
        qphirange: if a vector, the q- or theta-range onto which the
            integration is to be carried out. If an integer, the number
            of bins. In this case, the smallest and largest border is
            auto-determined. If None, the whole q- or phi-range will be
            auto-determined.
        center_override: if you want to set the beam center by hand. Leave
            it None if you want to use the default value (read from the
            bdf file).
        inttype: integration mode: 'r[adial]', 's[ector]' or
            'a[zimuthal]'. Case-insensitive.
        int_aux: auxiliary data for integration. If sector integration is
            preferred, then this should be a list of two. The first value
            is the starting azimuth angle, the second is the arc angle,
            both in radians. If azimuthal integration is requested, these
            are the smallest and largest q-values to be taken into account.
        save_with_extension: a string to append to the filename, just
            before the extension. If None, it will be the integration method.
        noplot: set this to True to suppress plotting.
    """
    GCareathreshold=10;
    
    if len(energyapp)!=len(energytheor):
        raise ValueError('You should supply as much apparent energies, as theoretical ones.')
    if len(energyapp)<2:
        print 'You should supply at least 2 different energy pairs to have a correct energy calibration! Now doing only a shift.'
    dat=[]
    fsnfound=[]
    print 'Loading header files...'
    for i in range(len(fsn)):
        bdfname='%s.bhf' %(filenameformat % (fsn[i]))
        try:
            dat1=B1io.bdf_read(bdfname);
        except IOError:
            print 'Cannot read file: %s, ignored.' %bdfname
            continue
        dat.append(dat1)
        fsnfound.append(fsn[i])
        if center_override is not None:
            dat[-1]['C']['xcen']=str(center_override[1])
            dat[-1]['C']['ycen']=str(center_override[0])
        print 'File %s loaded.'%bdfname
    print 'Done loading header files.'
    bgfsn=[]
    for i in range(len(dat)):
        try:
            bgfsn.append(dat[i]['C']['Background'])
        except KeyError:
            print 'No background set in header file for sample %s. This is what you want?' % dat[i]['C']['Sample']
    # now determine the length of an element of the sequence. This is done by
    # looking for repetition of the title of the 1st FSN.
    seqlen=len(dat) # initial value, if only one sequence exists.
    for i in range(1,len(dat)): # ignore the first
        if dat[i]['C']['Sample']==dat[0]['C']['Sample']:
            seqlen=i-1; 
            break
    print 'Sequence length is %u' % seqlen;
    nseq=float(len(dat))/seqlen
    if int(nseq)!=nseq:
        print 'Disregarding the last sequence, since it is incomplete.'
        nseq=np.floor(nseq);
    nseq=int(nseq) # force this, so the rest of the code won't complain about float indices
    #doing energy and distance correction
    dist=np.zeros(seqlen*nseq)
    energyreal=np.zeros(seqlen*nseq)
    for sequence in range(nseq): # evaluate each sequence
        print 'Correcting distances/energies for sequence %u/%u' %(sequence,nseq);
        for sample in range(seqlen): # evaluate each sample in the current sequence
            index=sequence*seqlen+sample;
            euncalib=float(dat[index]['M']['Energy'])
            duncalib=float(dat[index]['M']['Detector'])
            print 'Processing sample %u/%u (%s, %s)' %(sample,seqlen,dat[index]['C']['Frame'],dat[index]['C']['Sample'])
            dist[index]=(duncalib*step2cm+doffset)*10
            print 'Sample-detector distance: %f + %f = %f' % (duncalib,doffset,dist[index])
            energyreal[index]=fitting.energycalibration(energyapp,energytheor,euncalib)
            print 'Energy calibration: %f -> %f' %(euncalib,energyreal[index]);
    
    dist=utils.unique(dist);
    if len(dist)>1:
        raise RuntimeError,'Please give data measured with the same sample-to-detector distance!';
    dist=dist[0]
    print 'Distance is:',dist
    #processing argument resol.

    if type(resol)==type([]) or type(resol)==type(()) or type(resol)==np.ndarray:
        if len(resol)>2:
            print 'Wow! A %u dimensional detector :-) !' % len(resol);
        resx=resol[0];
        resy=resol[1];
    else:
        resx=resol;
        resy=resol;
    
    # determining common q-range if needed. First, try if qphirange is
    # a vector
    try:
        Nq=len(qphirange)
        # check if qphirange is a string (len() is also defined for strings)
        if type(qphirange)==type(''):
            raise TypeError # to get to the except clause
    except TypeError: # len() is not defined for qphirange, or qphirange is a string
        if qphirange is None: # fully automatic qphirange is to be determined
            if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                #choose value for azimuthal integration.
                raise NotImplementedError,"Determining the number of bins for azimuthal integration is not yet supported!"
            else: # radial or sector integration.
                Nq=max([dat[0]['xdim'],dat[0]['ydim']])/2.0
        else: # qphirange should be an integer, or a string-representation of an integer
            try:
                Nq=int(qphirange)
                qphirange=None # set it to None, to request auto-determination
            except ValueError:
                raise ValueError, "Invalid value for qphirange: %s" % repr(qphirange)
    # either way, we have Nq if we reached here.
       
    if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
        pass
    else:
        if qphirange is None: # common q-range should be generated for radial and sector integration
            maxenergy=max(energyreal);
            minenergy=min(energyreal);
            print 'Energy range: ',minenergy,'to',maxenergy,'eV'
            # in the BDF struct: X means the column direction, Y the row direction.
            bcx=float(dat[0]['C']['ycen'])+1
            bcy=float(dat[0]['C']['xcen'])+1
            D=utils2d.calculateDmatrix(mask,[resx,resy],bcx,bcy)[mask==1]
            print 'smallest distance from origin (pixels):',D.min()
            print 'largest distance from origin (pixels):',D.max()
            qmin=4*np.pi/HC*maxenergy*np.sin(0.5*np.arctan(D.min()/dist));
            qmax=4*np.pi/HC*minenergy*np.sin(0.5*np.arctan(D.max()/dist));
            qphirange=np.linspace(qmin,qmax,Nq);
            print 'Created common q-range: qmin: %f; qmax: %f; qstep: %f (%d points)' % \
                         (qmin,qmax,qphirange[1]-qphirange[0],Nq);
    for sequence in range(nseq): # evaluate each sequence
        print 'Evaluating sequence %u/%u' % (sequence,nseq);
        gcindex=sequence*seqlen+referenceindex;
        print 'Doing absolute calibration from sample %s' % dat[gcindex]['C']['Sample'];
        if referencethickness==1000:
            GCdata=scipy.io.loadmat("%s%s%s" % (_B1config['calibdir'],os.sep,'GCK1mm.mat'))['GCK1mm'];
            GCdata[:,0]=GCdata[:,0]*0.1; # convert q from 1/nm to 1/A
            thisreferencethickness=1020e-4;
        elif referencethickness==500:
            GCdata=scipy.io.loadmat("%s%s%s" % (_B1config['calibdir'],os.sep,'GCK500mkm.mat'))['GCK500mkm'];
            GCdata[:,0]=GCdata[:,0]*0.1;
            thisreferencethickness=485e-4;
        elif referencethickness==90:
            GCdata=scipy.io.loadmat("%s%s%s" % (_B1config['calibdir'],os.sep,'GCK90mkm.mat'))['GCK90mkm'];
            GCdata[:,0]=GCdata[:,0]*0.1;
            thisreferencethickness=90e-4;
        else:
            raise RuntimeError, 'Unknown reference thickness!';
        GCdata[:,1]=GCdata[:,1]*(0.28179e-12)**2; # scale intensities and errors into cm-1-s.
        GCdata[:,2]=GCdata[:,2]*(0.28179e-12)**2; # scale intensities and errors into cm-1-s.
        if ref_qrange is not None: # trim data
            GCindices=((GCdata[:,0]<=max(ref_qrange)) & (GCdata[:,0]>=min(ref_qrange)));
            GCdata=GCdata[GCindices,:];
        # now integrate the measured GC data to the q-range of GCdata
        print 'Glassy carbon reference: q_min=',np.min(GCdata[:,0]),'q_max=',np.max(GCdata[:,0]),'# of points=',len(GCdata[:,0])
        bdfname='%s.bdf' % (filenameformat % (fsnfound[gcindex]));
        print 'Loading GC data from file',bdfname;
        try:
            dat1=B1io.bdf_read(bdfname);
            if center_override is not None:
                if len(center_override)==2:
                    dat1['C']['xcen']=str(center_override[1])
                    dat1['C']['ycen']=str(center_override[0])
                else:
                    print "Finding origin."
                    orig=utils2d.findbeam_azimuthal(dat1['data'],(dat1['C']['ycen'],dat1['C']['xcen']),mask,center_override[0],center_override[1],center_override[2])
                    dat1['C']['xcen']=str(orig[1])
                    dat1['C']['ycen']=str(orig[0])
                    print "Found origin",orig[0],",",orig[1]
        except IOError:
            print "Cannot read reference file: %s. Skipping this sequence." % bdfname
            continue
        print 'Integrating GC data'
        tmp=time.time()
        [qGC,intGC,errGC,areaGC]=utils2d.radintC(dat1['data'].astype('double')/thisreferencethickness, \
                                        dat1['error'].astype('double')/thisreferencethickness, \
                                        energyreal[gcindex], \
                                        dist, \
                                        [resx,resy], \
                                        float(dat1['C']['ycen']), \
                                        float(dat1['C']['xcen']), \
                                        (1-mask).astype('uint8'),
                                        GCdata.astype('double')[:,0]);
        print 'Integration completed in %f seconds.' % (time.time()-tmp);
        print 'length of returned values:',len(qGC)
        goodindex=(areaGC>=GCareathreshold) & (np.isfinite(intGC)) & (np.isfinite(errGC))
        print 'valid q-bins (effective area > %f and both Intensity and its error are finite): %u' % (GCareathreshold,goodindex.sum())
        print '# of q-bins where the effective area is less than',GCareathreshold,':',(areaGC<GCareathreshold).sum()
        print '# of q-bins where either the intensity or its error is NaN:',(np.isnan(intGC) | np.isnan(errGC)).sum()
        print '# of q-bins where either the intensity or its error is infinite:',(np.isinf(intGC) | np.isinf(errGC)).sum()
        GCdata=GCdata[goodindex,:];
        qGC=qGC[goodindex,:];                                   
        intGC=intGC[goodindex,:];                                   
        errGC=errGC[goodindex,:];                                   
        areaGC=areaGC[goodindex,:];
        #Now the reference (GCdata) and the measured (qGC, intGC, errGC) data
        #are on the same q-scale, with unwanted pixels eliminated (see
        #ref_qrange). Thus we can calculate the multiplication factor by
        #integrating according to the trapezoid formula. Also, the measured
        #data have been divided by the thickness.
        mult,errmult=utils.multfactor(GCdata[:,0],GCdata[:,1],GCdata[:,2],intGC,errGC)
        if not noplot:
            pylab.clf()
            pylab.subplot(1,1,1)
            pylab.plot(GCdata[:,0],GCdata[:,1],'o');
            pylab.plot(qGC,intGC*mult,'.');
            pylab.xlabel('q (1/Angstrom)');
            pylab.legend(['Reference data for GC','Measured data for GC']);
            pylab.ylabel('Absolute intensity (1/cm)');
            #pylab.xscale('log')
            pylab.yscale('log')
            pylab.draw()
            pylab.savefig('GCcalib_%s.pdf' % save_with_extension, format='pdf')
            utils.pause();
        print 'Absolute calibration factor: %g +/- %g (= %g %%)' % (mult,errmult,errmult/mult*100);
        print 'Integrating samples';
        for sample in range(seqlen): # evaluate each sample in the current sequence
            index=sequence*seqlen+sample;
            if type(thicknesses)==type({}):
                try:
                    Sthick=thicknesses[dat[index]['C']['Sample']]
                except KeyError:
                    print 'Cannot find thickness for sample %s in thicknesses! Disregarding this sample.' % dat[index]['C']['Sample'];
            else:
                Sthick=thicknesses;
            print 'Using thickness %f (cm) for sample %s' % (Sthick,dat[index]['C']['Sample']);
            bdfname='%s.bdf' % (filenameformat % (fsnfound[index]));
            print 'Loading file %s' % bdfname;
            try:
                dat1=B1io.bdf_read(bdfname);
                if center_override is not None:
                    if len(center_override)==2:
                        dat1['C']['xcen']=str(center_override[1])
                        dat1['C']['ycen']=str(center_override[0])
                    else:
                        print "Finding origin."
                        orig=utils2d.findbeam_azimuthal(dat1['data'],(dat1['C']['ycen'],dat1['C']['xcen']),mask,center_override[0],center_override[1],center_override[2])
                        dat1['C']['xcen']=str(orig[1])
                        dat1['C']['ycen']=str(orig[0])
                        print "Found origin",orig[0],",",orig[1]
            except IOError:
                print 'Cannot read file: %s. Skipping.' % (bdfname);
                continue;
            print 'Converting to B1 format...'
            data,dataerr,header=B1io.bdf2B1(dat1,doffset,step2cm,energyreal[index],\
                        re.findall('[0-9]+',dat[gcindex]['C']['Frame'])[0],\
                        thisreferencethickness,0.5*(resx+resy),mult,errmult,Sthick)
            print 'Integrating sample %u/%u (%s, %s)...' %(sample,seqlen,dat1['bdf'],dat1['C']['Sample']);
            tmp=time.time()
            if inttype.upper()=='RADIAL'[:len(inttype)]:
                qs1,ints1,errs1,areas1,mask1=utils2d.radintC(dat1['data'].astype('double')/Sthick,\
                                            dat1['error'].astype('double')/Sthick,\
                                            energyreal[index],\
                                            dist,\
                                            [resx,resy],\
                                            float(dat1['C']['ycen']),\
                                            float(dat1['C']['xcen']),\
                                            (1-mask).astype('uint8'),\
                                            qphirange.astype('double'),\
                                            returnavgq=True,returnmask=True);
                if save_with_extension is None:
                    save_with_extension='radial'
            elif inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                qs1,ints1,errs1,areas1,mask1=utils2d.azimintqC(dat1['data'].astype('double')/Sthick,\
                                            dat1['error'].astype('double')/Sthick,\
                                            energyreal[index],\
                                            dist,\
                                            [resx,resy],\
                                            [float(dat1['C']['ycen']),\
                                            float(dat1['C']['xcen'])],\
                                            (1-mask).astype('uint8'),
                                            Nq,qmin=int_aux[0],qmax=int_aux[1],returnmask=True);
                if save_with_extension is None:
                    save_with_extension='azimuthal'
            elif inttype.upper()=='SECTOR'[:len(inttype)]:
                qs1,ints1,errs1,areas1,mask1=utils2d.radintC(dat1['data'].astype('double')/Sthick,\
                                            dat1['error'].astype('double')/Sthick,\
                                            energyreal[index],\
                                            dist,\
                                            [resx,resy],\
                                            float(dat1['C']['ycen']),\
                                            float(dat1['C']['xcen']),\
                                            (1-mask).astype('uint8'),\
                                            qphirange.astype('double'),\
                                            phi0=int_aux[0],dphi=int_aux[1],\
                                            returnavgq=True,returnmask=True);
                if save_with_extension is None:
                    save_with_extension='sector'
            else:
                raise ValueError,'Invalid integration mode: %s',inttype
            print 'Integration completed in %f seconds.' %(time.time()-tmp);
            errs1=np.sqrt(ints1**2*errmult**2+mult**2*errs1**2);
            ints1=ints1*mult;
            outname='%s_%s.dat'%(dat[index]['C']['Frame'][:-4],save_with_extension);
            B1io.write1dsasdict({'q':qs1,'Intensity':ints1,'Error':errs1},outname)
            if not noplot:
                pylab.clf()
                pylab.subplot(2,2,1);
                guitools.plot2dmatrix((dat1['data']),mask=1-mask1,header=header,showqscale=True)
                pylab.subplot(2,2,2);
                pylab.cla();
                validindex=(np.isfinite(ints1) & np.isfinite(errs1))
                if (1-validindex).sum()>0:
                    print "WARNING!!! Some nonfinite points are present among the integrated values!"
                    print "NaNs: ",(np.isnan(ints1) | np.isnan(errs1)).sum()
                    print "Infinities: ",(np.isinf(ints1) | np.isinf(errs1)).sum()
                if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                    pylab.plot(qs1[validindex],ints1[validindex])
                    pylab.xlabel(u'theta (rad)');
                    pylab.xscale('linear')
                    pylab.yscale('log')
                else:
                    pylab.plot(qphirange[validindex],ints1[validindex])
                    #pylab.errorbar(qphirange[validindex],ints1[validindex],errs1[validindex]);
                    pylab.xlabel('q (1/Angstrom)');
                    pylab.xscale('log')
                    pylab.yscale('log')
                pylab.ylabel('Absolute intensity (1/cm)');
                pylab.title('%s (%s)' % (dat[index]['C']['Frame'],dat[index]['C']['Sample']));
                pylab.axis('auto')

                pylab.subplot(2,2,3);
                if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                    pylab.plot(qs1,areas1)
                    pylab.xlabel('theta (rad)')
                else:
                    pylab.plot(qphirange,areas1);
                    pylab.xlabel('q (1/Angstrom)');
                pylab.ylabel('Effective area');
                if (inttype.upper()=='SECTOR'[:len(inttype)]) or (inttype.upper()=='RADIAL'[:len(inttype)]):
                    pylab.subplot(2,2,4);
                    pylab.plot(qphirange,qs1/qphirange,'.',markersize=3)
                    pylab.xlabel('Original q values')
                    pylab.ylabel('Averaged / original q-values')
                pylab.draw()
                pylab.gcf().show()
                pylab.savefig('int_%u_%s_%s.pdf' % (int(re.findall('[0-9]+',dat1['C']['Frame'])[0]),dat1['C']['Sample'],save_with_extension), format='pdf')
                utils.pause()
                pylab.clf();



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
            'energyprecision':1,
            'orgfileformat':'ORG%05u.DAT',
            'headerfileformat':'org_%05u.header',
            'intnormfileformat':'intnorm%u.dat',
            'intnormlogfileformat':'intnorm%u.log',
            'int2dnormfileformat':'int2dnorm%u.mat',
            'int2dnorm_intfileformat':'int2dnorm%u.dat',
            'int2dnorm_errfileformat':'err2dnorm%u.dat'
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

def reintegrateBessy(fsn,filenameformat,mask,thicknesses,referencethickness,referenceindex,doffset,step2cm,resol,energyapp,energytheor,ref_qrange=None,qphirange=None,center_override=None,inttype='radial',int_aux=None,save_with_extension=None,noplot=False):
    """Re-integrate 2d corrected SAXS patterns to q-bins.
    
    Inputs:
        fsn: range of file sequence numbers. Only give one ASAXS
            sequence at a time! The first fsn should be the first frame
            of the sequence.
        filenameformat: C-style format string for the filename without
            extension, e.g. 's%07u_000'
        mask: mask matrix. 0 is masked, 1 is unmasked
        thicknesses: a single thickness value for all measurements OR a
            dictionary, the keys being the sample names. In cm-s.
        referencethickness: thickness of the reference sample, in microns
        referenceindex: the index of the reference measurement in the
            sequence. 0 is the first measurement (usually empty beam).
        doffset: additive correction to the detector position read from
            the BDF file. In cm-s.
        step2cm: multiplicative correction to the detector position read
            from the BDF file. The true sample-to-detector distance is
            calculated: SD [cm] = detx*step2cm+doffset
        resol: resolution of the detector (pixel size), in mm/pixel
        energyapp: list of apparent energies
        energytheor: list of theoretical energies, corresponding to the
            apparent energies
        ref_qrange: q-range to be taken into account for the reference.
            Only the first and the last value of this will be used. Set
            it to None if the complete available reference dataset is to
            be used.
        qphirange: if a vector, the q- or theta-range onto which the
            integration is to be carried out. If an integer, the number
            of bins. In this case, the smallest and largest border is
            auto-determined. If None, the whole q- or phi-range will be
            auto-determined.
        center_override: if you want to set the beam center by hand. Leave
            it None if you want to use the default value (read from the
            bdf file).
        inttype: integration mode: 'r[adial]', 's[ector]' or
            'a[zimuthal]'. Case-insensitive.
        int_aux: auxiliary data for integration. If sector integration is
            preferred, then this should be a list of two. The first value
            is the starting azimuth angle, the second is the arc angle,
            both in radians. If azimuthal integration is requested, these
            are the smallest and largest q-values to be taken into account.
        save_with_extension: a string to append to the filename, just
            before the extension. If None, it will be the integration method.
        noplot: set this to True to suppress plotting.
    """
    GCareathreshold=getconfig()['GCareathreshold'];
    
    if len(energyapp)!=len(energytheor):
        raise ValueError('You should supply as much apparent energies, as theoretical ones.')
    if len(energyapp)<2:
        print 'You should supply at least 2 different energy pairs to have a correct energy calibration! Now doing only a shift.'
    dat=[]
    print 'Loading header files...'
    fsnfound=[]
    for i in range(len(fsn)):
        bdfname='%s.bhf' %(filenameformat % (fsn[i]))
        try:
            dat1=B1io.bdf_read(bdfname);
        except IOError:
            print 'Cannot read file: %s, ignored.' % bdfname
            continue
        dat.append(dat1)
        if center_override is not None:
            dat[-1]['C']['xcen']=str(center_override[1])
            dat[-1]['C']['ycen']=str(center_override[0])
        fsnfound.append(fsn[i])
        print 'File %s loaded.'%bdfname
    print 'Done loading header files.'
    # now determine the length of an element of the sequence. This is done by
    # looking for repetition of the title of the 1st FSN.
    seqlen=len(dat) # initial value, if only one sequence exists.
    for i in range(1,len(dat)): # ignore the first
        if dat[i]['C']['Sample']==dat[0]['C']['Sample']:
            seqlen=i-1; 
            break
    print 'Sequence length is %u' % seqlen;
    nseq=float(len(dat))/seqlen # number of sequences
    if int(nseq)!=nseq: # not integer: some (usually the last) sequence is incomplete.
        print 'Disregarding the last sequence, since it is incomplete.'
    nseq=int(nseq) # force this, so the rest of the code won't complain about float indices
    #doing energy and distance correction
    dist=np.zeros(seqlen*nseq)
    energyreal=np.zeros(seqlen*nseq)
    for sequence in range(nseq): # evaluate each sequence
        print 'Correcting distances/energies for sequence %u/%u' %(sequence,nseq);
        for sample in range(seqlen): # evaluate each sample in the current sequence
            index=sequence*seqlen+sample;
            euncalib=float(dat[index]['M']['Energy'])
            duncalib=float(dat[index]['M']['Detector'])
            print 'Processing sample %u/%u (%s, %s)' %(sample,seqlen,dat[index]['C']['Frame'],dat[index]['C']['Sample'])
            dist[index]=(duncalib*step2cm+doffset)*10
            print 'Sample-detector distance: %f + %f = %f' % (duncalib,doffset,dist[index])
            if len(energyapp)==1:
                energyreal[index]=euncalib+energytheor[0]-energyapp[0];
            else:
                energyreal[index]=fitting.energycalibration(energyapp,energytheor,euncalib)
            print 'Energy calibration: %f -> %f' %(euncalib,energyreal[index]);
    
    dist=utils.unique(dist);
    if len(dist)>1:
        raise RuntimeError,'Please give data measured with the same sample-to-detector distance!';
    dist=dist[0]
    print 'Distance is:',dist
    #processing argument resol.

    if type(resol)==type([]) or type(resol)==type(()) or type(resol)==np.ndarray:
        if len(resol)>2:
            print 'Wow! A %u dimensional detector :-) !' % len(resol);
        resx=resol[0];
        resy=resol[1];
    else:
        resx=resol;
        resy=resol;
    
    # determining common q-range if needed. First, try if qphirange is
    # a vector
    try:
        Nq=len(qphirange)
        # check if qphirange is a string (len() is also defined for strings)
        if type(qphirange)==type(''):
            raise TypeError # to get to the except clause
    except TypeError: # len() is not defined for qphirange, or qphirange is a string
        if qphirange is None: # fully automatic qphirange is to be determined
            if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                #choose value for azimuthal integration.
                raise NotImplementedError,"Determining the number of bins for azimuthal integration is not yet supported!"
            else: # radial or sector integration.
                Nq=max([dat[0]['xdim'],dat[0]['ydim']])/2.0
        else: # qphirange should be an integer, or a string-representation of an integer
            try:
                Nq=int(qphirange)
                qphirange=None # set it to None, to request auto-determination
            except ValueError:
                raise ValueError, "Invalid value for qphirange: %s" % repr(qphirange)
    # either way, we have Nq if we reached here.
       
    if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
        pass
    else:
        if qphirange is None: # common q-range should be generated for radial and sector integration
            maxenergy=max(energyreal);
            minenergy=min(energyreal);
            print 'Energy range: ',minenergy,'to',maxenergy,'eV'
            # in the BDF struct: X means the column direction, Y the row direction.
            bcx=float(dat[0]['C']['ycen'])+1
            bcy=float(dat[0]['C']['xcen'])+1
            D=utils2d.calculateDmatrix(mask,[resx,resy],bcx,bcy)[mask==1]
            print 'smallest distance from origin (pixels):',D.min()
            print 'largest distance from origin (pixels):',D.max()
            qmin=4*np.pi/HC*maxenergy*np.sin(0.5*np.arctan(D.min()/dist));
            qmax=4*np.pi/HC*minenergy*np.sin(0.5*np.arctan(D.max()/dist));
            qphirange=np.linspace(qmin,qmax,Nq);
            print 'Created common q-range: qmin: %f; qmax: %f; qstep: %f (%d points)' % \
                         (qmin,qmax,qphirange[1]-qphirange[0],Nq);
#    qs=[]; ints=[]; errs=[]; areas=[];
    for sequence in range(nseq): # evaluate each sequence
        print 'Evaluating sequence %u/%u' % (sequence,nseq);
        gcindex=sequence*seqlen+referenceindex;
        print 'Doing absolute calibration from sample %s' % dat[gcindex]['C']['Sample'];
        if referencethickness==1000:
            GCdata=scipy.io.loadmat("%s%s%s" % (_B1config['calibdir'],os.sep,'GCK1mm.mat'))['GCK1mm'];
            GCdata[:,0]=GCdata[:,0]*0.1; # convert q from 1/nm to 1/A
            thisreferencethickness=1020e-4;
        elif referencethickness==500:
            GCdata=scipy.io.loadmat("%s%s%s" % (_B1config['calibdir'],os.sep,'GCK500mkm.mat'))['GCK500mkm'];
            GCdata[:,0]=GCdata[:,0]*0.1;
            thisreferencethickness=485e-4;
        elif referencethickness==90:
            GCdata=scipy.io.loadmat("%s%s%s" % (_B1config['calibdir'],os.sep,'GCK90mkm.mat'))['GCK90mkm'];
            GCdata[:,0]=GCdata[:,0]*0.1;
            thisreferencethickness=90e-4;
        else:
            raise RuntimeError, 'Unknown reference thickness!';
        GCdata[:,1]=GCdata[:,1]*(0.28179e-12)**2; # scale intensities and errors into cm-1-s.
        GCdata[:,2]=GCdata[:,2]*(0.28179e-12)**2; # scale intensities and errors into cm-1-s.
        if ref_qrange is not None: # trim data
            GCindices=((GCdata[:,0]<=max(ref_qrange)) & (GCdata[:,0]>=min(ref_qrange)));
            GCdata=GCdata[GCindices,:];
        # now integrate the measured GC data to the q-range of GCdata
        print 'Glassy carbon reference: q_min=',np.min(GCdata[:,0]),'q_max=',np.max(GCdata[:,0]),'# of points=',len(GCdata[:,0])
        bdfname='%s.bdf' % (filenameformat % (fsnfound[gcindex]));
        print 'Loading GC data from file',bdfname;
        try:
            dat1=B1io.bdf_read(bdfname);
            if center_override is not None:
                if len(center_override)==2:
                    dat1['C']['xcen']=str(center_override[1])
                    dat1['C']['ycen']=str(center_override[0])
                else:
                    print "Finding origin."
                    orig=utils2d.findbeam_azimuthal(dat1['data'],(dat1['C']['ycen'],dat1['C']['xcen']),mask,center_override[0],center_override[1],center_override[2])
                    dat1['C']['xcen']=str(orig[1])
                    dat1['C']['ycen']=str(orig[0])
                    print "Found origin",orig[0],",",orig[1]
        except IOError:
            print "Cannot read reference file: %s. Skipping this sequence." % bdfname
            continue
        print 'Integrating GC data'
        tmp=time.time()
        [qGC,intGC,errGC,areaGC]=utils2d.radintC(dat1['data'].astype('double')/thisreferencethickness, \
                                        dat1['error'].astype('double')/thisreferencethickness, \
                                        energyreal[gcindex], \
                                        dist, \
                                        [resx,resy], \
                                        float(dat1['C']['ycen']), \
                                        float(dat1['C']['xcen']), \
                                        (1-mask).astype('uint8'),
                                        GCdata.astype('double')[:,0]);
        print 'Integration completed in %f seconds.' % (time.time()-tmp);
        print 'length of returned values:',len(qGC)
        goodindex=(areaGC>=GCareathreshold) & (np.isfinite(intGC)) & (np.isfinite(errGC))
        print 'valid q-bins (effective area > %f and both Intensity and its error are finite): %u' % (GCareathreshold,goodindex.sum())
        print '# of q-bins where the effective area is less than',GCareathreshold,':',(areaGC<GCareathreshold).sum()
        print '# of q-bins where either the intensity or its error is NaN:',(np.isnan(intGC) | np.isnan(errGC)).sum()
        print '# of q-bins where either the intensity or its error is infinite:',(np.isinf(intGC) | np.isinf(errGC)).sum()
        GCdata=GCdata[goodindex,:];
        qGC=qGC[goodindex,:];                                   
        intGC=intGC[goodindex,:];                                   
        errGC=errGC[goodindex,:];                                   
        areaGC=areaGC[goodindex,:];
        #Now the reference (GCdata) and the measured (qGC, intGC, errGC) data
        #are on the same q-scale, with unwanted pixels eliminated (see
        #ref_qrange). Thus we can calculate the multiplication factor by
        #integrating according to the trapezoid formula. Also, the measured
        #data have been divided by the thickness.
        mult,errmult=utils.multfactor(GCdata[:,0],GCdata[:,1],GCdata[:,2],intGC,errGC)
        if not noplot:
            pylab.clf()
            pylab.subplot(1,1,1)
            pylab.plot(GCdata[:,0],GCdata[:,1],'o');
            pylab.plot(qGC,intGC*mult,'.');
            pylab.xlabel('q (1/Angstrom)');
            pylab.legend(['Reference data for GC','Measured data for GC']);
            pylab.ylabel('Absolute intensity (1/cm)');
            #pylab.xscale('log')
            pylab.yscale('log')
            pylab.draw()
            pylab.savefig('GCcalib_%s.pdf' % save_with_extension, format='pdf')
            utils.pause();
        print 'Absolute calibration factor: %g +/- %g (= %g %%)' % (mult,errmult,errmult/mult*100);
        print 'Integrating samples';
        for sample in range(seqlen): # evaluate each sample in the current sequence
            index=sequence*seqlen+sample;
            if type(thicknesses)==type({}):
                try:
                    Sthick=thicknesses[dat[index]['C']['Sample']]
                except KeyError:
                    print 'Cannot find thickness for sample %s in thicknesses! Disregarding this sample.' % dat[index]['C']['Sample'];
            else:
                Sthick=thicknesses;
            print 'Using thickness %f (cm) for sample %s' % (Sthick,dat[index]['C']['Sample']);
            bdfname='%s.bdf' % (filenameformat % (fsnfound[index]));
            print 'Loading file %s' % bdfname;
            try:
                dat1=B1io.bdf_read(bdfname);
                if center_override is not None:
                    if len(center_override)==2:
                        dat1['C']['xcen']=str(center_override[1])
                        dat1['C']['ycen']=str(center_override[0])
                    else:
                        print "Finding origin."
                        orig=utils2d.findbeam_azimuthal(dat1['data'],(dat1['C']['ycen'],dat1['C']['xcen']),mask,center_override[0],center_override[1],center_override[2])
                        dat1['C']['xcen']=str(orig[1])
                        dat1['C']['ycen']=str(orig[0])
                        print "Found origin",orig[0],",",orig[1]
            except IOError:
                print 'Cannot read file: %s. Skipping.' % (bdfname);
                continue;
            print 'Converting to B1 format...'
            data,dataerr,header=B1io.bdf2B1(dat1,doffset,step2cm,energyreal[index],\
                        re.findall('[0-9]+',dat[gcindex]['C']['Frame'])[0],\
                        thisreferencethickness,0.5*(resx+resy),mult,errmult,Sthick)
            print 'Integrating sample %u/%u (%s, %s)...' %(sample,seqlen,dat1['bdf'],dat1['C']['Sample']);
            tmp=time.time()
            if inttype.upper()=='RADIAL'[:len(inttype)]:
                qs1,ints1,errs1,areas1,mask1=utils2d.radintC(dat1['data'].astype('double')/Sthick,\
                                            dat1['error'].astype('double')/Sthick,\
                                            energyreal[index],\
                                            dist,\
                                            [resx,resy],\
                                            float(dat1['C']['ycen']),\
                                            float(dat1['C']['xcen']),\
                                            (1-mask).astype('uint8'),\
                                            qphirange.astype('double'),\
                                            returnavgq=True,returnmask=True);
                if save_with_extension is None:
                    save_with_extension='radial'
            elif inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                qs1,ints1,errs1,areas1,mask1=utils2d.azimintqC(dat1['data'].astype('double')/Sthick,\
                                            dat1['error'].astype('double')/Sthick,\
                                            energyreal[index],\
                                            dist,\
                                            [resx,resy],\
                                            [float(dat1['C']['ycen']),\
                                            float(dat1['C']['xcen'])],\
                                            (1-mask).astype('uint8'),
                                            Nq,qmin=int_aux[0],qmax=int_aux[1],returnmask=True);
                if save_with_extension is None:
                    save_with_extension='azimuthal'
            elif inttype.upper()=='SECTOR'[:len(inttype)]:
                qs1,ints1,errs1,areas1,mask1=utils2d.radintC(dat1['data'].astype('double')/Sthick,\
                                            dat1['error'].astype('double')/Sthick,\
                                            energyreal[index],\
                                            dist,\
                                            [resx,resy],\
                                            float(dat1['C']['ycen']),\
                                            float(dat1['C']['xcen']),\
                                            (1-mask).astype('uint8'),\
                                            qphirange.astype('double'),\
                                            phi0=int_aux[0],dphi=int_aux[1],\
                                            returnavgq=True,returnmask=True);
                if save_with_extension is None:
                    save_with_extension='sector'
            else:
                raise ValueError,'Invalid integration mode: %s',inttype
            print 'Integration completed in %f seconds.' %(time.time()-tmp);
            errs1=np.sqrt(ints1**2*errmult**2+mult**2*errs1**2);
            ints1=ints1*mult;
            outname='%s_%s.dat'%(dat[index]['C']['Frame'][:-4],save_with_extension);
            B1io.write1dsasdict({'q':qs1,'Intensity':ints1,'Error':errs1},outname)
            if not noplot:
                pylab.clf()
                pylab.subplot(2,2,1);
                guitools.plot2dmatrix((dat1['data']),mask=1-mask1,header=header,showqscale=True)
                pylab.subplot(2,2,2);
                pylab.cla();
                validindex=(np.isfinite(ints1) & np.isfinite(errs1))
                if (1-validindex).sum()>0:
                    print "WARNING!!! Some nonfinite points are present among the integrated values!"
                    print "NaNs: ",(np.isnan(ints1) | np.isnan(errs1)).sum()
                    print "Infinities: ",(np.isinf(ints1) | np.isinf(errs1)).sum()
                if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                    pylab.plot(qs1[validindex],ints1[validindex])
                    pylab.xlabel(u'theta (rad)');
                    pylab.xscale('linear')
                    pylab.yscale('log')
                else:
                    pylab.plot(qphirange[validindex],ints1[validindex])
                    #pylab.errorbar(qphirange[validindex],ints1[validindex],errs1[validindex]);
                    pylab.xlabel('q (1/Angstrom)');
                    pylab.xscale('log')
                    pylab.yscale('log')
                pylab.ylabel('Absolute intensity (1/cm)');
                pylab.title('%s (%s)' % (dat[index]['C']['Frame'],dat[index]['C']['Sample']));
                pylab.axis('auto')

                pylab.subplot(2,2,3);
                if inttype.upper()=='AZIMUTHAL'[:len(inttype)]:
                    pylab.plot(qs1,areas1)
                    pylab.xlabel('theta (rad)')
                else:
                    pylab.plot(qphirange,areas1);
                    pylab.xlabel('q (1/Angstrom)');
                pylab.ylabel('Effective area');
                if (inttype.upper()=='SECTOR'[:len(inttype)]) or (inttype.upper()=='RADIAL'[:len(inttype)]):
                    pylab.subplot(2,2,4);
                    pylab.plot(qphirange,qs1/qphirange,'.',markersize=3)
                    pylab.xlabel('Original q values')
                    pylab.ylabel('Averaged / original q-values')
                pylab.draw()
                pylab.gcf().show()
                pylab.savefig('int_%u_%s_%s.pdf' % (int(re.findall('[0-9]+',dat1['C']['Frame'])[0]),dat1['C']['Sample'],save_with_extension), format='pdf')
                utils.pause()
                pylab.clf();
    
#~ def addfsns(fileprefix,fsns,fileend,fieldinheader=None,valueoffield=None,dirs=[]):
    #~ """
    #~ """
    #~ data,header=B1io.read2dB1data(fileprefix,fsns,fileend,dirs=dirs)
    
    #~ dataout=None
    #~ headerout=[]
    #~ summed=[]
    #~ for k in range(len(header)):
        #~ h=header[k]
        #~ if (abs(h['Energy']-header[0]['Energy'])<0.5) and \
            #~ (h['Dist']==header[0]['Dist']) and \
            #~ (h['Title']==header[0]['Title']):
                #~ if(h['rot1']!=header[0]['rot1']) or  (h['rot2']!=header[0]['rot2']):
                    #~ print "Warning! Rotation of sample in FSN %d (%s) is different from FSN %d (%s)." % (h['FSN'],h['Title'],header[0]['FSN'],header[0]['Title'])
                    #~ shrubbery=raw_input('Do you still want to add the data? (y/n)   ')
                    #~ if shrubbery.strip().upper()[0]!='Y':
                        #~ return
                #~ if(h['PosRef']!=header[0]['PosRef']):
                    #~ print "Warning! Position of reference sample in FSN %d (%s) is different from FSN %d (%s)." % (h['FSN'],h['Title'],header[0]['FSN'],header[0]['Title'])
                    #~ shrubbery=raw_input('Do you still want to add the data? (y/n)   ')
                    #~ if shrubbery.strip().upper()[0]!='Y':
                        #~ return
                #~ if dataout is None:
                    #~ dataout=data[k].copy()
                #~ else:
                    #~ dataout=dataout+data[k]
                #~ headerout.append(h)
                #~ summed.append(h['FSN'])
    #~ return dataout,headerout,summed
def B1findbeam(data,header,orifsn,orig,mask,quiet=False):
    """Find origin (beam position) on scattering images
    
    Inputs:
        data: list of scattering image matrices of one sequence
        header: list of header structures corresponding to data.
        orifsn: which element of <data> and <header> should be used for
            determining the origin. 0 means empty beam.
            Or you can give a tuple or a list of two: in this case these
            will be the coordinates of the origin and no auto-searching
            will be performed.
        mask: mask matrix. 1 for non-masked, 0 for masked
        orig: helper data for the beam finding procedures. You have
            several possibilities:
            A) a vector/list/tuple of TWO: findbeam_sector() will be
                tried. In this case this is the initial value of the
                beam center
            B) a vector/list/tuple of FOUR: xmin,xmax,ymin,ymax:
                the borders of the rectangle, around the beam, if a
                semitransparent beam-stop was used. In this case
                findbeam_semitransparent() will be tried, and the beam
                center will be determined for each measurement,
                independently (disregarding the value of orifsn).
            C) a vector/list/tuple of FIVE: Ntheta,dmin,dmax,bcx,bcy:
                findbeam_azimuthal will be used. Ntheta, dmin and dmax
                are the respective parameters for azimintpix(), while
                bcx and bcy are the x and y coordinates for the origin
                at the first guess.
            D) a mask matrix (1 means nonmasked, 0 means masked), the
                same size as that of the measurement data. In this
                case findbeam_gravity() will be used.
        quiet: if True, do not plot anything. This is definitely faster.
            Defaults to False.
    Outputs: 
        coords: a list of tuples: each tuple contains the beam center
            coordinates of the corresponding scattering measurement. The
            beam positions are saved to the headers as well.
    """
    try:
        lenorifsn=len(orifsn) # if orifsn is not a list, a TypeError exception gets thrown here.
        if lenorifsn==2:
            orig=(orifsn[0],orifsn[1])
            coords=[orig]*len(data)
            return coords
        else:
            print "Malformed orifsn parameter for B1integrate: ",orifsn
            raise ValueError("Malformed orifsn parameter for B1integrate()")
    except TypeError: # which means that len(orifsn) was not valid -> orifsn is a scalar.
        orig1=None
        try:
            print "Finding beam, len(orig)=",len(orig)
            if len(orig)==2:
                print "Determining origin (by the 'slices' method) from file FSN %d %s" %(header[orifsn]['FSN'],header[orifsn]['Title'])
                orig1=utils2d.findbeam_slices(data[orifsn],orig,mask)
                print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                if not quiet:
                    guitools.testorigin(data[orifsn],orig1,mask)
                    utils.pause()
                coords=[(orig1[0],orig1[1])]*len(data)
                print coords
            elif len(orig)==5:
                print "Determining origin (by the 'azimuthal' method) from file FSN %d %s" %(header[orifsn]['FSN'],header[orifsn]['Title'])
                orig1=utils2d.findbeam_azimuthal(data[orifsn],orig[3:5],mask,Ntheta=orig[0],dmin=orig[1],dmax=orig[2])
                print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                if not quiet:
                    guitools.testorigin(data[orifsn],orig1,mask,dmin=orig[1],dmax=orig[2])
                    utils.pause()
                coords=[(orig1[0],orig1[1])]*len(data)
                print coords
            elif len(orig)==4:
                coords=[]
                for k in range(len(data)):
                    print "Determining origin (by the 'semitransparent' method) for file FSN %d %s" %(header[k]['FSN'],header[k]['Title'])
                    orig1=utils2d.findbeam_semitransparent(data[k],orig)
                    print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                    coords.append((orig1[0],orig1[1]))
                    if not quiet:
                        guitools.testorigin(data[k],orig1,mask)
                        utils.pause()
                print coords
            elif orig.shape==data[orifsn].shape:
                print "Determining origin (by the 'gravity' method) from file FSN %d %s" %(header[orifsn]['FSN'],header[orifsn]['Title'])
                orig1=utils2d.findbeam_gravity(data[orifsn],orig)
                print "Determined origin to be %.2f %.2f." % (orig1[0],orig1[1])
                coords=[(orig1[0],orig1[1])]*len(data)
                if not quiet:
                    guitools.testorigin(data[orifsn-1],orig1,mask)
                    utils.pause()
                print coords
        except:
            print "Finding the origin did not succeed"
            raise
            return None
    print "Saving origins into headers..."
    for k in range(len(header)):
        header[k]['BeamPosX']=coords[k][0]
        header[k]['BeamPosY']=coords[k][1]
    return coords
def geomcorrectiontheta(tth,dist):
    """Create matrix to correct scattered intensity for spatial angle.
    
    Inputs:
        tth: two-theta values, in a matrix
        dist: sample-to-detector distance
    
    Output:
        correction matrix (dist**2/cos(tth)**3).
        
    Notes:
        this value corresponds to the spatial angle covered by each pixel.
        
        
    """
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
    if components is None:
        warnings.warn('gasabsorptioncorrectiontheta(): argument <components> was not defined, using default values for B1@DORISIII in Hasylab, Gabriel detector!')
        components=[{'name':'detector gas area','thick':50,'data':'TransmissionAr910Torr1mm298K.dat'},
                    {'name':'air gap','thick':50,'data':'TransmissionAir760Torr1mm298K.dat'},
                    {'name':'detector window','thick':0.1,'data':'TransmissionBe1mm.dat'},
                    {'name':'flight tube window','thick':0.15,'data':'TransmissionPolyimide1mm.dat'}]
    cor=np.ones(tth.shape)
    for c in components:
        c['travel']=c['thick']/np.cos(tth) # the travel length in the current component
        spam=np.loadtxt("%s%s%s" % (_B1config['calibdir'],os.sep,c['data']))
        tr=np.interp(energycalibrated,spam[:,0],spam[:,1],left=spam[0,1],right=spam[0,-1])
        c['mu']=np.log(tr) # minus mu, d=1, in 1/mm
        cor=cor/np.exp(c['travel']*c['mu'])
    return cor
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
    for d in data:
        tmp={};
        tmp['q']=qrange
        for k in d.keys():
            tmp[k]=np.interp(qrange,d['q'],d[k])
        data2.append(tmp)
    return data2;
def scalewaxs(fsns,mask2d,dirs):
    """Scale waxs curves to saxs files
    
    Inputs:
        fsns: fsn range
        mask2d: mask for the 2d scattering matrices. Zero is masked, nonzero is non-masked.
        dirs: list of directories to be forwarded to IO routines
        
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
        A,Aerr,param=B1io.read2dintfile(fsn,dirs=dirs)
        if len(A)<1:
            continue
        waxsdata=B1io.readwaxscor(fsn,dirs=dirs)
        if len(waxsdata)<1:
            continue
        D=utils2d.calculateDmatrix(mask2d,param[0]['PixelSize'],param[0]['BeamPosX'],
                           param[0]['BeamPosY'])
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
                         param[0]['PixelSize'],param[0]['BeamPosX'],
                         param[0]['BeamPosY'],1-mask2d,qrange)
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
#        mult1=param[0]['NormFactor']
#        errmult1=param[0]['NormFactorRelativeError']*mult1*0.01
        waxsdata[0]['Error']=np.sqrt((waxsdata[0]['Error']*mult)**2+
                                     (errmult*waxsdata[0]['Intensity'])**2)
        waxsdata[0]['Intensity']=waxsdata[0]['Intensity']*mult
        print 'mult: ',mult,'+/-',errmult
#        print 'mult1: ',mult1,'+/-',errmult1
        B1io.writeintfile(waxsdata[0]['q'],waxsdata[0]['Intensity'],waxsdata[0]['Error'],param[0],filetype='waxsscaled')
        [q,I,E,Area]=utils2d.radintC(A[0],Aerr[0],param[0]['EnergyCalibrated'],param[0]['Dist'],
                            param[0]['PixelSize'],param[0]['BeamPosX'],
                            param[0]['BeamPosY'],1-mask2d,q=np.linspace(0,qmax,np.sqrt(mask2d.shape[0]**2+mask2d.shape[1]**2)))
        pylab.figure()
        pylab.subplot(1,1,1)
        pylab.loglog(q,I,label='SAXS')
        pylab.loglog(waxsdata[0]['q'],waxsdata[0]['Intensity'],label='WAXS')
        pylab.legend()
        pylab.title('FSN %d: %s' % (param[0]['FSN'], param[0]['Title']))
        pylab.xlabel('q (1/Angstrom)')
        pylab.ylabel('Scattering cross-section (1/cm)')
        pylab.savefig('scalewaxs%d.png' % param[0]['FSN'],dpi=300,transparent='True',format='png')
        pylab.close(pylab.gcf())
def reintegrateB1(fsnrange,mask,qrange=None,samples=None,savefiletype='intbinned',dirs=[],plot=False,sanitize=None):
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
        plot: if results of each integration should be plotted via plotintegrated()
        sanitize: if sanitization of the integrated data is preferred, set this to
            the field name according to which the sanitization should be done,
            eg. 'Intensity', 'Error', 'Area'. Otherwise leave it None (default).
        
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
                qs,ints,errs,areas,maskout=utils2d.radintC(data[0],dataerr[0],p['EnergyCalibrated'],
                                        p['Dist'],p['PixelSize'],p['BeamPosX'],
                                        p['BeamPosY'],(1-mask).astype(np.uint8),qrange,returnavgq=True,returnmask=True);
                intdata={'q':qs,'Intensity':ints,'Error':errs,'Area':areas,'qorig':qrange}
                if sanitize is not None:
                    intdata=utils.sanitizeint(intdata,accordingto=sanitize)
                    if len(intdata['q'])<len(qs):
                        print "WARNING! There were some q-bins which had to be sanitized, because of no intensity."
                        print "Number of q-bins removed:",len(qs)-len(intdata['q'])
                        print "q min:",intdata['q'].min()
                        print "q max:",intdata['q'].max()
                # as of 27.10.2010, saving averaged q-s.
                B1io.write1dsasdict(intdata,'%s%d.dat'%(savefiletype,p['FSN']))
                print 'done.'
                if plot:
                    guitools.plotintegrated(data[0],intdata['q'],intdata['Intensity'],error=intdata['Error'],area=intdata['Area'],qtheor=intdata['qorig'],mask=1-maskout,param=p)
                    utils.pause()
                del data
                del dataerr
                del qs
                del ints
                del errs
                del areas
def sumfsns(fsns,samples=None,filetype='intnorm',waxsfiletype='waxsscaled',
            dirs=[],plot=False,
            classifyfunc=lambda plist:utils.classify_params_fields(plist,'Title','Energy','Dist'),
            classifyfunc_waxs=lambda plist:utils.classify_params_fields(plist,'Title','Energy'),
            errorpropagation='weight',q_epsilon=1e-3,sanitize='Intensity'):
    """Summarize scattering data.
    
    Inputs:
        fsns: FSN range
        samples: samples to evaluate. Leave it None to auto-determine
        filetype: 1D SAXS filetypes (ie. the beginning of the file) to summarize. 
        waxsfiletype: WAXS filetypes (ie. the beginning of the file) to summarize.
            It is also possible to give a format string, where only the FSN
            should be inserted, like 'waxs_%05d.cor'
        dirs: directories for searching input files.
        plot: True if you want to plot summarized images. False if not. 'stepbystep'
            if you want to pause() after each curve
        classifyfunc: function to classify log structures. Default is grouping
            according to Title, Energy and Dist. 
        classifyfunc_waxs: the same as classifyfunc, but it is used for WAXS
            data. Default is grouping according to Title and Energy only.
        errorpropagation: 'weight' or 'standard'. If 'weight' (default), intensity
            points are normalized by their squared error. If 'standard', each
            curve is taken with the same weight in account.
        q_epsilon: upon summarizing different measurements, (q1-q2)/len(q1) is
            calculated. If this one is smaller than q_epsilon, the two measurements
            are considered the same.
        sanitize: set it to the field name in the SAS dicts according to the automatic
            masking (sanitization) should be done, eg. 'Intensity' (default), 'Area',
            'Error', etc. Set it to None to skip sanitization.
    """
    if not hasattr(fsns,'__getitem__'): # if fsns cannot be indexed
        fsns=[fsns] # make a list of one
    params=B1io.readlogfile(fsns,dirs=dirs)
    if samples is not None:
        if type(samples)==type(''): # if it is a string
            samples=[samples] # make a list of one
        # reduce the set of files to be evaluated to the samples requested.
        params=[p for p in params if p['Title'] in samples]
    # a for loop for small- and wide-angle scattering. First iteration will be SAXS, second WAXS
    for ftype,clfunc,waxsprefix in zip([filetype, waxsfiletype],[classifyfunc,classifyfunc_waxs],['','waxs']):
        pclass=clfunc(params) # classify the parameters with the classification function
        nclass=0 # reset class index
        for plist in pclass: # for each param in the class
            nclass+=1 # increase class counter
            classfsns=repr([p1['FSN'] for p1 in plist]) # fsns in the class
            print 'Class',nclass,', FSNs:',classfsns
            print 'Sample name (first element of class):',plist[0]['Title']
            print 'Distance (first element of class):',plist[0]['Dist']
            print 'Energy (first element of class):',plist[0]['Energy']
            counter=0 # this counts the summed results.
            sanmask=None
            q=None
            w=None
            Isum=None
            Esum=None
            I2Dsum=None
            E2Dsum=None
            counter2D=0
            w2d=None
            fsns_found=[]
            if plot: # initialize plot.
                pylab.clf()
                pylab.xlabel(u'q (1/%c)' % 197)
                pylab.ylabel('Intensity (1/cm)')
                pylab.title('Class %d, FSNs: %s' % (nclass,classfsns))
            for p in plist: # for each param dictionary
                # read file
                try:
                    filename=ftype % p['FSN']
                except TypeError:
                    filename='%s%d.dat' % (ftype,p['FSN'])
                intdata=B1io.readintfile(filename,dirs=dirs)
                if len(intdata)<1:
                    print "sumfsns: cannot find file %s, skipping!" % filename
                    continue
                if sanitize is not None:
                    sanmask=(intdata[sanitize]>0).astype(np.double)
                else:
                    sanmask=np.ones(intdata['q'].shape)
                if counter==0:
                    q=intdata['q']
                    if errorpropagation=='weight':
                        w=1/(intdata['Error']**2)*sanmask
                    else:
                        w=sanmask
                        Esum=intdata['Error']**2*sanmask
                    Isum=intdata['Intensity']*w
                else:
                    if q.size!=intdata['q'].size:
                        print 'q-range of file %s differs from the others read before. Skipping.' % filename
                        continue
                    if np.sum(q-intdata['q'])/len(q)>q_epsilon:
                        print 'q-range of file %s differs from the others read before. Skipping.' % filename
                        continue
                    if errorpropagation=='weight':
                        w1=1/(intdata['Error']**2)*sanmask
                    else:
                        w1=sanmask
                        Esum+=intdata['Error']**2*sanmask
                    Isum=Isum+intdata['Intensity']*w1
                    w=w+w1
                if waxsprefix=='': # ONLY IN SAXS MODE:
                    # load 2D data
                    int2d,err2d,temp=B1io.read2dintfile(p['FSN'],dirs=dirs)
                    if len(int2d)<1:
                        print "Cannot load 2D intensity file for FSN %d, skipping." % p['FSN']
                    else:
                        if counter2D==0:
                            if errorpropagation=='weight':
                                w2d=1/(err2d[0]**2)
                            else:
                                w2d=1
                                E2Dsum=err2d[0]**2
                            I2Dsum=int2d[0]*w2d
                        else:
                            if errorpropagation=='weight':
                                w1=1/(err2d[0]**2)
                            else:
                                w1=1
                                E2Dsum+=err2d[0]**2
                            I2Dsum+=int2d[0]*w1
                            w2d+=w1
                        counter2D+=1
                if plot:
                    pylab.loglog(intdata['q'],intdata['Intensity'],'.-',label='FSN #%d, T=%f' %(p['FSN'],p['Transm']))
                    print "FSN: %d, Transm: %f" %(p['FSN'],p['Transm'])
                    try:
                        if plot.upper()=='STEPBYSTEP':
                            pylab.legend(loc='best')
                            utils.pause()
                    except:
                        pass
                fsns_found.append(p['FSN'])
                counter=counter+1
            if waxsprefix=='':
                if counter2D>0:
                    if errorpropagation=='weight':
                        E2Dsum=1/w2d
                    else:
                        E2Dsum=np.sqrt(E2Dsum)/w2d
                    I2Dsum=I2Dsum/w2d
                    E2Dsum[np.isnan(E2Dsum)]=0
                    I2Dsum[np.isnan(I2Dsum)]=0
                    B1io.write2dintfile(I2Dsum,E2Dsum,plist[0],norm='summed2d')
            if counter>0:
                if errorpropagation=='weight':
                    Esum=1/w
                else:
                    Esum=np.sqrt(Esum)/w
                Esum[np.isnan(Esum)]=0
                Isum=Isum/w
                Isum[np.isnan(Isum)]=0
                if plot:
                    pylab.loglog(q,Isum,'o',label='Sum',markerfacecolor='None',linewidth=5,markeredgewidth=1,markersize=10)
                    pylab.legend(loc='best')
                    pylab.draw()
                    pylab.gcf().show()
                    pylab.savefig('summing%s_class%d.pdf' % (waxsprefix,nclass),format='pdf',dpi=300)
                    utils.pause()
                
                B1io.writeintfile(q,Isum,Esum,plist[0],filetype='summed%s'%waxsprefix)
                try:
                    sumlog=open('summed%s%d.log' % (waxsprefix,plist[0]['FSN']),'wt+')
                    sumlog.write('FSN: %d\n' % plist[0]['FSN'])
                    sumlog.write('FSNs: ')
                    for i in [p['FSN'] for p in plist]:
                        sumlog.write('%d ' %i)
                    sumlog.write('\n')
                    sumlog.write('Sample name: %s\n' % plist[0]['Title'])
                    sumlog.write('Calibrated energy: %f \n' % plist[0]['EnergyCalibrated'])
                    sumlog.write('Temperature: %f \n' % plist[0]['Temperature'])
                    sumlog.write('Sample-to-detector distance (mm): %f\n' % plist[0]['Dist'])
                    sumlog.write('Energy (eV): %f\n' % plist[0]['Energy'])
                    sumlog.close()
                except IOError:
                    print "Cannot save logfile summed%s%d.log." % (waxsprefix,plist[0]['FSN'])
                    raise
            else:
                print 'No %sfiles were found for summing.' % waxsprefix


def unitefsns_old(fsns,distmaskdict,sample=None,qmin=None,qmax=None,qsep=None,
              dirs=[],ignorescalingerror=False,qtolerance=0.05,
              ignorewaxs=False,qsepw=None,
              classifyfunc=lambda plist:utils.classify_params_fields(plist,'Title','Energy'),
              classifyfunc_waxs=lambda plist:utils.classify_params_fields(plist,'Title','Energy'),
              filetype='summed',logfiletype='summed',waxsfiletype='summedwaxs',waxslogfiletype='summedwaxs',plot=True):
    """Unite summed scattering results.
    
    Inputs:
        fsns: range of file sequence numbers (list)
        distmaskdict: a distance-to-mask dictionary. Eg. if you measured at
            two sample-to-detector distances (say 935 and 3535 mm), and the
            corresponding masks reside in numpy arrays maskshort and masklong,
            then you should construct distmaskdict={935:maskshort,3535:masklong}.
            Notice that the distance should be the correct distance.
        sample (default: None): the sample name to be processed. If None, all
            samples found in the FSN range will be treated.
        qmin, qmax: limiting values of the common q-range. If None, autodetect
            from the mask.
        qsep: separating q-value. The resulting (united) curve will have its
            points before this value from the long geometry, after this value
            from the short geometry. If None, all points are used, short and
            long geometry are combed together.
        dirs: the dirs parameter, to be forwarded to I/O routines (like
            readlogfile).
        ignorescalingerror: if the error of the scaling factor should be
            neglected.
        qtolerance: when integrating scattering images onto the common q-range,
            bins where the relative distance of the averaged q-value and the
            desired one is larger than this value, are neglected.
        ignorewaxs: True if you would prefer to skip WAXS files.
        qsepw: like qsep, but for short geometry-WAXS passing.
        classifyfunc: classification function (see eg. classify_params_field).
            Default is to classify measurements with respect to Title and Energy.
        classifyfunc_waxs: classification function for WAXS curves. Defaults
            to classification with respect to Title and Energy.
        filetype: file type for saxs measurements (default: 'summed')
        waxsfiletype: file type for waxs measurements (default: 'summedwaxs')
        plot: if plotting is requested during uniting.
    
    Outputs: none, files are saved.
    
    Notes: the function finds the common q-range among measurements at different
        sample-to-detector distance. Re-integrates 2D scattering images onto
        that q-range, summarizes them, calculates the scaling factor and passes
        the original summed intensities (read from summed*.dat) to each other
        using this factor.
    """
    # read logfiles produced during summarization
    paramsum=B1io.readlogfile(fsns,norm=logfiletype,dirs=dirs)
    if not ignorewaxs:
        paramsumw=B1io.readlogfile(fsns,norm=waxslogfiletype,dirs=dirs)
        if len(paramsumw)<1:
            print "Cannot load %s*.log files, skipping WAXS!" % waxslogfiletype
            ignorewaxs=True
    # read summed data. Note, that datasumparam contains log structures read
    # from intnorm*.log files, thus corresponding to the first measurement of
    # which the summarized data was made.
    datasum,datasumparam=B1io.readintnorm(fsns,filetype,dirs=dirs)
    if not ignorewaxs:
        datasumw,datasumparamw=B1io.readintnorm(fsns,waxsfiletype,dirs=dirs)
    # argument <sample> defaults to all samplenames.
    if sample is not None:
        if type(sample)==type(''): #workaround if a string was given. This makes the next for loop possible
            sample=[sample]
    pclass=classifyfunc(paramsum)
    print "Number of classes:",len(pclass)
    if not ignorewaxs:
        pclassw=classifyfunc_waxs(paramsumw)
        print "Number of WAXS classes:",len(pclassw)
    nclass=0
    for plist in pclass:    #for every class
        nclass +=1
        print "Uniting measurements for class", nclass
        print "Sample name (first member of class):",plist[0]['Title']
        print "Energy (first member of class):",plist[0]['Energy']
        try:
            classfsns=[p['FSNs'] for p in plist]
        except KeyError:
            classfsns=[[p['FSN']] for p in plist]
        print "FSNs:",repr(classfsns)
        #find the distances
        dists=utils.unique([p['Dist'] for p in plist])
        onlyone=False
        if len(dists)<2: #less than two distances: no point of uniting!
            print "Measurements at only one distance exist from class %d. What do you want to unite?" % nclass
            onlyone=True
        if len(dists)>2: # more than two distances: difficult, not implemented.
            print "Measurenemts at more than two distances exist from sample %d. This is currently not supported. Sorry." % nclass
            continue
        d1=min(dists) #shortest distance
        d2=max(dists) #longest distance
        print "Two distances: %f and %f." % (d1,d2)
        # NOTE that if onlyone is True (measurements at only one distance were found), carry on as if nothing happened,
        # and correct things at the end. Ugly hack, I know. AW.
        
        # find the parameter structure corresponding to d1 and d2 and WAXS measurement
        ps1=[p for p in plist if (p['Dist']==d1)]
        if len(ps1)>1:
            print "WARNING! More than one summed files exist of class %d, distance %f. FSNs: %s" % (nclass,d1,repr([min(p['FSNs']) for p in ps1]))
        ps1=ps1[0]
        if not 'FSNs' in ps1.keys():
            ps1['FSNs']=[ps1['FSN']]
        # the summed dataset corresponding to the shortest geometry
        ds1,param1=[(d,p) for (d,p) in zip(datasum,datasumparam) if p['FSN']==ps1['FSN']][0]
        ps2=[p for p in plist if (p['Dist']==d2)]
        if len(ps2)>1:
            print "WARNING! More than one summed files exist of class %d, distance %f. FSNs: %s" % (nclass,d2,repr([min(p['FSNs']) for p in ps2]))
        ps2=ps2[0]
        if not 'FSNs' in ps2.keys():
            ps2['FSNs']=[ps2['FSN']]
        # the summed dataset corresponding to the longest geometry
        ds2,param2=[(d,p) for (d,p) in zip(datasum,datasumparam) if p['FSN']==ps2['FSN']][0] 
        print "Uniting two summed files: FSNs %d and %d"%(ps1['FSN'],ps2['FSN'])
        if not ignorewaxs:
            #print pclassw
            #utils.pause()
            psw=[]
            for pl in pclassw:
                for p in pl:
                    if (ps1['FSN'] in p['FSNs']) or (ps2['FSN'] in p['FSNs']):
                        psw.append(p)
            if len(psw)>1:
                print """WARNING! More than one summed waxs files exist of class %d.
                         First FSNs from short and large distance: %d and %d.
                         This can be caused by an incorrect summation. FSNs: %s""" %(nclass,ps1['FSN'],ps2['FSN'],repr([min(p['FSNs']) for p in psw]))
            if len(psw)<1:
                print "No class found among WAXS measurements, where FSN %d and FSN %d is present." % (ps1['FSN'],ps2['FSN'])
                waxs_notfound=True
            else:
                psw=psw[0]
                if not 'FSNs' in psw.keys():
                    psw['FSNs']=[psw['FSN']]
                dw,paramw=[(d,p) for (d,p) in zip(datasumw,datasumparamw) if p['FSN']==min(psw['FSNs'])][0] # the summed WAXS dataset
                print "Uniting summed waxs file "
                waxs_notfound=False
        #read intnorm logfiles corresponding to the two distances.
        try: # find the corresponding masks
            mask1=distmaskdict[d1]
            mask2=distmaskdict[d2]
        except TypeError:
            print "distmaskdict parameter is not a dictionary! STOPPING"
            return
        except KeyError,details:
            print "No mask defined in distmaskdict for distance %s. STOPPING" % details
            return
        #find the q-ranges for each distance.
        print "Finding q-range from mask"
        q1min,q1max,Nq1=utils2d.qrangefrommask(mask1,ps1['EnergyCalibrated'],ps1['Dist'],param1['PixelSize'],param1['BeamPosX'],param1['BeamPosY'])
        q2min,q2max,Nq2=utils2d.qrangefrommask(mask2,ps2['EnergyCalibrated'],ps2['Dist'],param2['PixelSize'],param2['BeamPosX'],param2['BeamPosY'])
        print "Found q-range from mask"
        if not ignorewaxs and not waxs_notfound:
            dataw=utils.sanitizeint(B1io.readintfile('%s%d.dat' % (waxsfiletype,paramw['FSN'])))
            qw=dataw['q']
            Iw=dataw['Intensity']
            Ew=dataw['Error']
        if qmin is None:
            qmin=q1min #auto-determination
        if qmax is None:
            qmax=q2max #auto-determination
        if (qmax<=qmin):
            print "No common qrange! Skipping."
            continue
        qrange=np.linspace(qmin,qmax,100) #the common q-range between short and long geometry
        #now re-integrate every measurement, taken at short geometry, to the common q-range
        q1=None;        I1=None;        E1=None;        N=0
        print "Loading and re-integrating 2D images to common q-range"
        for f in ps1['FSNs']:
            print "Loading FSN",f
            A,Aerr,p=B1io.read2dintfile(f,dirs=dirs)
            q0,I0,E0,A0=utils2d.radintC(A[0],Aerr[0],p[0]['EnergyCalibrated'],p[0]['Dist'],p[0]['PixelSize'],p[0]['BeamPosX'],p[0]['BeamPosY'],(1-mask1).astype(np.uint8),qrange,returnavgq=True)
            if q1 is None:
                q1=q0;  I1=I0;  E1=E0**2
            else:
                q1+=q0; I1+=I0; E1+=E0**2
            N+=1
        #make averages from the sums.
        E1=np.sqrt(E1)/N
        I1=(I1)/N
        q1=q1/N
        #do the same re-integration for long geometry
        q2=None;        I2=None;        E2=None;        N=0
        for f in ps2['FSNs']:
            print "Loading FSN",f
            A,Aerr,p=B1io.read2dintfile(f,dirs=dirs)
            q0,I0,E0,A0=utils2d.radintC(A[0],Aerr[0],p[0]['EnergyCalibrated'],p[0]['Dist'],p[0]['PixelSize'],p[0]['BeamPosX'],p[0]['BeamPosY'],(1-mask2).astype(np.uint8),qrange,returnavgq=True)
            if q2 is None:
                q2=q0;  I2=I0;  E2=E0**2
            else:
                q2+=q0; I2+=I0; E2+=E0**2
            N+=1
        #make averages from the sums.
        E2=np.sqrt(E2)/N
        I2=I2/N
        q2=q2/N
        #find the valid q-bins, where the absolute relative difference of the average and the expected q-value is less than qtolerance
        qgood=(np.absolute((q1-qrange)/qrange)<qtolerance) & (np.absolute((q2-qrange)/qrange)<qtolerance)
        print "Number of good q-bins (averaged q is near original q):",qgood.sum(),"from ",len(qrange)
        q2=q2[qgood];   I2=I2[qgood];   E2=E2[qgood];
        q1=q1[qgood];   I1=I1[qgood];   E1=E1[qgood]
        multlong2short,errmultlong2short=utils.multfactor(q1,I1,E1,I2,E2)
        print "Multiplication factor (long -> short): %f +/- %f" % (multlong2short,errmultlong2short)
        del q1,q2,I1,I2,E1,E2,N
        
        if ignorescalingerror:
            errmultlong2short=0
            print "Ignoring error of multiplication factor, by preference of the user!"
        if qsep is None:
            if onlyone:
                qsep=-np.inf
            else:
                qsep=0.5*(ds1['q'].min()+ds2['q'].max())
        if plot:
            pylab.clf()
            pylab.loglog(ds1['q'],ds1['Intensity'],label='Distance %.1f mm' % d1)
            if not onlyone:
                pylab.loglog(ds2['q'],ds2['Intensity']*multlong2short,label='Distance %.1f mm, multiplied' % d2)
        if not ignorewaxs and not waxs_notfound:
            #normalize WAXS data as well.
            #calculate common q-range
            qrangew=qw[qw<ds1['q'].max()]
            if len(qrangew)<2:
                print "Not enough common q-range between the WAXS data and the shortest distance! Skipping!"
                continue
            #re-integrate short distance once again, but on the common q-range between SAXS and WAXS
            q1=None;        I1=None;        E1=None;        N=0
            for f in ps1['FSNs']:
                A,Aerr,p=B1io.read2dintfile(f,dirs=dirs)
                q0,I0,E0,A0=utils2d.radintC(A[0],Aerr[0],p[0]['EnergyCalibrated'],p[0]['Dist'],p[0]['PixelSize'],p[0]['BeamPosX'],p[0]['BeamPosY'],(1-mask2).astype(np.uint8),qrangew,returnavgq=True)
                if q1 is None:
                    q1=q0;  I1=I0;  E1=E0**2
                else:
                    q1+=q0; I1+=I0; E1+=E0**2
                N+=1
            #make averages from the sums.
            E1=np.sqrt(E1)/N
            I1=I1/N
            q1=q1/N
            #find the valid q-bins, where the difference
            qgood=(np.absolute((q1-qrangew)/qrangew)<qtolerance)
            print "Number of good q-bins (averaged q is near original q):",qgood.sum(),"from ",len(qrangew)
            q1=q1[qgood];   I1=I1[qgood];   E1=E1[qgood];
            qw=qw[qw<=qrangew.max()]; Iw=Iw[qw<=qrangew.max()]; Ew=Ew[qw<=qrangew.max()];
            multwaxs2short,errmultwaxs2short=utils.multfactor(q1,I1,E1,Iw,Ew)
            if ignorescalingerror:
                errmultwaxs2short=0
            print "Multiplication factor (WAXS -> short): %f +/- %f" % (multwaxs2short,errmultwaxs2short)
            if qsepw is None:
                qsepw=qrangew.min()
            if plot:
                pylab.loglog(dataw['q'],dataw['Intensity'],label='WAXS, multiplied')
        else:
            qsepw=np.inf
        if np.isfinite(qsep) and plot:
            pylab.plot([qsep,qsep],[pylab.axis()[2],pylab.axis()[3]],label='separator q')
            pylab.plot([min(qrange),min(qrange)],[pylab.axis()[2],pylab.axis()[3]],label='lower bound of common range')
            pylab.plot([max(qrange),max(qrange)],[pylab.axis()[2],pylab.axis()[3]],label='upper bound of common range')
            
        if plot:
            pylab.legend()
            pylab.draw()
            pylab.xlabel(u'q (1/\xc5)')
            pylab.ylabel('Intensity (1/cm)')
            if ignorewaxs and not onlyone:
                pylab.title('Long->short: %g +/- %g' % (multlong2short, errmultlong2short));
            if not ignorewaxs and not onlyone:
                pylab.title('Long->short: %g +/- %g\nWAXS->short: %g +/- %g'% (multlong2short, errmultlong2short,multwaxs2short,errmultwaxs2short));
            if not ignorewaxs and onlyone:
                pylab.title('WAXS->short: %g +/- %g'% (multwaxs2short, errmultwaxs2short));
            utils.pause()
        datalong=utils.trimq(utils.multsasdict(ds2,multlong2short,errmultlong2short),qmax=qsep)
        datashort=utils.trimq(ds1,qmin=qsep,qmax=qsepw)
        if onlyone:
            tocombine=[datashort]
        else:
            tocombine=[datalong,datashort]
        if (not ignorewaxs) and (not waxs_notfound):
            datawaxs=utils.trimq(utils.multsasdict(dataw,multwaxs2short,errmultwaxs2short),qmin=qsepw)
            tocombine.append(datawaxs)
        tocombine=tuple(tocombine)
        datacomb=utils.combinesasdicts(*tocombine)
        unifsn=min(min(ps1['FSNs']),min(ps2['FSNs']))
        B1io.write1dsasdict(datacomb,'united%d.dat' % unifsn)
        print "File saved with FSN:",unifsn

def maskpilatusgaps(rownum,colnum,horizmodule=487,horizgap=7,vertmodule=195,vertgap=17):
    """Create a mask matrix which covers the gaps of a Pilatus detector.
    
    Inputs:
        rownum: number of rows
        colnum: number of columns
        horizmodule (default 487): width of each module, in pixels
        horizgap (default 7): width of the gaps between modules, in pixels
        vertmodule (default 195): height of each module, in pixels
        vertgap (default 17): height of the gap between modules, in pixels
        
    Outputs:
        the mask matrix, a two-dimensional numpy array, dtype is uint8, masked
        areas are 0-ed out. Unmasked is 1.
        
    """
    mask=np.ones((rownum,colnum),np.uint8)
    col,row=np.meshgrid(range(colnum),range(rownum))
    mask[col % (horizmodule+horizgap) >=horizmodule]=0
    mask[row % (vertmodule+vertgap) >=vertmodule]=0
    return mask
    
def B1_autointegrate(A,Aerr,param,mask,qrange=None,maskout_needed=False):
    q,I,E,Area,maskout=utils2d.radintC(A,Aerr,param['EnergyCalibrated'],
                               param['Dist'],param['PixelSize'],
                               param['BeamPosX'],param['BeamPosY'],
                               (1-mask).astype(np.uint8),q=qrange,returnavgq=True,returnmask=True)
    if maskout_needed:
        return {'q':qrange,'Intensity':I,'Error':E,'Area':Area,'qaverage':q,'maskout':maskout}
    else:
        return {'q':qrange,'Intensity':I,'Error':E,'Area':Area,'qaverage':q}

def unitefsns(fsns,distmaskdict,sample=None,qmin=None,qmax=None,qsep=None,
              dirs=[],ignorescalingerror=False,qtolerance=0.05,
              ignorewaxs=False,classifyfunc=lambda plist:utils.classify_params_fields(plist,'Title','Energy'),
              classifyfunc_waxs=lambda plist:utils.classify_params_fields(plist,'Title','Energy'),
              filetype='summed',logfiletype='summed',waxsfiletype='summedwaxs',waxslogfiletype='summedwaxs',plot=True,savefiletype='united'):
    """Unite summed scattering results.
    
    Inputs:
        fsns: range of file sequence numbers (list)
        distmaskdict: a distance-to-mask dictionary. Eg. if you measured at
            two sample-to-detector distances (say 935 and 3535 mm), and the
            corresponding masks reside in numpy arrays maskshort and masklong,
            then you should construct distmaskdict={935:maskshort,3535:masklong}.
            Notice that the distance should be the correct distance.
        sample (default: None): the sample name to be processed. If None, all
            samples found in the FSN range will be treated.
        qmin, qmax: limiting values of the common q-range. If None, autodetect
            from the mask.
        qsep: separating q-value. The resulting (united) curve will have its
            points before this value from the long geometry, after this value
            from the short geometry. If None, all points are used, short and
            long geometry are combed together.
        dirs: the dirs parameter, to be forwarded to I/O routines (like
            readlogfile).
        ignorescalingerror: if the error of the scaling factor should be
            neglected.
        qtolerance: when integrating scattering images onto the common q-range,
            bins where the relative distance of the averaged q-value and the
            desired one is larger than this value, are neglected.
        ignorewaxs: True if you would prefer to skip WAXS files.
        classifyfunc: classification function (see eg. classify_params_field).
            Default is to classify measurements with respect to Title and Energy.
        classifyfunc_waxs: classification function for WAXS curves. Defaults
            to classification with respect to Title and Energy.
        filetype: file type for saxs measurements (default: 'summed')
        logfiletype: type of the log files (default: 'summed')
        waxsfiletype: file type for waxs measurements (default: 'summedwaxs')
        waxslogfiletype: type of the waxs log files (default: 'summedwaxs')
        plot: if plotting is requested during uniting.
        savefiletype: filetype to save. Either a format string with a single "%d"
            for the FSN or the beginning of the filename, like "united".
    
    Outputs: none, files are saved.
    
    Notes: the function finds the common q-range among measurements at different
        sample-to-detector distance. Re-integrates 2D scattering images onto
        that q-range, summarizes them, calculates the scaling factor and passes
        the original summed intensities (read from summed*.dat) to each other
        using this factor.
    """
    if sample is None:
        sample=[]
    if np.isscalar(sample):
        sample=[sample]
    print "Reading logfiles of type %s"%logfiletype
    allparams=B1io.readlogfile(fsns,dirs=dirs,norm=logfiletype,quiet=True)
    print "%d logfiles have been loaded." % len(allparams)
    allparams=[p for p in allparams if p['Title'] in sample]
    for p in allparams: # if uniting non-summarized measurements, update the param structures to look as if they were summed from 1 measurement
        if 'FSNs' not in p.keys():
            p['FSNs']=[p['FSN']]
    classes=classifyfunc(allparams)
    print "Number of classes:",len(classes)
    if not ignorewaxs:
        allparamswaxs=B1io.readlogfile(fsns,dirs=dirs,norm=waxslogfiletype,quiet=True)
        if len(allparamswaxs)<1:
            print "WAXS log files not found, disabling WAXS in unitefsns()."
            ignorewaxs=True
        else:
            for p in allparamswaxs: # if uniting non-summarized measurements, update the param structures to look as if they were summed from 1 measurement
                if 'FSNs' not in p.keys():
                    p['FSNs']=[p['FSN']]
            classeswaxs=classifyfunc_waxs(allparamswaxs)
            if len(classes)!=len(classeswaxs):
                raise RuntimeError("Number of WAXS classes differs from number of SAXS classes!")
    for i in range(len(classes)):
        currentclass=classes[i]
        if not ignorewaxs:
            currentclasswaxs=classeswaxs[i]
        dists=utils.unique([p['Dist'] for p in currentclass])
        lastmult=1
        lasterrmult=0
        datastounite=[]
        lastqsep=np.inf
        allfsns=[]
        for di in range(len(dists)): # dists is ordered by utils.unique(), starts from the shortest.
            if di==0: # in this case, calculate the multiplication factor for SAXS and WAXS
                if ignorewaxs:
                    continue # no waxs measurements, then continue with i==1.
                if len(currentclasswaxs)>1:
                    print "more than one summarized WAXS files exist from class #%d. USING FIRST! FSNS:" % (i),[p['FSN'] for p in currentclasswaxs]
                fsnslong=currentclasswaxs[0]['FSNs']
                dataslong,paramslong=B1io.readintnorm(fsnslong,dirs=dirs,filetype=waxsfiletype,logfiletype=waxslogfiletype,quiet=True)
                datalong=dataslong[0]
                paramlong=paramslong[0]
                print "Uniting WAXS to distance",dists[di]
                shortdist=dists[di]
            else:
                currentclasslong=[c for c in currentclass if c['Dist']==dists[di]]
                if len(currentclasslong)>1:
                    print "more than one summarized files exist from class #%d. USING FIRST! FSNS:" % (i),[p['FSN'] for p in currentclasslong]
                fsnslong=currentclasslong[0]['FSNs']
                dataslong,paramslong=B1io.readintnorm(fsnslong,dirs=dirs,filetype=filetype,logfiletype=logfiletype,quiet=True)
                datalong=dataslong[0]
                paramlong=paramslong[0]
                print "Uniting distance",dists[di],"to",dists[di-1]
                shortdist=dists[di-1]
            currentclassshort=[c for c in currentclass if c['Dist']==shortdist]
            if len(currentclassshort)>1:
                print "more than one summarized files exist from class #%d. USING FIRST! FSNS:" % (i),[p['FSN'] for p in currentclassshort]
            fsnsshort=currentclassshort[0]['FSNs']
            datasshort,paramsshort=B1io.readintnorm(fsnsshort,dirs=dirs,filetype=filetype,logfiletype=logfiletype,quiet=True)
            datashort=datasshort[0]
            paramshort=paramsshort[0]
            allfsns.extend(fsnslong)
            allfsns.extend(fsnsshort)
            # now we have fsns, data and param for long (or WAXS) and short distance
            
            # !! NOTICE !! Under LONG distance, we mean the measurement which
            # will be scaled to the other. SHORT is the measurement, to which LONG
            # will be scaled !! Only WAXS makes a difference, because it has to be scaled
            # to SAXS.
            
            #let's find the multiplication factor. To accomplish that, we need the common q-range.
            # This common q-range is either supplied by the user (qmin, qmax) or has to determined by us.
            if qmin is None:
                ourqmin=max(min(datashort['q']),min(datalong['q']))
            elif np.isscalar(qmin):
                ourqmin=qmin
            else:
                if ignorewaxs and len(qmin)==len(dists)-1:
                    ourqmin=qmin[di-1]
                elif not ignorewaxs and len(qmax)==len(dists):
                    ourqmin=qmin[di]
                else:
                    raise ValueError("qmin should be either scalar or a list of the same size as many distances are present (minus one if WAXS is not present)")
            if qmax is None:
                ourqmax=min(max(datashort['q']),max(datalong['q']))
            elif np.isscalar(qmax):
                ourqmax=qmax
            else:
                if ignorewaxs and len(qmax)==len(dists)-1:
                    ourqmax=qmax[di-1]
                elif not ignorewaxs and len(qmax)==len(dists):
                    ourqmax=qmax[di]
                else:
                    raise ValueError("qmax should be either scalar or a list of the same size as many distances are present (minus one if WAXS is not present)")
            if qsep is None:
                ourqsep=0.5*(ourqmin+ourqmax)
            elif np.isscalar(qsep):
                ourqsep=qsep
            else:
                if ignorewaxs and len(qsep)==len(dists)-1:
                    ourqsep=qsep[di-1]
                elif not ignorewaxs and len(qsep)==len(dists):
                    ourqsep=qsep[di]
                else:
                    raise ValueError("qsep should be either scalar or a list of the same size as many distances are present (minus one if WAXS is not present)")
            # now we have qmin, qmax, qsep for uniting the current two distances.
            print "Common q-range: ",ourqmin," to ",ourqmax
            print "Separator q: ",ourqsep
            #re-integrate the two distances.
            if di==0: # WAXS:
                dataredlong=utils.trimq(datalong,qmin=ourqmin,qmax=ourqmax)
                if len(dataredlong['q'])<2:
                    raise ValueError("WAXS curve does not have enough q-points in the common q-range!")
                commonqrange=dataredlong['q']
            else:
                commonqrange=np.linspace(ourqmin,ourqmax,10);
                intdata=[]
                print "Loading int2dnorm files for long distance..."
                Along,Aerrlong,paramlong=B1io.read2dintfile(fsnslong,dirs=dirs,quiet=True)
                print "done."
                for j in range(len(Along)):
                    print "Re-integrating FSN %d"%paramlong[j]['FSN']
                    intdata.append(utils.SASDict(**B1_autointegrate(Along[j],Aerrlong[j],paramlong[j],distmaskdict[paramlong[j]['Dist']],commonqrange)))
                print "re-integration of long distance done."
                del Along
                del Aerrlong
                del paramlong
                dataredlong=utils.combinesasdicts(*intdata,accordingto='q')
                del intdata
            intdata=[]
            print "Loading int2dnorm files for short distance..."
            Ashort,Aerrshort,paramshort=B1io.read2dintfile(fsnsshort,dirs=dirs)
            print "done."
            for j in range(len(Ashort)):
                print "Re-integrating FSN %d"%paramshort[j]['FSN']
                intdata.append(utils.SASDict(**B1_autointegrate(Ashort[j],Aerrshort[j],paramshort[j],distmaskdict[paramshort[j]['Dist']],commonqrange)))
            del Ashort
            del Aerrshort
            del paramshort
            dataredshort=utils.combinesasdicts(*intdata,accordingto='q')
            del intdata
            print "Re-integration done."
            #now dataredlong and dataredshort contain the integrated data, reduced to the common q-range.
            
            # check if q-scales are matching.
            goodqs=dataredlong['q'][2*np.absolute(dataredlong['q']-dataredshort['q'])/(dataredlong['q']+dataredshort['q'])<qtolerance]
            print "Number of good q-s: %d out of %d" % (len(goodqs),len(dataredlong['q']))
            dataredlong=utils.trimq(dataredlong,qmin=goodqs.min(),qmax=goodqs.max())
            dataredshort=utils.trimq(dataredshort,qmin=goodqs.min(),qmax=goodqs.max())
            
            # calculate the multiplication factor to normalize long to short.
            mult1,errmult1=utils.multfactor(dataredlong['q'],dataredshort['Intensity'],dataredshort['Error'],dataredlong['Intensity'],dataredlong['Error'])
            print "Multiplication factor:",mult1,"+/-",errmult1
            if ignorescalingerror:
                print "Ignoring scaling errors"
                errmult1=0;
            # save into datastounite[]
            if di==0: # uniting WAXS to SAXS: save WAXS curve, normalized to SAXS
                print "Storing multiplied WAXS curve, trimming it from below at",ourqsep
                datastounite.append(utils.trimq(utils.multsasdict(datalong,mult1,errmult1),qmin=ourqsep))
            else: # uniting long SAXS to short SAXS: save short SAXS with scaling by the previous scaling factor.
                # first cut the previously saved SAXS curve from above. If the previously saved curve is WAXS, do not trim.
                #  save the current short distance, normalized by the last multfactor (if this is the first SAXS to be saved, lastmult is 1, lasterrmult is 0).
                print "Storing short distance curve, trimming it from above at",lastqsep
                datastounite.append(utils.trimq(utils.multsasdict(datashort,lastmult,lasterrmult),qmin=ourqsep,qmax=lastqsep))
                lastqsep=ourqsep
                # adjust lastmult and lasterrmult
                lasterrmult=np.sqrt((lastmult*errmult1)**2+(mult1*lasterrmult)**2)
                lastmult*=mult1
                # if this is the last iteration, save long distance as well.
                if di==len(dists)-1:
                    print "This is the last distance, storing long distance curve, trimming it from above at",ourqsep
                    datastounite.append(utils.trimq(utils.multsasdict(datalong,lastmult,lasterrmult),qmax=ourqsep))
            if plot:
                pylab.clf()
                datalongmult=utils.multsasdict(datalong,mult1,errmult1)
                pylab.loglog(datalongmult['q'],datalongmult['Intensity'],'.-',label='long')
                pylab.loglog(datashort['q'],datashort['Intensity'],'.-',label='short')
                pylab.loglog(dataredshort['q'],dataredshort['Intensity'],'.-',label='reduced short')
                pylab.loglog(dataredlong['q'],dataredlong['Intensity']*mult1,'.-',label='reduced long, scaled')
                a=pylab.axis()
                pylab.plot([ourqmin,ourqmin],[a[2],a[3]],'-',label='q min')
                pylab.plot([ourqmax,ourqmax],[a[2],a[3]],'-',label='q max')
                pylab.plot([ourqsep,ourqsep],[a[2],a[3]],'-',label='q sep')
                pylab.xlabel(u'q (1/\xc5)')
                pylab.ylabel('Intensity (1/cm)')
                pylab.legend(loc='best')
                utils.pause()
                del datalongmult
            del dataredlong
            del dataredshort
            del datashort
            del datalong
        #we are ready with datastounite[].
        print "Number of stored curves:",len(datastounite)
        if plot:
            pylab.clf()
            for data in datastounite:
                pylab.loglog(data['q'],data['Intensity'],'.-')
#                utils.pause()
            pylab.xlabel(u'q (1/\xc5)')
            pylab.ylabel('Intensity (1/cm)')
            pylab.gcf().show()
            pylab.draw()
        print "Combining stored curves"
        united=utils.combinesasdicts(*datastounite)
        if plot:
            pylab.loglog(united['q'],united['Intensity'],'-')
            utils.pause()
        try:
            fname=savefiletype % min(allfsns)
        except:
            fname='%s%d.dat' % (savefiletype,min(allfsns))
        B1io.write1dsasdict(united,fname)
        print "United curve saved as %s"%fname
        del datastounite
        del united
        
def getsamplenamesxls(fsns,xlsname,dirs,whattolist=None):
    """ getsamplenames revisited, XLS output.
    
    Inputs:
        fsns: FSN sequence
        xlsname: XLS file name to output listing
        dirs: either a single directory (string) or a list of directories, a la readheader()
        whattolist: format specifier for listing. Should be a list of tuples. Each tuple
            corresponds to a column in the worksheet, in sequence. The first element of
            each tuple is the column title, eg. 'Distance' or 'Calibrated energy (eV)'.
            The second element is either the corresponding field in the header dictionary
            ('Dist' or 'EnergyCalibrated'), or a tuple of them, eg. ('FSN', 'Title', 'Energy').
            If the column-descriptor tuple does not have a third element, the string
            representation of each field (str(param[i][fieldname])) will be written
            in the corresponding cell. If a third element is present, it is treated as a 
            format string, and the values of the fields are substituted.
    Outputs:
        an XLS workbook is saved.
    
    Notes:
        if whattolist is not specified exactly (ie. is None), then the output
            is similar to getsamplenames().
        module xlwt is needed in order for this function to work. If it cannot
            be imported, the other functions may work, only this function will
            raise a NotImplementedError.
    """
    if 'xlwt' not in sys.modules.keys():
        raise NotImplementedError('Module xlwt missing, function generatexls() cannot work without it.')
    params=B1io.readheader('org_',fsns,'.header',dirs)

    if whattolist is None:
        whattolist=[('FSN','FSN'),('Time','MeasTime'),('Energy','Energy','%.2f'),
                    ('Distance','Dist','%.0f'),('Position','PosSample','%.2f'),
                    ('Transmission','Transm','%.6f'),('Temperature','Temperature','%.2f'),
                    ('Title','Title'),('Date',('Day','Month','Year','Hour','Minutes'),'%02d.%02d.%04d %02d:%02d')]
    wb=xlwt.Workbook(encoding='utf8')
    ws=wb.add_sheet('Measurements')
    for i in range(len(whattolist)):
        ws.write(0,i,whattolist[i][0])
    for i in range(len(params)):
        for j in range(len(whattolist)):
            if np.isscalar(whattolist[j][1]):
                fields=[whattolist[j][1]]
            else:
                fields=whattolist[j][1]
            if len(whattolist[j])==2:
                ws.write(i+1,j,string.join([str(params[i][f]) for f in fields]))
            elif len(whattolist[j])>=3:
                ws.write(i+1,j,whattolist[j][2] % tuple([params[i][f] for f in fields]))
    wb.save(xlsname)

    