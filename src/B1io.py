#-----------------------------------------------------------------------------
# Name:        B1io.py
# Purpose:     I/O components for B1python
#
# Author:      Andras Wacha
#
# Created:     2010/02/22
# RCS-ID:      $Id: B1io.py $
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

import pylab
import numpy as np
import types
import os
import gzip
import zipfile
import string
import scipy.io
import utils
import re

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
        nseq=floor(nseq);
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
    qs=[]; ints=[]; errs=[]; areas=[];
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


def bdf2B1(bdf,doffset,steps2cm,energyreal,fsnref,thicknessref,pixelsize,mult,errmult,thickness):
    """Convert BDF file to B1-type logfile and int2dnorm files.

    Inputs:
        bdf: bdf dictionary
        doffset: detector offset in cm-s
        steps2cm: multiplicative correction to detector position (cm/motor step)
        energyreal: true (calibrated) energy
        fsnref: FSN for the reference file (Glassy carbon)
        thicknessref: thickness of the reference, in microns
        pixelsize: detector resolution (mm/pixel)
        mult: absolute normalization factor
        errmult: error of mult
        thickness: sample thickness
    """
    params={}
    try:
        params['FSNempty']=int(re.findall('[0-9]+',bdf['C']['Background'])[0])
    except KeyError:
        params['FSNempty']=0
    params['Temperature']=float(bdf['CS']['Temperature'])
    params['NormFactorRelativeError']=errmult/mult
    params['InjectionGC']=False
    params['InjectionEB']=False
    params['Monitor']=int(bdf['CS']['Monitor'])
    params['BeamsizeY']=0
    params['BeamsizeX']=0
    params['Thicknessref1']=float(thicknessref)/10000.
    params['PosSample']=float(bdf['M']['Sample_x'])
    params['BeamPosX']=float(bdf['C']['ycen'])
    params['BeamPosY']=float(bdf['C']['xcen'])
    params['dclevel']=0
    params['FSNref1']=int(fsnref)
    params['MeasTime']=float(bdf['C']['Sendtime'])
    params['PixelSize']=pixelsize
    params['PrimaryIntensity']=0
    params['ScatteringFlux']=0
    params['Rot1']=0
    params['RotXsample']=0
    params['Rot2']=0
    params['RotYsample']=0
    params['NormFactor']=mult
    params['FSN']=int(re.findall('[0-9]+',bdf['C']['Frame'])[0])
    params['Title']=bdf['C']['Sample']
    params['Dist']=(float(bdf['M']['Detector'])*steps2cm+doffset)*10.0
    params['Energy']=float(bdf['M']['Energy'])
    params['EnergyCalibrated']=energyreal
    params['Thickness']=thickness
    params['Transm']=float(bdf['CT']['trans'])
    params['Anode']=float(bdf['CS']['Anode'])
    #extended
    params['TransmError']=float(bdf['CT']['transerr'])
    writelogfile(params,[params['BeamPosX'],params['BeamPosY']],\
                 params['Thickness'],params['dclevel'],\
                 params['EnergyCalibrated'],params['Dist'],\
                 params['NormFactor'],errmult,params['FSNref1'],\
                 params['Thicknessref1'],params['InjectionGC'],\
                 params['InjectionEB'],params['PixelSize'])
    write2dintfile(bdf['data']*mult/thickness,np.sqrt((bdf['error']*mult)**2+(bdf['data']*errmult)**2)/thickness,params)
    return bdf['data']*mult/thickness,np.sqrt((bdf['error']*mult)**2+(bdf['data']*errmult)**2)/thickness,params
def bdf_read(filename):
    """Read bdf file (Bessy Data Format)

    Input:
        filename: the name of the file

    Output:
        bdf: the BDF structure

    Adapted the bdf_read.m macro from Sylvio Haas.
    """
    bdf={}
    bdf['his']=[] #empty list for history
    bdf['C']={} # empty list for bdf file descriptions
    bdf['M']={} # empty list for motor positions
    bdf['CS']={} # empty list for scan parameters
    bdf['CT']={} # empty list for transmission data
    bdf['CG']={} # empty list for gain values
    mne_list=[]; mne_value=[]
    gain_list=[]; gain_value=[]
    s_list=[]; s_value=[]
    t_list=[]; t_value=[]
    
    fid=open(filename,'rb') #if fails, an exception is raised
    line=fid.readline()
    while len(line)>0:
        mat=line.split()
        if len(mat)==0:
            line=fid.readline()
            continue
        prefix=mat[0]
        sz=len(mat)
        if prefix=='#C':
            if sz==4:
                if mat[1]=='xdim':
                    bdf['xdim']=float(mat[3])
                elif mat[1]=='ydim':
                    bdf['ydim']=float(mat[3])
                elif mat[1]=='type':
                    bdf['type']=mat[3]
                elif mat[1]=='bdf':
                    bdf['bdf']=mat[3]
                elif mat[2]=='=':
                    bdf['C'][mat[1]]=mat[3]
            else:
                if mat[1]=='Sample':
                    bdf['C']['Sample']=[mat[3:]]
        if prefix[:4]=="#CML":
            mne_list.extend(mat[1:])
        if prefix[:4]=="#CMV":
            mne_value.extend(mat[1:])
        if prefix[:4]=="#CGL":
            gain_list.extend(mat[1:])
        if prefix[:4]=="#CGV":
            gain_value.extend(mat[1:])
        if prefix[:4]=="#CSL":
            s_list.extend(mat[1:])
        if prefix[:4]=="#CSV":
            s_value.extend(mat[1:])
        if prefix[:4]=="#CTL":
            t_list.extend(mat[1:])
        if prefix[:4]=="#CTV":
            t_value.extend(mat[1:])
        if prefix[:2]=="#H":
            szp=len(prefix)+1
            tline='%s' % line[szp:]
            bdf['his'].append(tline)

        if line[:5]=='#DATA':
            darray=np.fromfile(fid,dtype=bdf['type'],count=int(bdf['xdim']*bdf['ydim']))
            bdf['data']=np.rot90((darray.reshape(bdf['xdim'],bdf['ydim'])).astype('double').T,1).copy() # this weird transformation is needed to get the matrix in the same form as bdf_read.m gets it.
        if line[:6]=='#ERROR':
            darray=np.fromfile(fid,dtype=bdf['type'],count=int(bdf['xdim']*bdf['ydim']))
            bdf['error']=np.rot90((darray.reshape(bdf['xdim'],bdf['ydim'])).astype('double').T,1).copy()
        line=fid.readline()
    if len(mne_list)==len(mne_value):
        for j in range(len(mne_list)):
            bdf['M'][mne_list[j]]=mne_value[j]
    if len(gain_list)==len(gain_value):
        for j in range(len(gain_list)):
            bdf['CG'][gain_list[j]]=gain_value[j]
    if len(s_list)==len(s_value):
        for j in range(len(s_list)):
            bdf['CS'][s_list[j]]=s_value[j]
    if len(t_list)==len(t_value):
        for j in range(len(t_list)):
            bdf['CT'][t_list[j]]=t_value[j]
    fid.close()
    return bdf
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
        p00=np.loadtxt('%s.P00' % basename)
    except IOError:
        try:
            p00=np.loadtxt('%s.p00' % basename)
        except:
            raise IOError('Cannot find %s.p00, neither %s.P00.' % (basename,basename))
    if p00 is not None:
        p00=p00[1:] # cut the leading -1
    try:
        e00=np.loadtxt('%s.E00' % basename)
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
    jusifaHC=12396.4 #Planck's constant times speed of light: incorrect
                     # constant in the old program on hasjusi1, which was
                     # taken over by the measurement program, to keep
                     # compatibility with that.
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
                pass #continue with the next directory
        if not filefound:
            print 'readheader: Cannot find file %s in given directories.' % name
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
                nx=int(string.strip(lines[10]))
                ny=int(string.strip(lines[11]))
                fid.seek(0,0)
                data=np.loadtxt(fid,skiprows=133)
                data=data.reshape((nx,ny))
                fid.close()
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
    if fileend.upper()=='.HEADER':
        fileend='.TIF'
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
        fileend='.tif'
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
def read2dintfile(fsns,dirs=[],norm=True):
    """Read corrected intensity and error matrices
    
    Input:
        fsns: one or more fsn-s in a list
        dirs: list of directories to try
        norm: True if int2dnorm*.mat file is to be loaded, False if
            int2darb*.mat is preferred
        
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
#                    print 'Cannot find file %s (also tried .zip and .gz)' % filename
                    return None
        return data
    # the core of read2dintfile
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    try:
        lenfsn=len(fsns)
    except TypeError:
        fsns=[fsns]
        lenfsn=1
    int2d=[]
    err2d=[]
    params=[]
    for fsn in fsns: # this also works if len(fsns)==1
        filefound=False
        for d in dirs:
            try: # first try to load the npz file. This is the most effective way.
                if norm:
                    tmp0=np.load('%s%sint2dnorm%d.npz' % (d,os.sep,fsn))
                else:
                    tmp0=np.load('%s%sint2darb%d.npz' % (d,os.sep,fsn))
                tmp=tmp0['Intensity']
                tmp1=tmp0['Error']
            except IOError:
                try: # first try to load the mat file. This is the second effective way.
                    if norm:
                        tmp0=scipy.io.loadmat('%s%sint2dnorm%d.mat' % (d,os.sep,fsn))
                    else:
                        tmp0=scipy.io.loadmat('%s%sint2darb%d.mat' % (d,os.sep,fsn))
                    tmp=tmp0['Intensity']
                    tmp1=tmp0['Error']
                except IOError: # if mat file is not found, try the ascii files
                    if norm:
    #                    print 'Cannot find file int2dnorm%d.mat: trying to read int2dnorm%d.dat(.gz|.zip) and err2dnorm%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
                        tmp=read2dascii('%s%sint2dnorm%d.dat' % (d,os.sep,fsn));
                        tmp1=read2dascii('%s%serr2dnorm%d.dat' % (d,os.sep,fsn));
                    else:
    #                    print 'Cannot find file int2darb%d.mat: trying to read int2darb%d.dat(.gz|.zip) and err2darb%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
                        tmp=read2dascii('%s%sint2darb%d.dat' % (d,os.sep,fsn));
                        tmp1=read2dascii('%s%serr2darb%d.dat' % (d,os.sep,fsn));
                except TypeError: # if mat file was found but scipy.io.loadmat was unable to read it
                    print "Malformed MAT file! Skipping."
                    continue # try from another directory
            tmp2=readlogfile(fsn,d) # read the logfile
            if (tmp is not None) and (tmp1 is not None) and (tmp2 is not None): # if all of int,err and log is read successfully
                int2d.append(tmp)
                err2d.append(tmp1)
                params.append(tmp2[0])
                filefound=True
                print 'Files corresponding to fsn %d were found.' % fsn
                break # file was found, do not try to load it again from another directory
        if not filefound:
            print "read2dintfile: Cannot find file(s ) for FSN %d" % fsn
    return int2d,err2d,params # return the lists
def write2dintfile(A,Aerr,params,norm=True,filetype='npz'):
    """Save the intensity and error matrices to int2dnorm<FSN>.mat
    
    Inputs:
        A: the intensity matrix
        Aerr: the error matrix
        params: the parameter dictionary
        norm: if int2dnorm files are to be saved. If it is false, int2darb files
            are saved (arb = arbitrary units, ie. not absolute intensity)
        filetype: 'npz' or 'mat'
    int2dnorm<FSN>.[mat or npz] is written. The parameter structure is not
        saved, since it should be saved already in intnorm<FSN>.log
    """
    if norm:
        fileprefix='int2dnorm%d' % params['FSN']
    else:
        fileprefix='int2darb%d' % params['FSN']
    if filetype.upper() in ['NPZ','NPY','NUMPY']:
        np.savez('%s.npz' % fileprefix, Intensity=A,Error=Aerr)
    elif filetype.upper() in ['MAT','MATLAB']:
        scipy.io.savemat('%s.mat' % fileprefix,{'Intensity':A,'Error':Aerr});
    else:
        raise ValueError,"Unknown file type: %s" % repr(filetype)
def readintfile(filename,dirs=[],sanitize=True):
    """Read intfiles.

    Input:
        filename: the file name, eg. intnorm123.dat
        dirs [optional]: a list of directories to try

    Output:
        A dictionary with 'q' holding the values for the momentum transfer,
            'Intensity' being the intensity vector and 'Error' has the error
            values. These three fields are numpy ndarrays. An empty dict
            is returned if file is not found.
    """
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    ret={}
    for d in dirs:
        try:
            if d=='.':
                fname=filename
            else:
                fname= "%s%s%s" % (d,os.sep,filename)
            fid=open(fname,'rt');
            lines=fid.readlines();
            fid.close();
            ret['q']=[]
            ret['Intensity']=[]
            ret['Error']=[]
            ret['Area']=[]
            #ret.update({'q':[],'Intensity':[],'Error':[],'Area':[]})
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
                        if sanitize:
                            if not (np.isfinite(tmpI) and np.isfinite(tmpe)):
                                continue
                        ret['q'].append(tmpq);
                        ret['Intensity'].append(tmpI);
                        ret['Error'].append(tmpe);
                        ret['Area'].append(tmpa);
                    except ValueError:
                        #skip erroneous line
                        pass
            ret['q']=pylab.array(ret['q'])
            ret['Intensity']=pylab.array(ret['Intensity'])
            ret['Error']=pylab.array(ret['Error'])
            ret['Area']=pylab.array(ret['Area'])
            if len([1 for x in ret['Area'] if pylab.isnan(x)==False])==0:
                del ret['Area']
            break # file was found, do not iterate over other directories
        except IOError:
            continue
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
        currdata={}
        currlog={}
        for d in dirs:
            filename='%s%s%s%d.dat' % (d,os.sep,filetype, fsn)
            tmp=readintfile(filename)
            if len(tmp)>0:
                currdata=tmp
                break # file was already found, do not try in another directory
        for d in dirs:
            tmp2=readlogfile(fsn,d)
            if len(tmp2)>0:
                currlog=tmp2
                break # file was already found, do not try in another directory
        if len(currdata)>0 and len(currlog)>0:
            data.append(currdata);
            param.append(currlog[0]);
    return data,param
def readbinned(fsn,dirs=[]):
    """Read intbinned*.dat files along with their headers.
    
    This is a shortcut to readintnorm(fsn,'intbinned',dirs)
    """
    return readintnorm(fsn,'intbinned',dirs);
def readlogfile(fsn,dirs=[],norm=True):
    """Read logfiles.
    
    Inputs:
        fsn: the file sequence number(s). It is possible to
            give a single value or a list
        dirs [optional]: a list of directories to try
        norm: if a normalized file is to be loaded (intnorm*.log). If
            False, intarb*.log will be loaded instead. Or, you can give a
            string. In that case, '%s%d.log' %(norm, <FSN>) will be loaded.
            
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
                        'Calibrated energy':'EnergyCalibrated',
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
    logfile_dict_str={'Sample title':'Title',
                      'Sample name':'Title'}
    #this dict. contains the bool parameters
    logfile_dict_bool={'Injection between Empty beam and sample measurements?':'InjectionEB',
                       'Injection between Glassy carbon and sample measurements?':'InjectionGC'
                       }
    logfile_dict_list={'FSNs':'FSNs'}
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
    if type(norm)==type(True):
        if norm:
            norm='intnorm'
        else:
            norm='intarb'
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    try:
        lenfsn=len(fsn)
    except TypeError:
        fsn=[fsn];
    params=[]; #initially empty
    for f in fsn:
        filefound=False
        for d in dirs:
            filebasename='%s%d.log' % (norm,f) #the name of the file
            filename='%s%s%s' %(d,os.sep,filebasename)
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
                    for k in logfile_dict_list.keys():
                        if name==k:
                            spam=getvaluestr(line).split()
                            shrubbery=[]
                            for x in spam:
                                try:
                                    shrubbery.append(float(x))
                                except:
                                    shrubbery.append(x)
                            param[logfile_dict_list[k]]=shrubbery
                param['Title']=string.replace(param['Title'],' ','_');
                param['Title']=string.replace(param['Title'],'-','_');
                params.append(param);
                filefound=True
                del lines;
                break # file was already found, do not try in another directory
            except IOError, detail:
                #print 'Cannot find file %s.' % filename
                pass
        if not filefound:
            print 'Cannot find file %s in any of the given directories.' % filebasename
    return params;
def writelogfile(header,ori,thick,dc,realenergy,distance,mult,errmult,reffsn,
                 thickGC,injectionGC,injectionEB,pixelsize,mode='Pilatus300k',norm=True):
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
        norm: if the normalization went good. If failed, intarb*.dat will be saved.
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
    if norm:
        name='intnorm%d.log' % header['FSN']
    else:
        name='intarb%d.log' % header['FSN']
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
        fid.write('Dark current subtracted (cps):\t%d\n' % dc)
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
        filefound=False
        for d in dirs:
            try:
                filename='%s%swaxs_%05d.cor' % (d,os.sep,fsn)
                tmp=pylab.loadtxt(filename)
                if tmp.shape[1]==3:
                    tmp1={'q':tmp[:,0],'Intensity':tmp[:,1],'Error':tmp[:,2]}
                waxsdata.append(tmp1)
                filefound=True
                break # file was found, do not try in further directories
            except IOError:
                pass
                #print '%s not found. Skipping it.' % filename
        if not filefound:
            print 'File waxs_%05d.cor was not found. Skipping.' % fsn
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
        filefound=False
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
                filefound=True
                break #file found, do not try further directories
            except IOError:
                pass
        if not filefound:
            print 'Cannot find file %s%05d%S.' % (filename,f,fileend)
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
        
        Sequences of different lengths are allowed
    """
    seqs=[]
    for i in range(len(headers)):
        if headers[i]['Title']==ebname:
            seqs.append([])
        if len(seqs)==0:
            print "Dropping measurement %d (%t) because no Empty beam before!" % (headers[i]['FSN'],headers[i]['Title'])
        else:
            seqs[-1].append(i)
    return seqs
def getsequencesfsn(headers,ebname='Empty_beam'):
    """Separate measurements made at different energies in an ASAXS
        sequence and return the lists of FSNs
    
    Inputs:
        header: header (or param) dictionary
        ebname: the title of the empty beam measurements.
    
    Output:
        a list of lists. Each sub-list in this list contains the FSNS in the
        supplied header structure which correspond to the sub-sequence.
    
        Sequences of different lengths are allowed.
    """
    seqs=[]
    for i in range(len(headers)):
        if headers[i]['Title']==ebname:
            seqs.append([])
        if len(seqs)==0:
            print "Dropping measurement %d (%t) because no Empty beam before!" % (headers[i]['FSN'],headers[i]['Title'])
        else:
            seqs[-1].append(headers[i]['FSN'])
    return seqs
def energiesfromparam(param):
    """Return the (uncalibrated) energies from the measurement files
    
    Inputs:
        param dictionary
        
    Outputs:
        a list of sorted energies
    """
    return utils.unique([p['Energy'] for p in param],lambda a,b:(abs(a-b)<2))
def samplenamesfromparam(param):
    """Return the sample names
    
    Inputs:
        param dictionary
        
    Output:
        a list of sorted sample titles
    """
    return utils.unique([p['Title'] for p in param])

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
            d['Energy']=fitting.energycalibration(energymeas,energycalib,pylab.array(energy[0]))
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
        columns.append(a.split()[2][10:]);
        a=f.readline(); rows=rows+1;
        #print a
    #print a[:4]
    f.seek(-len(a),1)
    rows=rows-1;
    #print rows
    matrix=np.loadtxt(f)
    f.close()
    return {'title':title,'mode':mode,'columns':columns,'data':matrix}
def savespheres(spheres,filename):
    """Save sphere structure in a file.
    
    Inputs:
        spheres: sphere matrix
        filename: filename
    """
    pylab.savetxt(filename,spheres,delimiter='\t')
