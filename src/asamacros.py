import numpy as np
import pylab
import fitting
import matplotlib.widgets
import guitools
import utils
import time
from c_asamacros import smearingmatrix, trapezoidshapefunction, smearingmatrixgonio, smearingmatrixflat
import xml.dom.minidom
import os
import shutil
import warnings
import B1io

_asa_config={'dataroot':'/afs/.bionano/misc/measurements'}

def asa_copyfiles(files,fromdir,todir,saxsprefix='saxs',waxsprefix='waxs',exts=['P00','E00','INF']):
    files_return=files[:]
    for prefix in [saxsprefix, waxsprefix]:
        if prefix is None:
            continue
        sas_from=os.path.join(_asa_config['dataroot'],prefix,fromdir)
        sas_to=os.path.join(todir,prefix)
        try:
            os.stat(sas_from)
        except OSError,v:
            if v.errno==2:
                print "Invalid input directory:",sas_from
                raise
            else:
                raise
        try:
            os.stat(sas_to)
        except OSError,v:
            if v.errno==2:
                try:
                    os.mkdir(sas_to)
                    print "Created data directory:",sas_to
                except OSError,v:
                    if v.errno==13:
                        print "Cannot create output directory:",sas_to
                        raise
                    else:
                        raise
        # if this line is reached, sas_from and sas_to exist.
        for f in files:
            extspresent={}
            for e in exts:
                try:
                    shutil.copy(os.path.join(sas_from,'%s.%s'%(f,e)),sas_to)
                    extspresent[e]=True
                except:
                    print "Cannot copy file %s. Checking if already copied." % os.path.join(sas_from,'%s.%s'%(f,e))
                    try:
                        f1=open(os.path.join(sas_to,'%s.%s'%(f,e)),'rt')
                        extspresent[e]=True
                        f1.close()
                        print "File %s.%s was FOUND." % (f,e)
                    except IOError:
#                    print "File %s.%s was not found." % (f,e)
                        extspresent[e]=False
            if not all(extspresent.values()):
                print "Skipping absent measurement %s." % f
                files_return=[x for x in files_return if x!=f]
    return files_return
                

def readxrdml(filename,twothetashift=0,returnSASDicts=False):
    """Read xrdml files made by the software for Panalytical/Philips X'Pert
    
    Input:
        filename: name of the file
        twothetashift: an additive correction for 2*theta.
        returnSASDicts: if SASDicts are to be returned. Possible values:
            False (default): a simple Python dictionary will be returned
            True: a SASDict will be returned, containing the summed data of more
                runs.
            'scans': a list of SASDicts will be returned, each consisting of a 
                single scan.
    Output:
        a dict of the x-ray diffraction data and much more. Field names should
            be self-explanatory.
    """
    data={}
    xrdml=xml.dom.minidom.parse(filename)
    xrdm=xrdml.firstChild
    if not xrdm.nodeName=='xrdMeasurements':
        raise ValueError,"First xml tag in file %f is not 'xrdMeasurements'." % filename
    data['status']=xrdm.attributes['status'].nodeValue
    comments=[cn for cn in xrdm.childNodes if cn.nodeName=='comment'][0]
    data['comment']=[cn.firstChild.nodeValue for cn in comments.childNodes if cn.nodeName=='entry']
    
    sample=[cn for cn in xrdm.childNodes if cn.nodeName=='sample'][0]
    data['sample']={}
    for i in sample.attributes.keys():
        data['sample'][i]=sample.attributes[i].nodeValue
    try:
        data['sample']['id']=[cn.firstChild.nodeValue for cn in sample.childNodes if cn.nodeName=='id'][0]
    except:
        data['sample']['id']=''
    try:
        data['sample']['name']=[cn.firstChild.nodeValue for cn in sample.childNodes if cn.nodeName=='name'][0]
    except:
        data['sample']['name']=''
    try:
        data['sample']['preparedBy']=[cn.firstChild.nodeValue for cn in sample.childNodes if cn.nodeName=='preparedBy'][0]
    except:
        data['sample']['preparedBy']=''
    
    measurements=[cn for cn in xrdm.childNodes if cn.nodeName=='xrdMeasurement']
    data['measurements']=[]
    for m in measurements:
        meas={}
        for i in m.attributes.keys():
            meas[i]=m.attributes[i].nodeValue
        measchilds=[mc for mc in m.childNodes if not mc.nodeName.startswith('#')]
        comments=[mc for mc in measchilds if mc.nodeName=='comment']
        usedwavelength=[mc for mc in measchilds if mc.nodeName=='usedWavelength'][0]
        incidentbeampath=[mc for mc in measchilds if mc.nodeName=='incidentBeamPath'][0]
        diffractedbeampath=[mc for mc in measchilds if mc.nodeName=='diffractedBeamPath'][0]
        scans=[mc for mc in measchilds if mc.nodeName=='scan']
        
        meas['comments']=[]
        for cm in comments:
            meas['comments'].extend([cn.firstChild.nodeValue for cn in cm.childNodes if cn.nodeName=='entry' and cn.firstChild is not None])
        
        meas['usedwavelength']={}
        for i in usedwavelength.attributes.keys():
            meas['usedwavelength'][i]=usedwavelength.attributes[i].nodeValue
        for i in [cn for cn in usedwavelength.childNodes if not cn.nodeName.startswith('#')]:
            try:
                meas['usedwavelength'][i.nodeName]=float(i.firstChild.nodeValue)
            except:
                meas['usedwavelength'][i.nodeName]=i.firstChild.nodeValue
                
        meas['incidentbeampath']={}
        meas['incidentbeampath']['radius']=float(incidentbeampath.getElementsByTagName('radius')[0].firstChild.nodeValue)
        meas['incidentbeampath']['xRayTube']={}
        xraytube=incidentbeampath.getElementsByTagName('xRayTube')[0]
        for i in xraytube.attributes.keys():
            meas['incidentbeampath']['xRayTube'][i]=xraytube.attributes[i].nodeValue
        meas['incidentbeampath']['xRayTube']['tension']=xraytube.getElementsByTagName('tension')[0].firstChild.nodeValue
        meas['incidentbeampath']['xRayTube']['current']=xraytube.getElementsByTagName('current')[0].firstChild.nodeValue
        meas['incidentbeampath']['xRayTube']['anodeMaterial']=xraytube.getElementsByTagName('anodeMaterial')[0].firstChild.nodeValue
        meas['incidentbeampath']['xRayTube']['focus']={}
        xraytubefocus=xraytube.getElementsByTagName('focus')[0]
        for i in xraytubefocus.attributes.keys():
            meas['incidentbeampath']['xRayTube']['focus'][i]=xraytubefocus.attributes[i].nodeValue
        for i in [c for c in xraytubefocus.childNodes if not c.nodeName.startswith('#')]:
            meas['incidentbeampath']['xRayTube']['focus'][i.nodeName]=i.firstChild.nodeValue

        sollerslit=incidentbeampath.getElementsByTagName('sollerSlit')[0]
        meas['incidentbeampath']['sollerslit']={}
        for i in sollerslit.attributes.keys():
            meas['incidentbeampath']['sollerslit'][i]=sollerslit.attributes[i].nodeValue
        for i in [c for c in sollerslit.childNodes if not c.nodeName.startswith('#')]:
            meas['incidentbeampath']['sollerslit'][i.nodeName]=i.firstChild.nodeValue
        
        meas['incidentbeampath']['mask']={}
        try:
            mask=incidentbeampath.getElementsByTagName('mask')[0]
            for i in mask.attributes.keys():
                meas['incidentbeampath']['mask'][i]=mask.attributes[i].nodeValue
            for i in [c for c in mask.childNodes if not c.nodeName.startswith('#')]:
                meas['incidentbeampath']['mask'][i.nodeName]=i.firstChild.nodeValue
        except:
            pass

        meas['diffractedbeampath']={}
        meas['diffractedbeampath']['radius']=float(diffractedbeampath.getElementsByTagName('radius')[0].firstChild.nodeValue)
        
        tmp=diffractedbeampath.getElementsByTagName('antiScatterSlit')[0]
        meas['diffractedbeampath']['antiScatterSlit']={}
        for i in tmp.attributes.keys():
            meas['diffractedbeampath']['antiScatterSlit'][i]=tmp.attributes[i].nodeValue
        for i in [c for c in tmp.childNodes if not c.nodeName.startswith('#')]:
            meas['diffractedbeampath']['antiScatterSlit'][i.nodeName]=i.firstChild.nodeValue

        tmp=diffractedbeampath.getElementsByTagName('sollerSlit')[0]
        meas['diffractedbeampath']['sollerSlit']={}
        for i in tmp.attributes.keys():
            meas['diffractedbeampath']['sollerSlit'][i]=tmp.attributes[i].nodeValue
        for i in [c for c in tmp.childNodes if not c.nodeName.startswith('#')]:
            meas['diffractedbeampath']['sollerSlit'][i.nodeName]=i.firstChild.nodeValue

        tmp=diffractedbeampath.getElementsByTagName('receivingSlit')[0]
        meas['diffractedbeampath']['receivingSlit']={}
        for i in tmp.attributes.keys():
            meas['diffractedbeampath']['receivingSlit'][i]=tmp.attributes[i].nodeValue
        for i in [c for c in tmp.childNodes if not c.nodeName.startswith('#')]:
            meas['diffractedbeampath']['receivingSlit'][i.nodeName]=i.firstChild.nodeValue
        
        detector=diffractedbeampath.getElementsByTagName('detector')[0]
        meas['diffractedbeampath']['detector']={}
        for i in detector.attributes.keys():
            meas['diffractedbeampath']['detector'][i]=detector.attributes[i].nodeValue
        for cn in [c for c in detector.childNodes if not c.nodeName.startswith('#')]:
            meas['diffractedbeampath']['detector'][cn.nodeName]={}
            for i in cn.attributes.keys():
                meas['diffractedbeampath']['detector'][cn.nodeName][i]=cn.attributes[i].nodeValue
            for i in [c for c in cn.childNodes if not c.nodeName.startswith('#')]:
                meas['diffractedbeampath']['detector'][cn.nodeName][i.nodeName]=i.firstChild.nodeValue
        
        meas['twotheta']=None
        meas['q']=None
        meas['normintensity']=None
        meas['scans']=[]
        counter=0
        returnlist=[]
        
        for s in scans:
            scan={}
            for i in s.attributes.keys():
                scan[i]=s.attributes[i].nodeValue
            header=s.getElementsByTagName('header')[0]
            scan['startTimeStamp']=header.getElementsByTagName('startTimeStamp')[0].firstChild.nodeValue
            try:
                scan['endTimeStamp']=header.getElementsByTagName('endTimeStamp')[0].firstChild.nodeValue
            except IndexError:
                scan['endTimeStamp']=None
            scan['authorname']=header.getElementsByTagName('author')[0].getElementsByTagName('name')[0].firstChild.nodeValue
            scan['applicationsoftware']=header.getElementsByTagName('source')[0].getElementsByTagName('applicationSoftware')[0].firstChild.nodeValue
            scan['applicationsoftwareversion']=header.getElementsByTagName('source')[0].getElementsByTagName('applicationSoftware')[0].attributes['version'].nodeValue
            scan['instrumentcontrolsoftware']=header.getElementsByTagName('source')[0].getElementsByTagName('instrumentControlSoftware')[0].firstChild.nodeValue
            scan['instrumentcontrolsoftwareversion']=header.getElementsByTagName('source')[0].getElementsByTagName('instrumentControlSoftware')[0].attributes['version'].nodeValue
            scan['instrumentID']=header.getElementsByTagName('source')[0].getElementsByTagName('instrumentID')[0].firstChild.nodeValue
            datapoints=s.getElementsByTagName('dataPoints')[0]
            for pos in [ p for p in datapoints.childNodes if p.nodeName=='positions']:
                scan[pos.attributes['axis'].nodeValue]={'start':None,'end':None,'common':None}
                for x in ['start', 'end', 'common']:
                    try:
                        xposition=pos.getElementsByTagName('%sPosition'%x)[0]
                        scan[pos.attributes['axis'].nodeValue][x]=float(xposition.firstChild.nodeValue)
                    except IndexError:
                        pass
            scan['commoncountingtime']=None
            try:
                cct=datapoints.getElementsByTagName('commonCountingTime')[0]
                scan['commoncountingtime']=float(cct.firstChild.nodeValue)
            except IndexError:
                pass
            
            scan['countingtimes']=None
            try:
                ct=datapoints.getElementsByTagName('countingTimes')[0]
                scan['countingtimes']=np.array([float(x) for x in ct.firstChild.nodeValue.split()])
                scan['countingtimes_units']=ct.attributes['unit'].nodeValue
            except IndexError:
                pass
                
            ints=datapoints.getElementsByTagName('intensities')[0]
            scan['intensities_units']=ints.attributes['unit'].nodeValue
            scan['intensities']=np.array([float(x) for x in ints.firstChild.nodeValue.split()])
        
            try:
                scanaxiscenter=s.getElementsByTagName('scanAxisCenter')[0]
                for pos in [ p for p in scanaxiscenter.childNodes if p.nodeName=='position']:
                    scan[pos.attributes['axis'].nodeValue]['axiscenter']=float(pos.firstChild.nodeValue)
            except IndexError:
                pass
            
            try:
                reflection=s.getElementsByTagName('reflection')[0]
                hkl=reflection.getElementsByTagName('hkl')[0]
                scan['reflection_hkl']={'h':float(hkl.getElementsByTagName('h')[0].firstChild.nodeValue),
                                        'k':float(hkl.getElementsByTagName('k')[0].firstChild.nodeValue),
                                        'l':float(hkl.getElementsByTagName('l')[0].firstChild.nodeValue)}
            except IndexError:
                pass
            scan['Error']=np.sqrt(scan['intensities'])
            
            if scan['countingtimes'] is not None:
                scan['Intensity']=scan['intensities']/scan['countingtimes']
                scan['Error']/=scan['countingtimes']
            else:
                scan['Intensity']=scan['intensities']/scan['commoncountingtime']
                scan['Error']/=scan['commoncountingtime']
            
            if scan['scanAxis']=='2Theta':
                scan['twotheta']=np.linspace(scan['2Theta']['start'],scan['2Theta']['end'],len(scan['intensities']))
            elif scan['scanAxis']=='Gonio':
                scan['twotheta']=np.linspace(scan['2Theta']['start'],scan['2Theta']['end'],len(scan['intensities']))
            else:
                raise NotImplementedError, "scanAxis is %s, which cannot yet be handled. Please contact the author of this program!" % scan['scanAxis']
            
            wavelength=meas['usedwavelength']['kAlpha1']*(1-meas['usedwavelength']['ratioKAlpha2KAlpha1'])+meas['usedwavelength']['kAlpha2']*(meas['usedwavelength']['ratioKAlpha2KAlpha1'])
            scan['twotheta']+=twothetashift;
            scan['q']=4*np.pi*np.sin(scan['twotheta']*np.pi/180.0*0.5)/wavelength
            

            if meas['twotheta'] is None:
                meas['twotheta']=scan['twotheta']
                meas['q']=scan['q']
                meas['Intensity']=scan['Intensity']
                meas['Error']=scan['Error']**2
                counter=1
            else:
                if len(meas['twotheta'])==len(scan['twotheta']) and (meas['twotheta']-scan['twotheta']).sum()==0:
                    meas['Intensity']=meas['Intensity']+scan['Intensity']
                    meas['Error']=meas['Error']+scan['Error']**2
                    counter+=1
            meas['scans'].append(scan)
            returnlist.append(utils.SASDict(scan['q'],scan['Intensity'],scan['Error'],twotheta=scan['twotheta']))
        meas['Intensity']/=counter
        meas['Error']=np.sqrt(meas['Error'])/counter
        data['measurements'].append(meas)
    if returnSASDicts==False:
        return data
    elif returnSASDicts==True:
        return utils.SASDict(data['measurements'][0]['q'],
                             data['measurements'][0]['Intensity'],
                             data['measurements'][0]['Error'],
                             twotheta=data['measurements'][0]['twotheta'])
    elif returnSASDicts.upper()=='SCANS':
        return returnlist

def generate_beamprofile_trapezoid(width,length,lengthtop,Nwid,Nlen):
    """Generate a beam profile dict, to a trapezoid approximation.
    
    Inputs:
        width: beam width
        length: beam length
        lengthtop: length of the top of the trapezoid (beam length)
        Nwid: number of points in the beam width direction
        Nlen: number of points in the beam length direction

    Outputs:
        a beam profile dict created by generate_beamprofile().
    
    Notes:
        The beam profile is the outer product of the width and length profile.
        The width profile is a simple upright triangle, the length profile is
        a symmetrical trapezoid, its bottom being <length> long, while its top
        is <lengthtop> long.
        Normalization: areas under width and height profiles are unity.
    
    """
    x=np.linspace(-width*0.5,width*0.5,Nwid)
    y=np.linspace(-length*0.5,length*0.5,Nlen)
    return generate_beamprofile(x,y,lambda x:trapezoidshapefunction(width,0,x),lambda y:trapezoidshapefunction(length,lengthtop,y))

def generate_beamprofile(x,y,funcx,funcy=None):
    """Generate a beam profile dict, which can be fed to directdesmear*()
    
    Inputs:
        x: beam width coordinates (mm), this is the direction parallel to the detector
        y: beam length coordinates (mm), this is the direction orthogonal to the detector
        funcx, funcy: if funcy is not None, both need to be unary functions,
            returning the value of the beam profile in the given direction. If
            funcy is None, funcx needs to be a binary function. In either case,
            they should accept matrices as arguments.
    
    Outputs:
        a beam profile structure with the following keys:
            'x': x coordinates
            'y': y coordinates
            'p': profile matrix (x: top-bottom, y: left-right)
    """
    if funcy is not None:
        p=np.outer(funcx(x),funcy(y))
    else:
        p=np.outer(funcx(x,y))
    return {'x':x,'y':y,'p':p}

def directdesmeargonio(tth,Intensity,Error,beamprofile_or_mat,L,NMC=0):
    """Do a direct desmear (Singh, Ghosh, Shannon) on a scattering curve recorded
    by a goniometer.
    
    Inputs:
        tth: two-theta range. Should be equally spaced.
        Intensity: intensity curve corresponding to tth
        Error: error curve
        beamprofile_or_mat: beam profile dict, as returned by the function
            generate_beamprofile() or a smearing matrix.
        L: sample-detector distance (goniometer radius)
        NMC: number of Monte-Carlo iterations for error propagation        
    Outputs: Idesm, [Edesm], mat
        Idesm: desmeared intensity
        Edesm: error of the desmeared intensity (returned only if NMC>=2)
        mat: smearing matrix
    """
    if type(beamprofile_or_mat)==type({}):
        beamprofile_or_mat=smearingmatrixgonio(tth.min(),tth.max(),len(tth),
                                               beamprofile_or_mat['p'],
                                               beamprofile_or_mat['x'],
                                               beamprofile_or_mat['y'],L)
    idesm=np.linalg.linalg.solve(beamprofile_or_mat,(Intensity).flatten())
    if NMC<2:
        return idesm,beamprofile_or_mat
    edesm=np.zeros(idesm.shape,np.double)
    for i in range(NMC):
        id1=np.linalg.linalg.solve(beamprofile_or_mat,(Intensity+Error*np.random.randn(len(Error))).flatten())
        edesm+=(idesm-id1)**2
    return idesm,np.sqrt(edesm)/(NMC-1),beamprofile_or_mat
                                               
def directdesmearflat(pix,Intensity,Error,beamprofile_or_mat,L,pixelsize,NMC=0):
    """Do a direct desmear (Singh, Ghosh, Shannon) on a scattering curve recorded
    with a flat detector.
    
    Inputs:
        pix: pixel coordinates of the intensity. Should be equally spaced. Pixel
            zero corresponds to the primary beam position.
        Intensity: intensity curve corresponding to pix
        Error: error curve
        beamprofile_or_mat: beam profile dict, as returned by the function
            generate_beamprofile() or a smearing matrix.
        L: sample-detector distance
        pixelsize: pixel size (in mm)
        NMC: number of Monte-Carlo iterations for error propagation.
        
    Outputs: Idesm, [Edesm], mat
        Idesm: desmeared intensity
        Edesm: error of the desmeared intensity (only if NMC>=2)
        mat: smearing matrix
    """
    if type(beamprofile_or_mat)==type({}):
        beamprofile_or_mat=smearingmatrixflat(pix.min(),pix.max(),pixelsize,
                                               beamprofile_or_mat['p'],
                                               beamprofile_or_mat['x'],
                                               beamprofile_or_mat['y'],L)
    idesm=np.linalg.linalg.solve(beamprofile_or_mat,(Intensity).flatten())
    if NMC<2:
        return idesm,beamprofile_or_mat
    edesm=np.zeros(idesm.shape,np.double)
    for i in range(NMC):
        id1=np.linalg.linalg.solve(beamprofile_or_mat,(Intensity+Error*np.random.randn(len(Error))).flatten())
        edesm+=(idesm-id1)**2
    return idesm,np.sqrt(edesm)/(NMC-1),beamprofile_or_mat

def desmearflat(x,Intensity,Error,beamprofile_or_mat,smoothing,L,pixelsize,title='',NMC=10):
    """De-smear scattering curves (flat detector)
    
    Inputs:
        x: pixel coordinates (0 is the beam position)
        Intensity: intensity
        Error: error
        beamprofile_or_mat: a valid input for directdesmearflat()
        smoothing: smoothing parameter
        L: sample-detector distance
        pixelsize: pixel size
        title: title to write over the smoothing diagram
        NMC: Number of Monte Carlo steps for the error propagation.
        
    Outputs:
        Idesm: de-smeared intensity
        Edesm: error of de-smeared intensity
        mat: smearing matrix
    """
    #set up smoothing
    if type(beamprofile_or_mat)==type({}):
        beamprofile_or_mat=smearingmatrixflat(x.min(),x.max(),pixelsize,
                                              beamprofile_or_mat['p'],
                                              beamprofile_or_mat['x'],
                                              beamprofile_or_mat['y'],L)
    def cbfunc(sm,ysm,axes,matrix=beamprofile_or_mat):
        Idesm,mat=directdesmearflat(x,ysm,Error,matrix,L,pixelsize,NMC=0)
        if type(matrix)==type({}):
            matrix['mat']=mat # the smearing matrix won't change during the iterations, better fix it to avoid re-calculation
        axes.plot(x,Idesm)
        axes.set_title(title)
    sm,ysm=guitools.testsmoothing(x,Intensity,smoothing,
                                  slidermin=np.power(10,np.log10(smoothing)-2),
                                  slidermax=np.power(10,np.log10(smoothing)+2),
                                  returnsmoothed=True,callback=cbfunc)
    Idesm,Edesm,mat=directdesmearflat(x,ysm,Error,beamprofile_or_mat,L,
                                      pixelsize,NMC=NMC)
    return Idesm,Edesm,mat

def directdesmear(data,smoothing,params,title='',returnerror=False):
    """Desmear the scattering data according to the direct desmearing
    algorithm by Singh, Ghosh and Shannon
    
    Inputs:
        data: measured intensity vector of arbitrary length (numpy array)
        smoothing: smoothing parameter for smoothcurve(). A scalar
            number. If not exactly known, a dictionary may be supplied with
            the following fields:
                low: lower threshold
                high: upper threshold
                val: initial value
                mode: 'lin' or 'log'
                smoothmode: 'flat', 'hanning', 'hamming', 'bartlett',
                    'blackman' or 'spline', for smoothcurve(). Optionally
                    a 'log' prefix can be applied, see the help text for
                    smoothcurve()
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
        title: display this title over the graph
        returnerror: defaults to False. If true, desmerror is returned.
                
    Outputs: (pixels,desmeared,smoothed,mat,params,smoothing,[desmerror])
        pixels: the pixel coordinates for the resulting curves
        desmeared: the desmeared curve
        smoothed: the smoothed curve
        mat: the desmearing matrix
        params: the desmearing parameters
        smoothing: smoothing parameter
        desmerror: absolute error of the desmeared curve (returned only if
            returnerror was True)
    """
    warnings.warn(DeprecationWarning("Function directdesmear() is deprecated, it will be removed in the future. Use desmearflat() or desmeargonio() instead."))
    #default values
    dparams={'pixelmin':-np.inf,'pixelmax':np.inf,
             'beamnumh':1024,'beamnumv':0}
    dparams.update(params)
    params=dparams
    
    # calculate the matrix
    if params.has_key('matrix') and type(params['matrix'])==np.ndarray:
        A=params['matrix']
    else:
        t=time.time()
        A=smearingmatrix(params['pixelmin'],params['pixelmax'],
                         params['beamcenter'],params['pixelsize'],
                         params['lengthbaseh'],params['lengthtoph'],
                         params['lengthbasev'],params['lengthtopv'],
                         params['beamnumh'],params['beamnumv'])
        t1=time.time()
        print "smearingmatrix took %f seconds" %(t1-t)
        params['matrix']=A
    #x coordinates in pixels
    pixels=np.arange(len(data))
    def smooth_and_desmear(pixels,data,params,smoothing,smmode,returnerror):
        # smoothing the dataset. Errors of the data are sqrt(data), weight will be therefore 1/data
        indices=(pixels<=params['pixelmax']) & (pixels>=params['pixelmin'])
        data=data[indices]
        pixels=pixels[indices]
        data1=fitting.smoothcurve(pixels,data,smoothing,smmode,extrapolate='Linear')
        desmeared=np.linalg.linalg.solve(params['matrix'],data1.reshape(len(data1),1))
        if returnerror:
            desmerror=np.sqrt(np.linalg.linalg.solve(params['matrix']**2,data1.reshape(len(data1),1)))
            ret=(pixels,desmeared,
                 data1,params['matrix'],params,smoothing,desmerror)
        else:
            ret=(pixels,desmeared,
                 data1,params['matrix'],params,smoothing)
        return ret
    if type(smoothing)!=type({}):
        res=smooth_and_desmear(pixels,data,params,smoothing,'spline',returnerror)
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
                                         np.log(smoothing['low']),
                                         np.log(smoothing['high']),
                                         np.log(smoothing['val']))
        elif smoothing['mode']=='lin':
            sl=matplotlib.widgets.Slider(axsl,'Smoothing',
                                         smoothing['low'],
                                         smoothing['high'],
                                         smoothing['val'])
        else:
            raise ValueError('Invalid value for smoothingmode: %s',
                             smoothing['mode'])
        def sliderfun(a=None,sl=sl,ax=ax,mode=smoothing['mode'],x=pixels,
                      y=data,p=params,smmode=smoothing['smoothmode']):
            if mode=='lin':
                sm=sl.val
            else:
                sm=np.exp(sl.val)
            [x1,y1,ysm,A,par,sm]=smooth_and_desmear(x,y,p,sm,smmode,returnerror=False)
            a=ax.axis()
            ax.cla()
            ax.semilogy(x,y,'.',label='Original')
            ax.semilogy(x1,ysm,'-',label='Smoothed (%lg)'%sm)
            ax.semilogy(x1,y1,'-',label='Desmeared')
            ax.legend(loc='best')
            ax.axis(a)
            ax.set_title(title)
            pylab.gcf().show()
            pylab.draw()
        sl.on_changed(sliderfun)
        [x1,y1,ysm,A,par,sm]=smooth_and_desmear(pixels,data,params,smoothing['val'],smoothing['smoothmode'],returnerror=False)
        ax.semilogy(pixels,data,'.',label='Original')
        ax.semilogy(x1,ysm,'-',label='Smoothed (%lg)'%smoothing['val'])
        ax.semilogy(x1,y1,'-',label='Desmeared')
        ax.legend(loc='best')
        ax.set_title(title)
        pylab.gcf().show()
        pylab.draw()
        while not f.donedesmear:
            pylab.waitforbuttonpress()
        if smoothing['mode']=='lin':
            sm=sl.val
        elif smoothing['mode']=='log':
            sm=np.exp(sl.val)
        else:
            raise ValueError('Invalid value for smoothingmode: %s',
                             smoothing['mode'])
        res=smooth_and_desmear(pixels,data,params,sm,smoothing['smoothmode'],returnerror)
        return res    
        
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
        tmp=guitools.findpeak(xdata,ydata,('Zoom to peak %d and press ENTER' % p),peakmode,scaling='log')
        pcoord.append(tmp)
    pcoord=np.array(pcoord)
    n=np.array(peaks)
    a=(n*wavelength)/(2*d)
    x=2*a*np.sqrt(1-a**2)/(1-2*a**2)
    LperH,xcent,LperHerr,xcenterr=fitting.linfit(x,pcoord)
    print 'pixelsize/dist:',1/LperH,'+/-',LperHerr/LperH**2
    print 'beam position:',xcent,'+/-',xcenterr
    if returnq:
        return calcqrangefrom1D(xdata,xcent,LperH,1,wavelength)
    else:
        return 1/LperH,xcent,LperHerr/LperH**2,xcenterr
def calcqrangefrom1D(pixels,beampos,dist,pixelsize,wavelength):
    """Calculate q-range from 1D geometry parameters.
    
    Inputs:
        pixels: list of pixel coordinates (eg. [0,1,2,3,4,5...])
        beampos: beam position, in pixel coordinates
        dist: sample-detector distance
        pixelsize: pixel size (in the same units as dist)
        wavelength: X-ray wavelength
        
    Outputs:
        q-range in a numpy array.
    
    Remarks:
        Although dist and pixelsize both appear as parameters, only their ratio
        is used in this program. The returned q-range is calculated correctly
        (ie. taking the flatness of the detector in account)
    """
    b=(np.array(pixels)-beampos)/(dist/pixelsize)
    return 4*np.pi*np.sqrt(0.5*(b**2+1-np.sqrt(b**2+1))/(b**2+1))/wavelength

def tripcalib(xdata,ydata,peakmode='Lorentz',wavelength=1.54,qvals=2*np.pi*np.array([0.21739,0.25641,0.27027]),returnq=True):
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
        tmp=guitools.findpeak(xdata,ydata,
                     ('Zoom to peak %d (q = %f) and press ENTER' % (p,qvals[p])),
                     peakmode,scaling='lin')
        pcoord.append(tmp)
    pcoord=np.array(pcoord)
    a,b,aerr,berr=fitting.linfit(pcoord,qvals)
    if returnq:
        return a*xdata+b
    else:
        return a,b,aerr,berr

def findbeamasa(asa,beampos=None,oriidx=None):
    """Finds the beam position of an ASA measurement.
    
    Inputs:
        asa: ASA dictionary (with the fields 'pixels' and 'position') or a list of them.
        beampos: if None (default): a figure is presented for user interaction.
            Otherwise it should be a floating point value, the beam position.
        oriidx: if *asa* is not a list, this is ignored. Otherwise this is the
            index in the list according to which the primary beam position is to
            be determined. Special values are None (determine beam position
            one-by-one), or 'avg' (to determine the beam position one-by-one and
            average them).
    """
    if type(asa)!=type([]) and type(asa)!=type(tuple()):
        asa=[asa]
        oriidx=None
    if beampos is not None: #we already have a beam position, set oriidx to None to skip beam finding.
        oriidx=None
    if type(oriidx)==type(1): #find the peak position acccording to the oriidx-th measurement
        pylab.clf()
        beampos=guitools.findpeak(asa[oriidx]['pixels'],asa[oriidx]['position'],
            prompt='Select the beam area and press ENTER or an empty area to cancel.')
    elif oriidx=='avg': # find all beam positions and average
        bps=np.zeros(len(asa))
        for i in range(len(asa)):
            pylab.clf()
            bps[i]=guitools.findpeak(asa[i]['pixel'],asa[i]['position'],
                prompt='Select the beam area and press ENTER or an empty area to cancel.')
        beampos=bps.mean()
    for a in asa:
        # If we are here, two cases are possible:
        #    1) we already have a beampos value
        #    2) beampos is None and oriidx is None.
        if beampos is not None: #1
            beampos1=beampos 
        else: #2
            pylab.clf()
            beampos1=guitools.findpeak(a['pixels'],a['position'],
                prompt='Select the beam area and press ENTER or an empty area to cancel.')
        a['params']['BeamPos']=beampos1
    return

def setdistanceasa(asa,dist,distminus):
    """Set the distance in ASA dicts.
    
    Inputs:
        asa: either a single ASA dict, or a list of them
        dist: distance in mm-s
        distminus: subtractive distance correction. Either a scalar, or a dict,
            with its keys being sample names and optionally None (default distminus).
    
    Outputs:
        none, asa dict(s) will be updated.
    """
    if type(asa)!=type([]) and type(asa)!=type(tuple()):
        asa=[asa]
    if type(distminus)!=type({}):
        distminus={None:distminus}
    for a in asa:
        if a['params']['basename'] in distminus.keys():
            a['params']['Dist']=dist-distminus[a['params']['basename']]
        elif None in distminus.keys():
            a['params']['Dist']=dist-distminus[None]
        else:
            a['params']['Dist']=dist
    return
    
def pixelsizefromagalkanoateasa(asa,peaks,d=48.68,peakmode='Lorentz'):
    hperl=agstcalib(asa['pixels'],asa['position'],peaks,peakmode,asa['params']['wavelength'],d)[0]
    return hperl*asa['params']['Dist']

def setpixelsizeasa(asa,pixelsize):
    if type(asa)!=type([]) and type(asa)!=type(tuple()):
        asa=[asa]
    for a in asa:
        a['params']['PixelSize']=pixelsize
        a['q']=calcqrangefrom1D(a['pixels'],a['params']['BeamPos'],
                                a['params']['Dist'],a['params']['PixelSize'],
                                a['params']['wavelength'])

def setasaparams(asa,**kwargs):
    if type(asa)!=type([]) and type(asa)!=type(tuple()):
        asa=[asa]
    for a in asa:
        for k in kwargs.keys():
            a['params'][k]=kwargs[k]

def processasa(filenames,dist,distminus,pixelsize,beamprofile_or_mat,wavelength=1.54,dirs='.'):
    if type(filenames)==type(''):
        filenames=[filenames]
    asas=B1io.readasa(filenames,dirs)
    setdistanceasa(asas,dist,distminus)
    for a in asas:
        a['params']['wavelength']=wavelength
    findbeamasa(asas)
    setpixelsizeasa(asas,pixelsize)
    for a in asas:
        a['Intensity']=a['position']
        a['Error']=a['poserror']
        if beamprofile_or_mat is None:
            continue
        Idesm,Edesm,mat=desmearflat(a['pixels']-a['params']['BeamPos'],
                                    a['position'],a['poserror'],
                                    beamprofile_or_mat,1,a['params']['Dist'],
                                    a['params']['PixelSize'],
                                    a['params']['Title'],NMC=1000)
        a['Idesm']=Idesm
        a['Edesm']=Edesm
        a['smearingmatrix']=mat
    return asas