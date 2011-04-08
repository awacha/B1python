#-----------------------------------------------------------------------------
# Name:        B1io.py
# Purpose:     I/O components for B1python
#
# Author:      Andras Wacha
#
#-----------------------------------------------------------------------------

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
import warnings
import fitting
import datetime

from c_B1io import cbfdecompress
HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units


try:
    import Image
except ImportError:
    warnings.warn('Cannot import module Image (Python Imaging Library). Only Pilatus300k and Gabriel images can be loaded (Pilatus100k, 1M etc. NOT).')

def readyellowsubmarine(nameformat,fsns=None,dirs='.'):
    """
    """
    if fsns is None:
        filenames=[nameformat]
    else:
        filenames=[nameformat%f for f in fsns]
    if type(dirs)!=type([]) and type(dirs)!=type(tuple()):
        dirs=[dirs]
    datas=[]
    params=[]
    for fn in filenames:
        for d in dirs:
            try:
                f=open(os.path.join(d,fn),'r')
            except IOError:
                continue
            try:
                s=f.read()            
                f.close()
                par={}
                par['FSN']=int(s[2:6])
                par['Owner']=s[6:0x18].split()[0]
                par['Title']='_'.join(s[6:0x18].split()[1:])
                par['MeasTime']=long(s[0x18:0x1e])
                par['Monitor']=long(s[0x1e:0x26])
                par['Day']=int(s[0x26:0x28])
                par['Month']=int(s[0x29:0x2b])
                par['Year']=int(s[0x2c:0x30])
                par['Hour']=int(s[0x30:0x32])
                par['Minute']=int(s[0x33:0x35])
                par['Second']=int(s[0x36:0x38])
                par['PosSample']=int(s[0x60:0x65])
                par['PosBS']=int(s[0x5b:0x60])
                par['PosDetector']=int(s[0x56:0x5b])
                par['max']=long(s[0x38:0x3d])
                par['selector_speed']=long(s[0x3d:0x42])
                par['wavelength']=long(s[0x42:0x44])
                par['Dist_Ech_det']=long(s[0x44:0x49])
                par['comments']=s[0x6d:0x100]
                par['sum']=long(s[0x65:0x6d])
                par['BeamPosX']=float(s[0x49:0x4d])
                par['BeamPosY']=float(s[0x4d:0x51])
                par['AngleBase']=float(s[0x51:0x56])
                par['Datetime']=datetime.datetime(par['Year'],par['Month'],par['Day'],par['Hour'],par['Minute'],par['Second'])
                
                params.append(par)
                datas.append(np.fromstring(s[0x100:],np.uint16).astype(np.double).reshape((64,64)))
                break
            except ValueError:
                print "File %s is invalid! Skipping."%fn
                continue
    if fsns is None:
        return datas[0],params[0]
    else:
        return datas,params


def readcbf(name):
    """Read a cbf (crystallographic binary format) file from a Dectris Pilatus detector.
    
    Inputs:
        name: filename
    
    Output:
        a numpy array of the scattering data
        
    Notes:
        currently only Little endian, "signed 32-bit integer" type and byte-offset compressed data
        are accepted.
    """
    def getvaluefromheader(header,caption,separator=':'):
        tmp=[x.split(separator)[1].strip() for x in header if x.startswith(caption)]
        if len(tmp)==0:
            raise ValueError ('Caption %s not present in CBF header!'%caption)
        else:
            return tmp[0]
    def cbfdecompress_old(data,dim1,dim2):
        index_input=0
        index_output=0
        value_current=0
        value_diff=0
        nbytes=len(data)
        output=np.zeros((dim1*dim2),np.double)
        while(index_input < nbytes):
            value_diff=ord(data[index_input])
            index_input+=1
            if value_diff !=0x80:
                if value_diff>=0x80:
                    value_diff=value_diff -0x100
            else: 
                if not ((ord(data[index_input])==0x00 ) and 
                    (ord(data[index_input+1])==0x80)):
                    value_diff=ord(data[index_input])+\
                                0x100*ord(data[index_input+1])
                    if value_diff >=0x8000:
                        value_diff=value_diff-0x10000
                    index_input+=2
                else:
                    index_input+=2
                    value_diff=ord(data[index_input])+\
                               0x100*ord(data[index_input+1])+\
                               0x10000*ord(data[index_input+2])+\
                               0x1000000*ord(data[index_input+3])
                    if value_diff>=0x80000000L:
                        value_diff=value_diff-0x100000000L
                    index_input+=4
            value_current+=value_diff
#            print index_output
            try:
                output[index_output]=value_current
            except IndexError:
                print "End of output array. Remaining input bytes:", len(data)-index_input
                print "remaining buffer:",data[index_input:]
                break
            index_output+=1
        if index_output != dim1*dim2:
            print "index_output is ",index_output-1
            print "dim1 is",dim1
            print "dim2 is",dim2
            print "dim1*dim2 is",dim1*dim2
            raise ValueError, "Binary data does not have enough points."
        return output.reshape((dim2,dim1))
    f=open(name,'rb')
    cbfbin=f.read()
    f.close()
    datastart=cbfbin.find('%c%c%c%c'%(12,26,4,213))+4
    header=[x.strip() for x in cbfbin[:datastart].split('\n')]
    if getvaluefromheader(header,'X-Binary-Element-Type')!='"signed 32-bit integer"':
        raise NotImplementedError('element type is not "signed 32-bit integer" in CBF, but %s.' % getvaluefromheader(header,'X-Binary-Element-Type'))
    if getvaluefromheader(header,'conversions','=')!='"x-CBF_BYTE_OFFSET"':
        raise NotImplementedError('compression is not "x-CBF_BYTE_OFFSET" in CBF!')
    dim1=long(getvaluefromheader(header,'X-Binary-Size-Fastest-Dimension'))
    dim2=long(getvaluefromheader(header,'X-Binary-Size-Second-Dimension'))
    nbytes=long(getvaluefromheader(header,'X-Binary-Size'))
    return cbfdecompress(cbfbin[datastart:datastart+nbytes],dim1,dim2)



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

def readasa(basename,dirs=[]):
    """Load SAXS/WAXS measurement files from ASA *.INF, *.P00 and *.E00 files.
    
    Input:
        basename: the basename (without extension) of the files. Can also be a
            list of strings
        dirs: list of directories (or just a single directory) to search files
            in. P00, INF and E00 should reside in the same directory.
    Output:
        An ASA dictionary (or a list of them) with the following fields:
            position: the counts for each pixel (numpy array), in cps
            energy: the energy spectrum (numpy array), in cps
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
            poserror: estimated error of the position (cps)
            energyerror: estimated error of the energy (cps)
    """
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if type(basename)==type(''):
        basenames=[basename]
        basename_scalar=True
    else:
        basenames=basename
        basename_scalar=False
    ret=[]
    for basename in basenames:
        for d in dirs:
            try:
                p00=np.loadtxt(os.path.join(d,'%s.P00' % basename))
            except IOError:
                try:
                    p00=np.loadtxt(os.path.join(d,'%s.p00' % basename))
                except:
                    p00=None
            if p00 is not None:
                p00=p00[1:] # cut the leading -1
            try:
                e00=np.loadtxt(os.path.join(d,'%s.E00' % basename))
            except IOError:
                try:
                    e00=pylab.loadtxt(os.path.join(d,'%s.e00' % basename))
                except:
                    e00=None
            if e00 is not None:
                e00=e00[1:] # cut the leading -1
            try:
                inffile=open(os.path.join(d,'%s.inf' % basename))
            except IOError:
                try:
                    inffile=open(os.path.join(d,'%s.Inf' % basename))
                except IOError:
                    try:
                        inffile=open(os.path.join(d,'%s.INF' % basename))
                    except:
                        inffile=None
                        params=None
            if (p00 is not None) and (e00 is not None) and (inffile is not None):
                break
            else:
                p00=None
                e00=None
                inffile=None
        if (p00 is None) or (e00 is None) or (inffile is None):
            print "Cannot find every file (*.P00, *.INF, *.E00) for sample %s in any directory" %basename
            continue
        if inffile is not None:
            params={}
            l1=inffile.readlines()
            l=[]
            for line in l1:
                if len(line.strip())>0:
                    l.append(line)
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
        ret.append({'position':p00/params['Livetime'],'energy':e00/params['Livetime'],
                'params':params,'pixels':pylab.arange(len(p00)),
                'poserror':np.sqrt(p00)/params['Livetime'],
                'energyerror':np.sqrt(e00)/params['Livetime']})
    if basename_scalar:
        return ret[0]
    else:
        return ret
def readheader(filename,fsn=None,fileend=None,dirs=[],quiet=False):
    """Reads header data from measurement files
    
    Inputs:
        filename: the beginning of the filename, or the whole filename, if fsn is None.
            If fsn is not None but fileend is, filename should be a printf-style format string,
            containing a single place to substitute the FSN number.
        fsn: the file sequence number or None if the whole filenam was supplied
            in filename. It can be a list as well.
        fileend: the end of the file. If it ends with .gz, then the file is
            treated as a gzip archive.
        dirs [optional]: a list of directories to try
        quiet: True if no warning messages should be printed
        
    Output:
        A list of header dictionaries. An empty list if no headers were read.
        
    Examples:
        read header data from 'ORG000123.DAT':
        
        header=readheader('ORG',123,'.DAT')
        
        or
        
        header=readheader('ORG00123.DAT')

        or
        
        header=readheader('ORG%05d.DAT',123)
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
    elif fileend is None:
        try:
            fsn[0]
        except TypeError:
            fsn=[fsn]
        try:
            names=[filename % x for x in fsn]
        except TypeError:
            raise ValueError("If fileend is None in readheader(), filename should be a format string!")
    else:
        try:
            names=['%s%05d%s' % (filename,x,fileend ) for x in fsn]
        except:
            names=['%s%05d%s' % (filename,fsn,fileend)]
    headers=[]
    for name in names:
        filefound=False
        for d in dirs:
            try:
                name1=os.path.join(d,name)
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
        if not filefound and not quiet:
            print 'readheader: Cannot find file %s in given directories.' % name
    return headers
def read2dB1data(filename,files=None,fileend=None,dirs=[],quiet=False):
    """Read 2D measurement files, along with their header data

    Inputs:
        filename: the beginning of the filename, or the whole filename
        files: the file sequence number or None if the whole filenam was supplied
            in filename. It is possible to give a list of fsns here.
        fileend: the end of the file.
        dirs [optional]: a list of directories to try
        quiet: True if no warning messages should be printed
        
    Outputs:
        A list of 2d scattering data matrices
        A list of header data
        
    Examples:
        Read FSN 123-130:
        a) measurements with the Gabriel detector:
        data,header=read2dB1data('ORG',range(123,131),'.DAT')
        b) measurements with a Pilatus* detector:
        #in this case the org_*.header files should be present in the same folder
        data,header=read2dB1data('org_',range(123,131),'.tif')
    """
    def readgabrieldata(filename,dirs,quiet=False):
        for d in dirs:
            try:
                filename1=os.path.join(d,filename)
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
        if not quiet:
            print 'Cannot find file %s. Tried directories:' % filename,dirs
        return None
    def readpilatusdata(filename,dirs,useCBF=False,quiet=False):
        oldloader=False
        for d in dirs:
            try:
                if useCBF:
                    data=readcbf(os.path.join(d,filename))
                    return data
                oldloader=False
                filename1=os.path.join(d,filename)
                im=Image.open(filename1)
                data=np.array(im.getdata(),'uint32').reshape(np.flipud(im.size))
                oldloader=False
                return data
            except IOError:
                if not quiet:
                    print "Tried file %s with no luck" % os.path.join(d,filename)
                pass
            except NameError:
                warnings.warn('Advanced loading of Pilatus images is disabled, since module Image (Python Imaging Library) is unavailable.')
                oldloader=True
        if oldloader:
            return readpilatus300kdata(filename,dirs,quiet=quiet)
        else:
            if not quiet:
                print 'Cannot find file %s. Make sure the path is correct.' % filename
            return None
            
    def readpilatus300kdata(filename,dirs,quiet=False): # should work for other detectors as well
        for d in dirs:
            try:
                filename1=os.path.join(d,filename)
                fid=open(filename1,'rb');
                datastr=fid.read();
                fid.close();
                data=pylab.fromstring(datastr[4096:],'uint32').reshape((619,487)).astype('double')
                return data;                
            except IOError:
                pass
        if not quiet:
            print 'Cannot find file %s. Make sure the path is correct.' % filename
        return None
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    if fileend is None:
        fileend=filename[string.rfind(filename,'.'):]
    if (files is not None):
        try:
            len(files)
        except:
            files=[files];
    if fileend.upper()=='.HEADER':
        fileend='.TIF'
    if fileend.upper()=='.CBF':
        useCBF=True
    else:
        useCBF=False
    
    if fileend.upper()=='.TIF' or fileend.upper()=='.TIFF' or useCBF: # pilatus300k mode
        filebegin=filename[:string.rfind(filename,'.')]
        if files is None:
            header=readheader(filebegin+'.header',dirs=dirs,quiet=quiet)
            data=readpilatusdata(filename,dirs=dirs,useCBF=useCBF,quiet=quiet)
            if (len(header)<1) or (data is None):
                return [],[]
            else:
                header=header[0]
                header['Detector']='Pilatus'
                return [data],[header]
        else:
            header=[];
            data=[];
            for fsn in files:
                tmp1=readheader('%s%05d%s' %(filename,fsn,'.header'),dirs=dirs,quiet=quiet)
                tmp2=readpilatusdata('%s%05d%s'%(filename,fsn,fileend),dirs=dirs,useCBF=useCBF,quiet=quiet)
                if (len(tmp1)>0) and (tmp2 is not None):
                    tmp1=tmp1[0]
                    tmp1['Detector']='Pilatus'
                    header.append(tmp1)
                    data.append(tmp2)
            return data,header
    else: # Gabriel mode, if fileend is neither TIF, nor TIFF, case insensitive
        if files is None: # read only 1 file
            header=readheader(filename,dirs=dirs,quiet=quiet);
            data=readgabrieldata(filename,dirs=dirs,quiet=quiet);
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
                tmp1=readheader('%s%05d%s' % (filename,fsn,fileend),dirs=dirs,quiet=quiet)
                tmp2=readgabrieldata('%s%05d%s' % (filename,fsn,fileend),dirs=dirs,quiet=quiet)
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
    if showtitles.upper().startswith('GABRIEL'):
        print 'FSN\tTime\tEnergy\tDist\tPos\tTransm\tSum/Tot %\tT (C)\tTitle\t\t\tDate'
    elif showtitles.upper().startswith('PILATUS'):
        print 'FSN\tTime\tEnergy\tDist\tPos\tTransm\tTitle\t\t\tDate'
        if fileend.upper()=='.HEADER':
            fileend='.tif'
    else:
        pass #do not print header
    for i in files:
        d,h=read2dB1data(filename,i,fileend,dirs,quiet=True);
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
def read2dintfile(fsns,dirs=[],norm=True,quiet=False):
    """Read corrected intensity and error matrices
    
    Input:
        fsns: one or more fsn-s in a list
        dirs: list of directories to try
        norm: True if int2dnorm*.mat file is to be loaded, False if
            int2darb*.mat is preferred. You can even supply the file prefix
            itself.
        quiet: True if no warning messages should be printed
        
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
        If the shape of the loaded error matrix is not equal to that of the
        intensity, the error matrix is overridden with a zero matrix.
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
    if np.isscalar(fsns):
        fsns=[fsns]
    int2d=[]
    err2d=[]
    params=[]
    for fsn in fsns: # this also works if len(fsns)==1
        filefound=False
        for d in dirs:
            try: # first try to load the npz file. This is the most effective way.
                if type(norm)==type(''):
                    fileprefixnorm=norm
                elif norm:
                    fileprefixnorm='int2dnorm'
                else:
                    fileprefixnorm='int2darb'
                tmp0=np.load(os.path.join(d,'%s%d.npz' % (fileprefixnorm,fsn)))
                tmp=tmp0['Intensity']
                tmp1=tmp0['Error']
            except IOError:
                try: # first try to load the mat file. This is the second effective way.
                    tmp0=scipy.io.loadmat(os.path.join(d,'%s%d.mat' % (fileprefixnorm,fsn)))
                    tmp=tmp0['Intensity']
                    tmp1=tmp0['Error']
                except IOError: # if mat file is not found, try the ascii files
                    if type(norm)==type(''):
                        warnings.warn(SyntaxWarning('Loading 2D ascii files when parameter <norm> for read2dintfile() is a string.'))
                        continue # try from another directory
                    if norm:
    #                    print 'Cannot find file int2dnorm%d.mat: trying to read int2dnorm%d.dat(.gz|.zip) and err2dnorm%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
                        tmp=read2dascii('%s%sint2dnorm%d.dat' % (d,os.sep,fsn));
                        tmp1=read2dascii('%s%serr2dnorm%d.dat' % (d,os.sep,fsn));
                    else:
    #                    print 'Cannot find file int2darb%d.mat: trying to read int2darb%d.dat(.gz|.zip) and err2darb%d.dat(.gz|.zip)' %(fsn,fsn,fsn)
                        tmp=read2dascii('%s%sint2darb%d.dat' % (d,os.sep,fsn));
                        tmp1=read2dascii('%s%serr2darb%d.dat' % (d,os.sep,fsn));
                except TypeError: # if mat file was found but scipy.io.loadmat was unable to read it
                    if not quiet:
                        print "Malformed MAT file! Skipping."
                    continue # try from another directory
            if (tmp is not None) and (tmp1 is not None): # if all of int,err and log is read successfully
                filefound=True
#                print 'Files corresponding to fsn %d were found.' % fsn
                break # file was found, do not try to load it again from another directory
        if filefound:
            tmp2=readlogfile(fsn,dirs=dirs)[0]
            if len(tmp2)>0:
                int2d.append(tmp)
                if tmp1.shape!=tmp.shape:    # test if the shapes of intensity and error matrices are the same. If not, let the error matrix be a zero matrix of the same size as the intensity.
                    tmp1=np.zeros(tmp.shape)
                err2d.append(tmp1)
                params.append(tmp2)                
        if not filefound and not quiet:
            print "read2dintfile: Cannot find file(s ) for FSN %d" % fsn
    return int2d,err2d,params # return the lists
def write2dintfile(A,Aerr,params,norm=True,filetype='npz'):
    """Save the intensity and error matrices to int2dnorm<FSN>.mat
    
    Inputs:
        A: the intensity matrix
        Aerr: the error matrix (can be None, if no error matrix is to be saved)
        params: the parameter dictionary
        norm: if int2dnorm files are to be saved. If it is false, int2darb files
            are saved (arb = arbitrary units, ie. not absolute intensity). If a string,
            save it to <norm>%d.<filetype>.
        filetype: 'npz' or 'mat'
    int2dnorm<FSN>.[mat or npz] is written. The parameter structure is not
        saved, since it should be saved already in intnorm<FSN>.log
    """
    if Aerr is None:
        Aerr=np.zeros((1,1))
    if type(norm)==type(''):
        fileprefix='%s%d' % (norm,params['FSN'])
    elif norm:
        fileprefix='int2dnorm%d' % params['FSN']
    else:
        fileprefix='int2darb%d' % params['FSN']
    if filetype.upper() in ['NPZ','NPY','NUMPY']:
        np.savez('%s.npz' % fileprefix, Intensity=A,Error=Aerr)
    elif filetype.upper() in ['MAT','MATLAB']:
        scipy.io.savemat('%s.mat' % fileprefix,{'Intensity':A,'Error':Aerr});
    else:
        raise ValueError,"Unknown file type: %s" % repr(filetype)
def readintfile(filename,dirs=[],sanitize=True,quiet=False):
    """Read intfiles.

    Input:
        filename: the file name, eg. intnorm123.dat
        dirs [optional]: a list of directories to try
        quiet: True if no warning messages should be printed

    Output:
        A dictionary with 'q' holding the values for the momentum transfer,
            'Intensity' being the intensity vector and 'Error' has the error
            values. These three fields are numpy ndarrays. An empty dict
            is returned if file is not found.
    """
    fields=['q','Intensity','Error','Area','qavg','qstd']
    
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
            ni=np.loadtxt(fname)
            for k,i in zip(fields,range(ni.shape[1])):
                ret[k]=ni[:,i]
            ret=utils.SASDict(**ret)
            if sanitize:
                ret.sanitize()
            break # file was found, do not iterate over other directories
        except IOError:
            continue
    if len(ret)==0 and not quiet:
        print "readintfile: could not find file %s in given directories." % filename
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
def readintnorm(fsns, filetype='intnorm',dirs=[],logfiletype='intnorm',quiet=False):
    """Read intnorm*.dat files along with their headers
    
    Inputs:
        fsns: one or more fsn-s.
        filetype: prefix of the filename
        logfiletype: prefix of the log filename
        dirs [optional]: a list of directories to try
        quiet: True if no warning messages should be printed
        
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
            tmp=readintfile(filename,quiet=quiet)
            if len(tmp)>0:
                currdata=tmp
                break # file was already found, do not try in another directory
        currlog=readlogfile(fsn,dirs,norm=logfiletype,quiet=quiet)
        if len(currdata)>0 and len(currlog)>0:
            data.append(currdata);
            param.append(currlog[0]);
    return data,param
def readbinned(fsn,dirs=[],quiet=False):
    """Read intbinned*.dat files along with their headers.
    
    This is a shortcut to readintnorm(fsn,'intbinned',dirs)
    """
    return readintnorm(fsn,'intbinned',dirs,quiet=quiet);
def readsummed(fsn,**kwargs):
    """Read summed*.dat files along with their headers.
    
    All arguments are forwarded to readintnorm().
    """
    return readintnorm(fsn,filetype='summed',**kwargs)
def readunited(fsn,**kwargs):
    """Read united*.dat files along with their headers.
    
    All arguments are forwarded to readintnorm().
    """
    return readintnorm(fsn,filetype='united',**kwargs)
    
def readlogfile(fsn,dirs=[],norm=True,quiet=False):
    """Read logfiles.
    
    Inputs:
        fsn: the file sequence number(s). It is possible to
            give a single value or a list
        dirs [optional]: a list of directories to try
        norm: if a normalized file is to be loaded (intnorm*.log). If
            False, intarb*.log will be loaded instead. Or, you can give a
            string. In that case, '%s%d.log' %(norm, <FSN>) will be loaded.
        quiet: True if no warning messages should be printed
            
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
    if np.isscalar(fsn):
        fsn=[fsn]
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
            except IOError:
                #print 'Cannot find file %s.' % filename
                pass
        if not filefound and not quiet:
            print 'Cannot find file %s in any of the given directories.' % filebasename
    return params;
            
def writelogfile(header,ori,thick,dc,realenergy,distance,mult,errmult,reffsn,
                 thickGC,injectionGC,injectionEB,pixelsize,mode='Pilatus',norm=True):
    """Write logfiles.
    
    Inputs:
        header: header structure as read by readheader()
        ori: origin vector of 2
        thick: thickness of the sample (cm)
        dc: if mode=='Pilatus' then this is the DC level which is subtracted.
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
        mode: 'Pilatus' or 'Gabriel'. If invalid, it defaults to 'Gabriel'
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
    if mode.upper().startswith('PILATUS'):
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
def readabt(filename,dirs='.'):
    """Read abt_*.fio type files.
    
    Input:
        filename: the name of the file.
        dirs: directories to search for files in
        
    Output:
        A dictionary with the following fields:
            'title': the sample title
            'mode': 'Energy' or 'Motor'
            'columns': the description of the columns in 'data'
            'data': the data found in the file, in a matrix.
            'dataset': a structured array a la numpy, containing the same data
                as in 'data', but in another representation.
    """
    if type(dirs)==type(''):
        dirs=[dirs]
    if len(dirs)==0:
        dirs=['.']
    ret={}
    for d in dirs:
        try:
            f=open(os.path.join(d,filename),'rt');
        except IOError:
            print 'Cannot open file %s' % filename
            continue
        # now the file is open
        lines=f.readlines()
        #prune comment lines starting with an exclamation mark (!).
        i=0
        while i<len(lines):
            lines[i]=lines[i].strip()
            if lines[i].startswith('!') or len(lines[i])==0:
                lines.pop(i)
                i-=1
            i+=1
        # find the parameter part
        ret['params']={}
        idx=lines.index('%p')+1
        while idx<len(lines) and (not lines[idx].startswith('%')):
            ls=lines[idx].split('=')
            if len(ls)==2:
                ret['params'][ls[0].strip()]=float(ls[1].strip())
            idx+=1
        # find the comment part
        idx=lines.index('%c')
        # first comment line is like: MOT12-Scan started at 21-Sep-2009 13:43:56, ended 13:47:53
        l=lines[idx+1]
        if l.startswith('MOT'):
            ret['mode']='Motor'
        elif l.startswith('ENERGY'):
            ret['mode']='Energy'
        else:
            print 'Unknown scan type!'
            return None
        # find the string containing the start time in dd-mmm-yyyy hh:mm:ss format
        str=l[(l.index('started at ')+len('started at ')):l.index(', ended')]
        date,time1=str.split(' ')
        ret['params']['day'],ret['params']['month'],ret['params']['year']=date.split('-')
        ret['params']['day']=int(ret['params']['day'])
        ret['params']['year']=int(ret['params']['year'])
        ret['params']['hourstart'],ret['params']['minutestart'],ret['params']['secondstart']=[int(x) for x in time1.split(':')]
        str=l[l.index('ended')+len('ended '):]
        ret['params']['hourend'],ret['params']['minuteend'],ret['params']['secondend']=[int(x) for x in str.split(':')]
        
        l=lines[idx+2]
        if l.startswith('Name:'):
            ret['name']=l.split(':')[1].split()[0]
        else:
            raise ValueError('File %s is invalid!' % filename)
        
        l=lines[idx+3]
        if l.startswith('Counter readings are'):
            ret['title']=''
            idx-=1
        else:
            ret['title']=l.strip()
        
        #idx+4 is "Counter readings are offset corrected..."
        l=lines[idx+5]
        if not l.startswith('%'):
            ret['offsets']={}
            lis=l.split()
            while len(lis)>0:
                ret['offsets'][lis.pop()]=float(lis.pop())
        idx=lines.index('%d')+1
        ret['columns']=[];
        while lines[idx].startswith('Col'):
            ret['columns'].append(lines[idx].split()[2][10:])
            idx+=1
        datalines=lines[idx:]
        for i in range(len(datalines)):
            datalines[i]=[float(x) for x in datalines[i].split()]
        ret['data']=np.array(datalines)
        ret['dataset']=np.array([tuple(a) for a in ret['data'].tolist()], dtype=zip(ret['columns'],[np.double]*len(ret['columns'])))
        return ret;
    return None
def savespheres(spheres,filename):
    """Save sphere structure in a file.
    
    Inputs:
        spheres: sphere matrix
        filename: filename
    """
    pylab.savetxt(filename,spheres,delimiter='\t')
def readmask(filename,fieldname=None,dirs='.'):
    """Try to load a maskfile (matlab(R) matrix file)
    
    Inputs:
        filename: the input file name
        fieldname: field in the mat file. None to autodetect.
        dirs: list of directory names to try
        
    Outputs:
        the mask in a numpy array of type np.uint8
    """
    if type(dirs)==type(''):
        dirs=[dirs]
    f=None
    for d in dirs:
        try:
            f=scipy.io.loadmat(os.path.join(d,filename))
        except IOError:
            f=None
            continue
        else:
            break
    if f is None:
        raise IOError('Cannot find mask file in any of the given directories!')
    if fieldname is not None:
        return f[fieldname].astype(np.uint8)
    else:
        validkeys=[k for k in f.keys() if not (k.startswith('_') and k.endswith('_'))];
        if len(validkeys)<1:
            raise ValueError('mask file contains no masks!')
        if len(validkeys)>1:
            raise ValueError('mask file contains multiple masks!')
        return f[validkeys[0]].astype(np.uint8)
def writeparamfile(filename,param):
    """Write the param structure into a logfile. See writelogfile() for an explanation.
    
    Inputs:
        filename: name of the file.
        param: param structure (dictionary).
        
    Notes:
        exceptions pass through to the caller.
    """
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
    logfile_dict_str={'Sample name':'Title',
                      'Sample title':'Title'}
    #this dict. contains the bool parameters
    logfile_dict_bool={'Injection between Empty beam and sample measurements?':'InjectionEB',
                       'Injection between Glassy carbon and sample measurements?':'InjectionGC'
                       }
    logfile_dict_list={'FSNs':'FSNs'}
    logfile_order=['FSN','Title','Dist','Thickness','Transm','PosSample',
                   'Temperature','MeasTime','ScatteringFlux','dclevel','FSNdc',
                   'FSNempty','InjectionEB','FSNref1','Thicknessref1',
                   'InjectionGC','Energy','EnergyCalibrated',('BeamPosX','BeamPosY'),
                   'NormFactor','NormFactorRelativeError',('BeamsizeX','BeamsizeY'),
                   'PixelSize','Monitor','PrimaryIntensity','RotXsample','RotYsample']
    allkeys=param.keys()
    def boolrepr(x):
        if x:
            return 'y'
        else:
            return 'n'
    listrepr=lambda l:' '.join([str(x) for x in l])
    f=open(filename,'wt')
    def writekey(k,allkeys=allkeys):
        #try if it is a float argument
        k1=[x for x in logfile_dict_float.keys() if logfile_dict_float[x]==k]
        if len(k1)>0:
            if type(k)==type(''):
                k=[k,]
            k=[x for x in k if x in allkeys]
            if len(k)==0:
                return
            f.write(k1[0])
            f.write(':\t')
            f.write(' '.join([str(param[x]) for x in k]))
            f.write('\n')
            for x in k:
                allkeys.remove(x)
            return
        #try if it is a str argument
        k1=[x for x in logfile_dict_str.keys() if logfile_dict_str[x]==k]
        if len(k1)>0:
            if type(k)==type(''):
                k=[k,]
            k=[x for x in k if x in allkeys]
            if len(k)==0:
                return
            f.write(k1[0])
            f.write(':\t')
            f.write(' '.join([str(param[x]) for x in k]))
            f.write('\n')
            for x in k:
                allkeys.remove(x)
            return
        #try if it is a bool argument
        k1=[x for x in logfile_dict_bool.keys() if logfile_dict_bool[x]==k]
        if len(k1)>0:
            if type(k)==type(''):
                k=[k,]
            k=[x for x in k if x in allkeys]
            if len(k)==0:
                return
            f.write(k1[0])
            f.write(':\t')
            f.write(' '.join([boolrepr(param[x]) for x in k]))
            f.write('\n')
            for x in k:
                allkeys.remove(x)
            return
        #try if it is a list
        k1=[x for x in logfile_dict_list.keys() if logfile_dict_list[x]==k]
        if len(k1)>0:
            if type(k)==type(''):
                k=[k,]
            k=[x for x in k if x in allkeys]
            if len(k)==0:
                return
            f.write(k1[0])
            f.write(':\t')
            f.write(' '.join([listrepr(param[x]) for x in k]))
            f.write('\n')
            for x in k:
                allkeys.remove(x)
            return
                
    for k in logfile_order:
        writekey(k)
    for k in allkeys:
        writekey(k)
    f.close()