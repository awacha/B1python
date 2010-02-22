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

import pylab
import numpy as np
import types
import os
import gzip
import zipfile
import string
import scipy.io
import utils
import B1macros

class int1d(dict):
    pass

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
                pass #continue with the next directory
        if not filefound:
            print 'readheader: Cannot find file %s.' % name
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
                fid.rewind()
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
        filefound=False
        for d in dirs:
            try: # first try to load the mat file. This is the most effective way.
                if norm:
                    tmp0=scipy.io.loadmat('%s%sint2dnorm%d.mat' % (d,os.sep,fsn))
                else:
                    tmp0=scipy.io.loadmat('%s%sint2darb%d.mat' % (d,os.sep,fsn))
                tmp=tmp0['Intensity'].copy()
                tmp1=tmp0['Error'].copy()
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
                break # file was found, do not try to load it again from another directory
        if not filefound:
            print "read2dintfile: Cannot find file(s ) for FSN %d" % fsn
    return int2d,err2d,params # return the lists
def write2dintfile(A,Aerr,params,norm=True):
    """Save the intensity and error matrices to int2dnorm<FSN>.mat
    
    Inputs:
        A: the intensity matrix
        Aerr: the error matrix
        params: the parameter dictionary
        
    int2dnorm<FSN>.mat is written. The parameter structure is not saved,
        since it should be saved already in intnorm<FSN>.log
    """
    if norm:
        filename='int2dnorm%d.mat' % params['FSN'];
    else:
        filename='int2darb%d.mat' % params['FSN'];
    scipy.io.savemat(filename,{'Intensity':A,'Error':Aerr});
def readintfile(filename,dirs=[]):
    """Read intfiles.

    Input:
        filename: the file name, eg. intnorm123.dat
        dirs [optional]: a list of directories to try

    Output:
        A dictionary with 'q' holding the values for the momentum transfer,
            'Intensity' being the intensity vector and 'Error' has the error
            values. These three fields are numpy ndarrays.
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
            False, intarb*.log will be loaded instead.
            
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
            if norm:
                filename='%s%sintnorm%d.log' % (d,os.sep,f) #the name of the file
            else:
                filename='%s%sintarb%d.log' % (d,os.sep,f) #the name of the file
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
        for d in dirs:
            try:
                filename='%s%swaxs_%05d.cor' % (d,os.sep,fsn)
                tmp=pylab.loadtxt(filename)
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
            d['Energy']=B1macros.energycalibration(energymeas,energycalib,pylab.array(energy[0]))
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
def savespheres(spheres,filename):
    """Save sphere structure in a file.
    
    Inputs:
        spheres: sphere matrix
        filename: filename
    """
    pylab.savetxt(filename,spheres,delimiter='\t')
