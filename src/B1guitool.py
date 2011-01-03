#!/usr/bin/env python
import Tkinter
import matplotlib
matplotlib.use("TkAgg")
import B1python
import pylab
import time
import numpy as np
import os
import scipy.io

configfilename=os.path.expanduser('.B1guitool')
default_inputdirs=''
#default_inputdirs=r'r:\misc\jusifa\Projekte\2010\0420Bota; r:\misc\jusifa\Projekte\2010\0420Bota\data1; r:\misc\jusifa\Projekte\2010\0420Bota\eval1; r:\misc\jusifa\Projekte\2010\0420Bota\processing'
default_outputdir='.'
default_mask='mask.mat'
try:
    f=open(configfilename,'rt');
    lines=f.readlines();
    for l in lines:
        if l.startswith('Default inputdirs='):
            default_inputdirs=l.strip().split('=')[1].strip()
        elif l.startswith('Default outputdir='):
            default_outputdir=l.strip().split('=')[1].strip()
        elif l.startswith('Default mask='):
            default_mask=l.strip().split('=')[1].strip()
    f.close()
except:
    pass

class MainWindow:
    def __init__(self,master):
#        master.tk_setPalette('lightblue')
#        master.tk_strictMotif(True)
        self.inputdirlist=None
        master.protocol("WM_DELETE_WINDOW",self.quit)
        self.figure=pylab.figure()
        self.figurenum=self.figure.number
        #self.figure.canvas.manager.window.protocol("WM_DELETE_WINDOW",self.figure.canvas.manager.window.withdraw)
        self.master=master
        frame=Tkinter.Frame(master)
        master.columnconfigure(0,weight=1)
        master.rowconfigure(0,weight=1)
        frame.columnconfigure(1,weight=1)
        frame.grid(sticky='NSEW')
        master.wm_title('B1guitool powered by B1python v%s'%B1python.VERSION)
        Tkinter.Label(frame,text='Input directories (separated by semicolons)').grid(row=0,column=0,sticky='NSW')
        self.inputdirs=Tkinter.Entry(frame)
        self.inputdirs.grid(row=0,column=1,sticky='NSEW')
        self.inputdirs.insert(0,default_inputdirs)
        Tkinter.Label(frame,text='Output directory').grid(row=1,column=0,sticky='NSW')
        self.outputdir=Tkinter.Entry(frame)
        self.outputdir.grid(row=1,column=1,sticky='NSEW')
        self.outputdir.insert(0,default_outputdir)
        Tkinter.Label(frame,text='File sequence number (FSN)').grid(row=2,column=0,sticky='NSW')
        self.fsn=Tkinter.Spinbox(frame,from_=0,to=99999,command=self.newfsn)
        self.fsn.bind('<Return>',self.newfsn)
        self.fsn.grid(row=2,column=1,sticky='NSEW')
        f=Tkinter.Frame(frame)
        f.grid(row=3,column=0,columnspan=2,sticky="NSEW")
        lf=Tkinter.LabelFrame(f,text='Plot type')
        lf.grid(row=0,column=0,columnspan=1,sticky="NSEW")
        self.rbvar=Tkinter.IntVar()
        Tkinter.Radiobutton(lf,text='Two-dimensional image',variable=self.rbvar,value=1,command=self.newplottype).pack(anchor='w')
        Tkinter.Radiobutton(lf,text='Radial average',variable=self.rbvar,value=2,command=self.newplottype).pack(anchor='w')
        Tkinter.Radiobutton(lf,text='Azimuthal average',variable=self.rbvar,value=3,command=self.newplottype).pack(anchor='w')
        self.rbvar.set(1)
        f1=Tkinter.Frame(f)
        f1.grid(row=0,column=1,sticky="NSEW")
        f1.columnconfigure(1,weight=1)
        self.qminmanual=Tkinter.IntVar()
        Tkinter.Checkbutton(f1,onvalue=1,offvalue=0,variable=self.qminmanual,text='q min:',command=self.toggleqminqmax).grid(row=0,column=0,sticky="NSW")
        self.qmin=Tkinter.Entry(f1)
        self.qmin.grid(row=0,column=1,sticky='NSEW')
        self.qmin.insert(0,'0')
        self.qmaxmanual=Tkinter.IntVar()
        Tkinter.Checkbutton(f1,onvalue=1,offvalue=0,variable=self.qmaxmanual,text='q max:',command=self.toggleqminqmax).grid(row=1,column=0,sticky="NSW")
        self.qmax=Tkinter.Entry(f1)
        self.qmax.grid(row=1,column=1,sticky='NSEW')
        self.qmax.insert(0,'0')
        self.Nqmanual=Tkinter.IntVar()
        Tkinter.Checkbutton(f1,onvalue=1,offvalue=0,variable=self.Nqmanual,text='Nr of bins:',command=self.toggleqminqmax).grid(row=2,column=0,sticky="NSW")
        self.Nq=Tkinter.Entry(f1)
        self.Nq.grid(row=2,column=1,sticky='NSEW')
        self.Nq.insert(0,'100')

        self.phi0manual=Tkinter.IntVar()
        Tkinter.Checkbutton(f1,onvalue=1,offvalue=0,variable=self.phi0manual,text='Phi0',command=self.toggleqminqmax).grid(row=0,column=2,sticky="NSW")
        self.phi0=Tkinter.Entry(f1)
        self.phi0.grid(row=0,column=3)
        self.phi0.insert(0,'0')

        self.dphimanual=Tkinter.IntVar()
        Tkinter.Checkbutton(f1,onvalue=1,offvalue=0,variable=self.dphimanual,text='dPhi',command=self.toggleqminqmax).grid(row=1,column=2,sticky="NSW")
        self.dphi=Tkinter.Entry(f1)
        self.dphi.grid(row=1,column=3)
        self.dphi.insert(0,'0')


        self.toggleqminqmax()
        f1=Tkinter.Frame(f)
        f.columnconfigure(3,weight=1)
        f.columnconfigure(2,weight=1)
        f1.grid(row=0,column=2,sticky="NSEW")
        f1.columnconfigure(0,weight=1)
        Tkinter.Button(f1,text='Plot',command=self.replot).grid(row=0,column=0,sticky="NSEW")
        Tkinter.Button(f1,text='Clear',command=self.clearplot).grid(row=2,column=0,sticky="NSEW")
        Tkinter.Button(f1,text='Quit',command=self.quit).grid(row=3,column=0,sticky="NSEW")
        f1=Tkinter.LabelFrame(f,text='Flags')
        f1.columnconfigure(0,weight=1)
        f1.grid(row=0,column=3,sticky="NSEW")
        self.maskflag=Tkinter.Label(f1,text='No mask',anchor='w',justify='left')
        self.maskflag.grid(sticky='we')
        self.intdataflag=Tkinter.Label(f1,text='No 1D data',anchor='w',justify='left')
        self.intdataflag.grid(sticky='we')
        self.busyflag=Tkinter.Label(f1,text='Idle',anchor='w',justify='left')
        self.busyflag.grid(sticky='we')
        self.setflag('mask',False)
        self.setflag('intdata',False)
        self.setflag('busy',False)
        
        
        lf=Tkinter.LabelFrame(frame,text='Mask operations')
        lf.grid(columnspan=2,sticky="NEWS")
        lf.columnconfigure(6,weight=1)
        Tkinter.Button(lf,text='Create new',command=self.createmask).grid(row=0,column=0,sticky="NSEW")
        Tkinter.Button(lf,text='Clear',command=self.clearmask).grid(row=0,column=1,sticky="NSEW")
        Tkinter.Button(lf,text='Adjust',command=self.adjustmask).grid(row=0,column=2,sticky="NSEW")
        Tkinter.Button(lf,text='Save to',command=self.savemask).grid(row=0,column=3,sticky="NSEW")
        Tkinter.Button(lf,text='Load from',command=self.loadmask).grid(row=0,column=4,sticky="NSEW")
        Tkinter.Label(lf,text='Filename:').grid(row=0,column=5,sticky='NSEW')
        self.maskfilename=Tkinter.Entry(lf)
        self.maskfilename.grid(row=0,column=6,sticky='NSEW')
        self.maskfilename.insert(0,default_mask)

        lf=Tkinter.LabelFrame(frame,text='Operations on the current integrated dataset')
        lf.grid(columnspan=2,sticky="NEWS")
        lf.columnconfigure(4,weight=1)
        Tkinter.Button(lf,text='Fitting...',command=self.fit).grid(row=0,column=0,sticky="NSEW")
        Tkinter.Button(lf,text='Save to',command=self.saveto).grid(row=0,column=1,sticky="NSEW")
        Tkinter.Button(lf,text='Load from',command=self.loadfrom).grid(row=0,column=2,sticky="NSEW")
        Tkinter.Label(lf,text='Filename:').grid(row=0,column=3,sticky='NSEW')
        self.savefilename=Tkinter.Entry(lf)
        self.savefilename.grid(row=0,column=4,sticky='NSEW')
        self.savefilename.insert(0,'')

        lf=Tkinter.LabelFrame(frame,text='Log')
        lf.grid(columnspan=2,sticky="NEWS")
        lf.columnconfigure(0,weight=1)
        lf.rowconfigure(0,weight=1)
        frame.rowconfigure(lf.grid_info()['row'],weight=1)
        self.logtext=Tkinter.Text(lf,state='disabled',height=5)
        self.logtext.grid(row=0,column=0,columnspan=1,sticky="NSEW")
        self.logtext.tag_config("ERROR",foreground="white",background="red")
        self.logtext.tag_config("TIMESTAMP",foreground="black",background="white")
        self.logtext.tag_config("NORMAL",foreground="black",background="white")
        self.logtext.tag_config("WARNING",foreground="black",background="yellow")
        self.logtext.tag_config("INFO",foreground="green",background="white")
        sc=Tkinter.Scrollbar(lf,orient=Tkinter.VERTICAL)
        sc.grid(row=0,column=1,sticky="NSEW")
        self.logtext['yscrollcommand']=sc.set
        sc['command']=self.logtext.yview
        f=Tkinter.Frame(lf)
        f.grid(row=1,column=0,columnspan=2)
        Tkinter.Button(f,text="Save",command=self.savelog).grid(row=0,column=0,sticky="NSEW")
        Tkinter.Button(f,text="Clear",command=self.clearlog).grid(row=0,column=1,sticky="NSEW")
        self.currentdataset=None
        self.currentparam=None
        self.maskmatrix=None
    def setflag(self,flag,status=True):
        flags={'mask':(self.maskflag,{'background':None,'text':'No mask'},{'background':'green','text':'Mask loaded'}),
               'busy':(self.busyflag,{'background':'green','text':'Idle'},{'background':'orange','text':'Busy'}),
               'intdata':(self.intdataflag,{'background':None,'text':'No 1D data'},{'background':'green','text':'1D data present'})}
        if flag in flags.keys():
            for k in flags[flag][int(status)+1].keys():
                if flags[flag][int(status)+1][k] is None:
                    flags[flag][0][k]=self.master['background']
                else:
                    flags[flag][0][k]=flags[flag][int(status)+1][k]
        self.master.update()
    def clearmask(self):
        self.maskmatrix=None
        self.logger('Mask cleared.','NORMAL')
        self.setflag('mask',False)
    def createmask(self):
        fsn=self.getfsn()
        data,dataerr,param=B1python.read2dintfile(fsn,dirs=self.getinputdirs())
        if len(data)!=1 and len(dataerr)!=1 and len(param)!=1:
            self.logger("Could not find files for FSN %d"%fsn,priority="error")
            return
        self.maskmatrix=np.ones(data.shape,dtype=np.uint8)
        self.logger('Created an empty %d x %d mask'%data[0].shape)
        self.setflag('mask',True)
    def savemask(self):
        if (self.maskmatrix is not None) and (len(self.maskfilename.get().strip())>0):
            try:
                matfilename=self.maskfilename.get()
                matname=os.path.splitext(os.path.split(self.maskfilename.get())[-1])[0]
                scipy.io.savemat(matfilename,{matname:self.maskmatrix},appendmat=True)
            except IOError,string:
                self.logger('Could not save file (%s)'%string,'ERROR')
                return
            if not matfilename.lower().endswith('.mat'):
                matfilename='%s.mat'%matfilename
            self.logger('Saved mask in file %s with label %s.'%(matfilename,matname))
            
    def loadmask(self):
        foundindir=None
        for d in self.getinputdirs():
            try:
                mat=scipy.io.loadmat(os.path.join(d,self.maskfilename.get()))
                foundindir=d
                matname=[x for x in mat.keys() if not x.startswith('_')]
                if len(matname)<=0:
                    self.logger('Malformed mask file: %s. IGNORING.'%os.path.join(d,self.maskfilename.get()),'WARNING')
                    foundindir=None
                elif len(matname)>1:
                    self.logger('Multiple masks in file %s. IGNORING.'%os.path.join(d,self.maskfilename.get()),'WARNING')
                    foundindir=None
                else:
                    self.maskmatrix=mat[matname[0]].astype(np.uint8)
                    self.logger('%s loaded from file %s'%(matname[0],os.path.join(d,self.maskfilename.get())))
            except IOError:
                pass
            if foundindir is not None:
                break
        if not foundindir:
            #if this point is reached, mask could not be loaded. Inform the user.
            self.logger('Mask file %s could not be found!'%self.maskfilename.get(),'ERROR')
        else:
            self.setflag('mask',True)
    def adjustmask(self):
        fsn=self.getfsn()
        data,dataerr,param=B1python.read2dintfile(fsn,dirs=self.getinputdirs())
        if len(data)!=1 and len(dataerr)!=1 and len(param)!=1:
            self.logger("Could not find files for FSN %d"%fsn,priority="error")
            return
        if self.maskmatrix is None:
            self.maskmatrix=np.zeros(data[0].shape,dtype=np.uint8)
        try:
            self.maskmatrix=B1python.makemask(self.maskmatrix,data[0])
        except:
            self.logger("Mask editing was interrupted by user!",'WARNING')
    def fit(self):
        if self.currentdataset is None or self.currentparam is None:
            self.logger('Error: No integrated dataset present! Integrate or load something first!',"ERROR")
            return
        B1python.basicfittinggui(B1python.SASDict(**(self.currentdataset)))
    def loadfrom(self):
        data=B1python.readintfile(self.savefilename.get())
        if len(data)==0:
            self.logger('Error: Cannot load intensity from file %s' % self.savefilename.get(),'ERROR')
        self.currentdataset=data
        self.currentparam=self.savefilename.get()
        self.setflag('intdata',True)
    def saveto(self):
        if self.currentdataset is None or self.currentparam is None:
            self.logger('Error: No integrated dataset present! Integrate or load something first!',"ERROR")
            return
        try:
            B1python.write1dsasdict(self.currentdataset,self.savefilename.get())
            self.logger('Successfully saved file %s' % self.savefilename.get(),'Info')
        except:
            self.logger('Error saving file %s'%self.savefilename.get(),'Error')
    def toggleqminqmax(self):
        if self.qminmanual.get()==0:
            self.qmin['state']='disabled'
        else:
            self.qmin['state']='normal'
        if self.qmaxmanual.get()==0:
            self.qmax['state']='disabled'
        else:
            self.qmax['state']='normal'
        if self.Nqmanual.get()==0:
            self.Nq['state']='disabled'
        else:
            self.Nq['state']='normal'
        if self.phi0manual.get()==0:
            self.phi0['state']='disabled'
        else:
            self.phi0['state']='normal'
        if self.dphimanual.get()==0:
            self.dphi['state']='disabled'
        else:
            self.dphi['state']='normal'
    def logger(self,text,priority=None):
        self.logtext['state']='normal'
        self.logtext.insert(Tkinter.END,"%s: "%time.ctime(),("TIMESTAMP",))
        if priority is None:
            priority="NORMAL"
        self.logtext.insert(Tkinter.END,text,(priority.upper(),))
        self.logtext.insert(Tkinter.END,"\n",("NORMAL",))
        self.logtext.see(Tkinter.END)
        self.logtext['state']='disabled'
        self.master.update()
    def savelog(self):
        pass
    def clearlog(self):
        self.logtext['state']='normal'
        self.logtext.delete("0.0",Tkinter.END)
        self.logtext['state']='disabled'
    def replot(self):
        self.setflag('busy',True)
        fsn=self.getfsn()
        data,dataerr,param=B1python.read2dintfile(fsn,dirs=self.getinputdirs())
        self.setflag('busy',False)
        if len(data)!=1 and len(dataerr)!=1 and len(param)!=1:
            self.logger("Could not find files for FSN %d"%fsn,priority="error")
            return
        else:
            self.logger('Files for FSN %d loaded successfully'%fsn,priority='INFO')
        mask=self.maskmatrix
        self.setflag('busy',True)
        if self.getplottype()=='2D':
            try:
                self.figure.show()
            except:
                self.figure=pylab.figure(self.figurenum)
            pylab.figure(self.figurenum)
            self.figure.clf()
            B1python.plot2dmatrix(data[0],header=param[0],showqscale=True,mask=self.maskmatrix)
            pylab.xlabel(u'q (1/%c)'%197)
            pylab.ylabel(u'q (1/%c)'%197)
            pylab.colorbar()
        elif self.getplottype()=='1D':
            if mask is None:
                mask=np.ones(data[0].shape,dtype=np.uint8)
            if not (self.qminmanual.get() and self.qmaxmanual.get() and self.Nqmanual.get()):
                qmin,qmax,Nq=B1python.qrangefrommask(mask, param[0]['EnergyCalibrated'],param[0]['Dist'],param[0]['PixelSize'],param[0]['BeamPosX'],param[0]['BeamPosY'])
                self.logger('Auto-determined integration bounds: %.8f < q < %.8f, optimal number of bins is %d' %(qmin,qmax,Nq))
            if self.qminmanual.get():
                qmin=float(self.qmin.get())
            else:
                self.qmin['state']='normal'
                self.qmin.delete(0,Tkinter.END)
                self.qmin.insert(0,'%.8f' % qmin)
                self.qmin['state']='disabled'
            if self.qmaxmanual.get():
                qmax=float(self.qmax.get())
            else:
                self.qmax['state']='normal'
                self.qmax.delete(0,Tkinter.END)
                self.qmax.insert(0,'%.8f' % qmax)
                self.qmax['state']='disabled'
            if self.Nqmanual.get():
                Nq=long(self.Nq.get())
            else:
                self.Nq['state']='normal'
                self.Nq.delete(0,Tkinter.END)
                self.Nq.insert(0,'%.8f' % Nq)
                self.Nq['state']='disabled'
            phi0=None
            dphi=None
            if self.phi0manual.get():
                phi0=float(self.phi0.get())*np.pi/180.0
            if self.dphimanual.get():
                dphi=float(self.dphi.get())*np.pi/180.0
            qrange=np.linspace(qmin,qmax,Nq)
            self.logger('Integration starting...',"INFO")
            t0=time.time()
            q,I,E,A,maskout=B1python.radintC(data[0],dataerr[0],param[0]['EnergyCalibrated'],param[0]['Dist'],param[0]['PixelSize'],param[0]['BeamPosX'],param[0]['BeamPosY'],1-mask,qrange,phi0=phi0,dphi=dphi,returnmask=True,returnavgq=True)
            maskout=1-maskout
            self.currentdataset=B1python.SASDict(q,I,E,Area=A)
            self.currentdataset.sanitize()
            self.currentparam=param[0]
            self.setflag('intdata',True)
            self.logger('Integration finished in %.2f seconds'%(time.time()-t0),"INFO")
            try:
                self.figure.show()
            except:
                self.figure=pylab.figure(self.figurenum)
            pylab.figure(self.figurenum)
            B1python.plotintegrated(data[0],q,I,E,A,qrange,maskout,param[0],mode='radial')
        elif self.setplottype()=='Azimuthal':
            self.logger('Azimuthal integration not yet supported!','WARNING')
        self.setflag('busy',False)
        pylab.draw()
        pylab.gcf().show()
        del data
        del dataerr
        del param
    def getinputdirs(self):
        if self.inputdirlist is not None and self.inputdirlist[0]==self.inputdirs.get():
            return self.inputdirlist[1]
        #otherwise 
        self.logger('Parsing input directories, please wait...','INFO')
        dirs=[ x.strip() for x in self.inputdirs.get().split(';')]
        result=set()
        for d in dirs:
            for i in os.walk(d):
                result.add(i[0])
        result=list(result)
        result.sort()
        self.inputdirlist=[self.inputdirs.get(),result]
        self.logger('%d subdirectories found.'%len(result),'INFO')
        return self.inputdirlist[1]
        
    def clearplot(self):
        print "Clear plot"
        pylab.clf()
        pylab.draw()
        pylab.gcf().show()
    def newfsn(self,event=None):
        self.replot()
    def getfsn(self):
        return int(self.fsn.get())
    def getplottype(self):
        if self.rbvar.get()==1:
            return "2D"
        elif self.rbvar.get()==2:
            return "1D"
        elif self.rbvar.get()==3:
            return "Azimuthal"
        else:
            raise ValueError,"Invalid plot type: %d" % self.rbvar.get()
    def newplottype(self):
        pylab.clf()
        pylab.draw()
        pylab.gcf().show()
        self.replot()
    def quit(self):
        pylab.close('all')
        try:
            f=open(configfilename,'wt');
            f.write('Default inputdirs= %s\n'%self.inputdirs.get())
            f.write('Default outputdir= %s\n'%self.outputdir.get())
            f.write('Default mask= %s\n'%self.maskfilename.get())
            f.close()
        except IOError:
            pass
        self.master.destroy()



root=Tkinter.Tk()
a=MainWindow(root)
root.mainloop()
