import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets
import matplotlib.nxutils
import scipy.io
import utils
import fitting
import utils2d
import time
import B1io

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

def plotintegrated(A,q,intensity,error=None,area=None,qtheor=None,mask=None,param=None,xscale='log',yscale='log',mode='radial'):
    """Plot the results of a radial/azimuthal integration
    
    Inputs:
        A: 2D image
        q: list of scattering variables (4*pi*sin(theta)/lambda) or azimuth
            angles. Should be the integrated (averaged) one, ie. returned by
            radintC or azimintqC.
        intensity: list of scattering intensity
        error: error of the scattering intensity. None if unapplicable
        area: effective area for bins. None if unapplicable.
        qtheor: theoretical values for q (or azimuth angle). None if
            unapplicable.
        mask: mask matrix. 1 means unmasked, 0 masked. None if unapplicable
        param: parameter structure. Supplied opaquely to plot2dmatrix(). None
            if unapplicable
        xscale: 'li[near]' or 'lo[garithmic]'. Characters in square brackets
            are optional. This controls the plotting of all x axes. Default is
            'log'.
        yscale: 'li[near]' or 'lo[garithmic]', see xscale. This only controls
            the scaling of the y axis in the I vs. q plot. Area vs. q is
            unaffected. Default value: 'log'.
        mode: mode of the integration. 'r[adial]' or 'a[zimuthal]'. Default is
            'radial'
    """
    plt.clf()
    if (area is not None) or (qtheor is not None):
        Nsubplotrows=2
    else:
        Nsubplotrows=1
    plt.subplot(Nsubplotrows,2,1)
    plot2dmatrix(A,mask=mask,header=param,showqscale=True)
    plt.subplot(Nsubplotrows,2,2)
    
    if mode.upper()=='AZIMUTHAL'[:max(1,len(mode))]:
        xlabel=u'azimuth angle (rad)'
    elif (mode.upper()=='RADIAL'[:max(1,len(mode))]) or (mode.upper()=='SECTOR'[:max(1,len(mode))]) (mode.upper()=='SLICE'[:max(1,len(mode))]):
        xlabel=u'q (1/%c)' % 197
    else:
        raise ValueError, "Invalid integration mode: %s" %mode
    
    if error is not None:
        plt.errorbar(q,intensity,error)
    else:
        plt.plot(q,intensity)
        
    if xscale.upper()=='LOGARITHMIC'[:max(2,len(xscale))]:
        plt.xscale('log')
    elif xscale.upper()=='LINEAR'[:max(2,len(xscale))]:
        plt.xscale('linear')
    else:
        raise ValueError, 'invalid scaling mode %s' % xscale

    if yscale.upper()=='LOGARITHMIC'[:max(2,len(yscale))]:
        plt.yscale('log')
    elif yscale.upper()=='LINEAR'[:max(2,len(yscale))]:
        plt.yscale('linear')
    else:
        raise ValueError, 'invalid scaling mode %s' % yscale
        
    plt.xlabel(xlabel)
    plt.ylabel(u'Intensity')
    plt.title('Integrated intensity')
    
    if area is not None:        
        plt.subplot(Nsubplotrows,2,3)
        plt.plot(q,area)
        plt.xlabel(xlabel)
        plt.ylabel(u'Effective area')
        if xscale.upper()=='LOGARITHMIC'[:max(2,len(xscale))]:
            plt.xscale('log')
        elif xscale.upper()=='LINEAR'[:max(2,len(xscale))]:
            plt.xscale('linear')
        else:
            raise ValueError, 'invalid scaling mode %s' % xscale
        plt.title('Effective area')
    if qtheor is not None:
        plt.subplot(Nsubplotrows,2,4)
        plt.plot(q,q/qtheor,'.')
        plt.xlabel(xlabel)
        plt.ylabel(u'Effective q / desired q (==1 if correct)')
        if xscale.upper()=='LOGARITHMIC'[:max(2,len(xscale))]:
            plt.xscale('log')
        elif xscale.upper()=='LINEAR'[:max(2,len(xscale))]:
            plt.xscale('linear')
        else:
            raise ValueError, 'invalid scaling mode %s' % xscale
        plt.title('q_eff/q_theor')
    plt.gcf().show()

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
            applies as to marker. Default value is 1.
        gui [optional]: display graphical user interface to show/hide plotted
            lines independently. Default value is False (no gui)
    Outputs:
        list of names of found samples, to be fed eg. to legend().
    
    """
    legends=[]
    if not (isinstance(energies,list) or isinstance(energies,tuple)):
        energies=[energies];
    colors=['blue','green','red','black','magenta'];
    if not (isinstance(samplename,list) or isinstance(samplename,tuple)):
        samplename=[samplename]
    if not (isinstance(marker,list) or isinstance(marker,tuple)):
        marker=[marker]
    if not (isinstance(mult,list) or isinstance(mult,tuple)):
        mult=[mult]
    if len(marker)==1:
        marker=marker*len(samplename)
    if len(mult)==1:
        mult=mult*len(samplename)
    if (len(marker)<len(samplename)) or (len(mult) <len(samplename)):
        raise ValueError
    if gui==True:
        fig=plt.figure()
        buttonax=fig.add_axes((0.65,0.1,0.3,0.05))
        guiax=fig.add_axes((0.65,0.15,0.3,0.75))
        ax=fig.add_axes((0.1,0.1,0.5,0.8))
        btn=matplotlib.widgets.Button(buttonax,'Close GUI')
        def fun(event):
            fig=plt.gcf()
            fig.delaxes(fig.axes[0])
            fig.delaxes(fig.axes[0])
            fig.axes[0].set_position((0.1,0.1,0.8,0.85))
        btn.on_clicked(fun)
        guiax.set_title('Visibility selector')
        texts=[]
        handles=[]
    else:
        fig=plt.gcf()
        ax=plt.gca()
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
                        #h=ax.errorbar(data[k]['q'],data[k]['Intensity']*mult[s],
                        #              data[k]['Error']*mult[s],
                        #              marker=marker[s],color=colors[e])
                        #ax.set_xscale('log')
                        #ax.set_yscale('log')
                        legends.append(param[k]['Title'])
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
            plt.draw()
    return legends
def plot2dmatrix(A,maxval=None,mask=None,header=None,qs=[],showqscale=True,contour=None,pmin=0,pmax=1,blacknegative=False,crosshair=True,zscaling='log',axes=None):
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
        blacknegative: True if you want to black out nonpositive pixels
        crosshair: True if you want to draw a beam-center testing cross-hair
        zscaling: 'log' or 'linear' (color scale model)
    """
    if axes is None:
        axes=plt.gca()
    tmp=A.copy(); # this is needed as Python uses the pass-by-object method,
                  # so A is the SAME as the version of the caller. tmp=A would
                  # render tmp the SAME (physically) as A. If we only would
                  # call np.log(tmp), it won't be an error, as np.log()
                  # does not tamper with the content of its argument, but
                  # returns a new matrix. However, when we do the magic with
                  # maxval, it would be a problem, as elements of the original
                  # matrix were modified.
    if maxval is not None:
        tmp[tmp>maxval]=max(tmp[tmp<=maxval])
    nonpos=(tmp<=0)
    tmp[nonpos]=tmp[tmp>0].min()
    if zscaling.upper()=='LOG':
        tmp=np.log(tmp);
    tmp[np.isnan(tmp)]=tmp[-np.isnan(tmp)].min();
    if (header is not None) and (showqscale):
        # x: row direction (vertical on plot)
        xmin=0-(header['BeamPosX']-1)*header['PixelSize']
        xmax=(tmp.shape[0]-(header['BeamPosX']-1))*header['PixelSize']
        ymin=0-(header['BeamPosY']-1)*header['PixelSize']
        ymax=(tmp.shape[1]-(header['BeamPosY']-1))*header['PixelSize']
        qxmin=4*np.pi*np.sin(0.5*np.arctan(xmin/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qxmax=4*np.pi*np.sin(0.5*np.arctan(xmax/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qymin=4*np.pi*np.sin(0.5*np.arctan(ymin/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qymax=4*np.pi*np.sin(0.5*np.arctan(ymax/header['Dist']))*header['EnergyCalibrated']/float(HC)
        extent=[qymin,qymax,qxmax,qxmin]
        bcrow=0
        bccol=0
    else:
        extent=None
        if header is not None:
            bcrow=header['BeamPosX']-1
            bccol=header['BeamPosY']-1
    if extent is None:
        extent1=[1,tmp.shape[0],1,tmp.shape[1]]
    else:
        extent1=extent;
    if contour is None:
        axes.imshow(tmp,extent=extent,interpolation='nearest',vmin=tmp.min()+pmin*(tmp.max()-tmp.min()),vmax=tmp.min()+pmax*(tmp.max()-tmp.min()));
    else:
        X,Y=np.meshgrid(np.linspace(extent1[2],extent1[3],tmp.shape[1]),
                           np.linspace(extent1[0],extent1[1],tmp.shape[0]))
        axes.contour(X,Y,tmp,contour)
    if blacknegative:
        black=np.zeros((A.shape[0],A.shape[1],4))
        black[:,:,3][nonpos]=1
        axes.imshow(black,extent=extent,interpolation='nearest')
    if mask is not None:
        white=np.ones((mask.shape[0],mask.shape[1],4))
        white[:,:,3]=np.array(1-mask).astype('float')*0.7
        axes.imshow(white,extent=extent,interpolation='nearest')
    for q in qs:
        a=axes.axis()
        axes.plot(q*np.cos(np.linspace(0,2*np.pi,2000)),
                   q*np.sin(np.linspace(0,2*np.pi,2000)),
                   color='white',linewidth=3)
        axes.axis(a)
    if header is not None:
        if 'FSN' in header.keys() and 'Title' in header.keys():
            axes.set_title("#%d: %s" % (header['FSN'], header['Title']))
    if crosshair and header is not None:
        a=axes.axis()
        axes.plot([extent1[0],extent1[1]],[bcrow,bcrow],'-',color='white')
        axes.plot([bccol,bccol],[extent1[2],extent1[3]],'-',color='white')
        axes.axis(a)
        
def makemask(mask,A,savefile=None):
    """Make mask matrix.
    
    Inputs:
        mask: preliminary mask matrix. Give None to create a fresh one
        A: background image. The size of mask and this should be equal.
        savefile [optional]: a file name to save the mask to.
    Output:
        the mask matrix. The masked (shaded in the GUI) pixels will be
        False, non-masked True
    """
    def clickevent(event):
        fig=plt.gcf()
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
                Col,Row=np.meshgrid(np.arange(fig.mydata['mask'].shape[1]),
                                       np.arange(fig.mydata['mask'].shape[0]))
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
                R=np.sqrt((x0-event.xdata)**2+(y0-event.ydata)**2)
                Col,Row=np.meshgrid(np.arange(fig.mydata['mask'].shape[1])-x0,
                                       np.arange(fig.mydata['mask'].shape[0])-y0)
                fig.mydata['selection']=np.sqrt(Col**2+Row**2)<=R
                fig.mydata['ax'].set_title('Mask/unmask region with the appropriate button!')
                a=fig.mydata['ax'].axis()
                fig.mydata['ax'].plot(x0+R*np.cos(np.linspace(0,2*np.pi,2000)),
                                      y0+R*np.sin(np.linspace(0,2*np.pi,2000)),
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
                    Col,Row=np.meshgrid(np.arange(fig.mydata['mask'].shape[1]),
                                           np.arange(fig.mydata['mask'].shape[0]))
                    Points=np.zeros((Col.size,2))
                    Points[:,0]=Col.flatten()
                    Points[:,1]=Row.flatten()
                    fig.mydata['selection']=np.zeros(Col.shape).astype('bool')
                    ptsin=matplotlib.nxutils.points_inside_poly(Points,fig.mydata['selectdata'])
                    fig.mydata['selection'][ptsin.reshape(Col.shape)]=True
                    fig.mydata['selectdata']=[]
                    fig.mydata['mode']=None
            elif fig.mydata['mode']=='PHUNT':
                fig.mydata['mask'][np.floor(event.ydata+.5),np.floor(event.xdata+.5)]=not(fig.mydata['mask'][np.floor(event.ydata+.5),np.floor(event.xdata+.5)])
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
            plt.gcf().toexit=True
    if mask is None:
        mask=np.ones(A.shape)
    if A.shape!=mask.shape:
        print 'The shapes of A and mask should be equal.'
        return None
    fig=plt.gcf();
    fig.clf()
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
    fig.canvas.mpl_connect('button_press_event',clickevent)
    fig.toexit=False
    fig.show()
    firstdraw=1;
    while fig.toexit==False:
        if fig.mydata['redrawneeded']:
            if not firstdraw:
                ax=fig.mydata['ax'].axis();
            fig.mydata['redrawneeded']=False
            fig.mydata['ax'].cla()
            plt.axes(fig.mydata['ax'])
            plot2dmatrix(A,mask=fig.mydata['mask'])
            fig.mydata['ax'].set_title('')
            if not firstdraw:
                fig.mydata['ax'].axis(ax);
            firstdraw=0;
        plt.draw()
        fig.waitforbuttonpress()
    #ax.imshow(maskplot)
    #plt.show()
    mask=fig.mydata['mask']
    #plt.close(fig)
    fig.clf()
#    plt.title('Mask Done')
    if savefile is not None:
        print 'Saving file'
        scipy.io.savemat(savefile,{'mask':mask})
    return mask
    

def basicfittinggui_old(data,title='',blocking=False):
    """Graphical user interface to carry out basic (Guinier, Porod) fitting
    to 1D scattering data.
    
    Inputs:
        data: 1D dataset
        title: title to display
        blocking: False if the function should return just after drawing the
            fitting gui. True if it should wait for closing the figure window.
    Output:
        If blocking was False then none, this leaves a figure open for further
            user interactions.
        If blocking was True then after the window was destroyed, a list of
            the fits and their parameters are returned.
    """
    listoffits=[]
    
    leftborder=0.05
    topborder=0.9
    bottomborder=0.1
    leftbox_end=0.3
    data=utils.flatten1dsasdict(data)
    fig=plt.figure()
    plt.clf()
    plots=['Guinier','Guinier thickness','Guinier cross-section','Porod','lin-lin','lin-log','log-lin','log-log']
    buttons=['Guinier','Guinier thickness','Guinier cross-section','Porod','Power law','Power law with c.background','Power law with l.background']
    fitfuns=[fitting.guinierfit,fitting.guinierthicknessfit,fitting.guiniercrosssectionfit,fitting.porodfit,fitting.powerfit,fitting.powerfitwithbackground,fitting.powerfitwithlinearbackground]
    for i in range(len(buttons)):
        ax=plt.axes((leftborder,topborder-(i+1)*(0.8)/(len(buttons)+len(plots)),leftbox_end,0.7/(len(buttons)+len(plots))))
        but=matplotlib.widgets.Button(ax,buttons[i])
        def onclick(A=None,B=None,data=data,type=buttons[i],fitfun=fitfuns[i]):
            a=plt.axis()
            plottype=plt.gcf().plottype
            plt.figure()
            if plottype=='Guinier':
                xt=data['q']**2
                yt=np.log(data['Intensity'])
            elif plottype=='Guinier thickness':
                xt=data['q']**2
                yt=np.log(data['Intensity'])*xt
            elif plottype=='Guinier cross-section':
                xt=data['q']**2
                yt=np.log(data['Intensity'])*data['q']
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
            listoffits.append({'type':type,'res':res,'time':time.time(),'qmin':qmin,'qmax':qmax})
#            if len(res)==4:
#                plt.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg' % (type,res[0],res[2],res[1],res[3]))
#            elif len(res)==6:
#                plt.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg;\n %lg +/- %lg' % (type,res[0],res[3],res[1],res[4],res[2],res[5]))
#            elif len(res)==8:
#                plt.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg;\n %lg +/- %lg; %lg +/- %lg' % (type,res[0],res[4],res[1],res[5],res[2],res[6],res[3],res[7]))
            plt.gcf().show()
        but.on_clicked(onclick)
    ax=plt.axes((leftborder,topborder-(len(buttons)+len(plots))*(0.8)/(len(buttons)+len(plots)),leftbox_end,0.7/(len(buttons)+len(plots))*len(plots) ))
    plt.title('Plot types')
    rb=matplotlib.widgets.RadioButtons(ax,plots,active=7)
    plt.gcf().blocking=blocking
    if plt.gcf().blocking:
        ax=plt.axes((leftborder,0.03,leftbox_end,bottomborder-0.03))
        b=matplotlib.widgets.Button(ax,"Done")
        fig=plt.gcf()
        fig.fittingdone=False
        def onclick1(A=None,B=None,fig=fig):
            fig.fittingdone=True
            plt.gcf().blocking=False
     #       print "blocking should now end"
        b.on_clicked(onclick1)
    plt.axes((0.4,bottomborder,0.5,0.8))
    def onselectplottype(plottype,q=data['q'],I=data['Intensity'],title=title):
        plt.cla()
        plt.gcf().plottype==plottype
        if plottype=='Guinier':
            x=q**2
            y=np.log(I)
            plt.plot(x,y,'.')
            plt.xlabel('q^2')
            plt.ylabel('ln I')
        elif plottype=='Guinier thickness':
            x=q**2
            y=np.log(I)*q**2
            plt.plot(x,y,'.')
            plt.xlabel('q^2')
            plt.ylabel('ln I*q^2')
        elif plottype=='Guinier cross-section':
            x=q**2
            y=np.log(I)*q
            plt.plot(x,y,'.')
            plt.xlabel('q^2')
            plt.ylabel('ln I*q')            
        elif plottype=='Porod':
            x=q**4
            y=I*q**4
            plt.plot(x,y,'.')
            plt.xlabel('q^4')
            plt.ylabel('I*q^4')
        elif plottype=='lin-lin':
            plt.plot(q,I,'.')
            plt.xlabel('q')
            plt.ylabel('I')
        elif plottype=='lin-log':
            plt.semilogx(q,I,'.')
            plt.xlabel('q')
            plt.ylabel('I')
        elif plottype=='log-lin':
            plt.semilogy(q,I,'.')
            plt.xlabel('q')
            plt.ylabel('I')
        elif plottype=='log-log':
            plt.loglog(q,I,'.')
            plt.xlabel('q')
            plt.ylabel('I')
        plt.title(title)
        plt.gcf().plottype=plottype
        plt.gcf().show()
    rb.on_clicked(onselectplottype)
    plt.title(title)
    plt.loglog(data['q'],data['Intensity'],'.')
    plt.gcf().plottype='log-log'
    plt.gcf().show()
    plt.draw()
    fig=plt.gcf()
    while blocking:
        fig.waitforbuttonpress()
#        print "buttonpress"
        if fig.fittingdone:
            blocking=False
            #print "exiting"
    #print "returning"
    return listoffits
        
def basicfittinggui(data,title='',blocking=False):
    """Graphical user interface to carry out basic (Guinier, Porod, etc.)
    fitting to 1D scattering data.
    
    Inputs:
        data: 1D dataset
        title: title to display
        blocking: False if the function should return just after drawing the
            fitting gui. True if it should wait for closing the figure window.
    Output:
        If blocking was False then none, this leaves a figure open for further
            user interactions.
        If blocking was True then after the window was destroyed, a list of
            the fits and their parameters are returned.
    """
    
    if not isinstance(data,utils.SASDict):
        return basicfittinggui_old(data,title,blocking)
    else:
        return data.basicfittinggui(title,blocking)


def testsmoothing(x,y,smoothing=1e-5,slidermin=None,slidermax=None,
                  xscaling='linear',yscaling='log',smoothingmode='spline',
                  callback=None,returnsmoothed=False,tkgui=False):
    """Tool to test and adjust smoothing of a curve.
    
    Inputs:
        x: abscissa
        y: ordinate
        smoothing: initial value of the smoothing parameter
        slidermin: minimum value of the slider. If None, defaults to 10^(log10(smoothing)-2)
        slidermax: maximum value of the slider. If None, defaults to 10^(log10(smoothing)+2)
        xscaling: scaling ('log' or 'linear') of the abscissa
        yscaling: scaling ('log' or 'linear') of the ordinate
        smoothingmode: mode of smoothing (see fitting.smoothcurve() for valid
            entries)
        callback: either None (do nothing) or a callable, accepting three
            arguments (the smoothing parameter, the smoothed curve and the axes)
            to do something, even plot.
        returnsmoothed: if the smoothed curve is to be returned [False by default]            
        tkgui [default=False]: if a Tk gui is desired instead of the matplotlib
            style.
            
    Outputs: s, [smoothedy]
        s: the final smoothing parameter
        smoothedy: the smoothed curve (if returnsmoothed was True)
        
    Notes:
        a matplotlib figure will be presented with a slider to allow the user to
        change the smoothing parameter.
    """
    #create the slider above
    fig=plt.gcf()
    fig.clf()
    if slidermin is None:
        slidermin=np.power(10,np.log10(smoothing)-2)
    if slidermax is None:
        slidermax=np.power(10,np.log10(smoothing)+2)

    if tkgui:
        import Tkinter as Tk
        root=Tk.Tk()
    else:
        root=None
        
    def butoff(a=None,fig=fig,tkroot=root):
        """This is going to be called if the OK button is pressed"""
        fig.smoothingdone=True
        fig.cancel=False
        if tkroot is not None:
            tkroot.destroy()
            tkroot.quit()
    def butcancel(a=None,fig=fig,tkroot=root):
        """This is going to be called if the OK button is pressed"""
        fig.smoothingdone=True
        fig.cancel=True
        if tkroot is not None:
            tkroot.destroy()
            tkroot.quit()

    if tkgui:
        root.wm_title("Adjust smoothing parameters")
        root.columnconfigure(1,weight=1)
        Tk.Label(root,text='Smoothing:',justify=Tk.LEFT,anchor=Tk.W).grid(sticky='NSEW')
        sl=Tk.Scale(root,orient='horizontal',length=300)
        sl.grid(row=0,column=1,sticky='NSEW')
        sl['to']=np.log10(slidermax)
        sl['from']=np.log10(slidermin)
        sl['resolution']=(sl['to']-sl['from'])/300.;
        sl.set(np.log10(smoothing))
        frm=Tk.Frame(root)
        frm.columnconfigure(0,weight=1)
        frm.columnconfigure(1,weight=1)
        frm.grid(row=1,columnspan=2,sticky='NSEW')
        b=Tk.Button(frm,text='OK',command=butoff)
        b.grid(row=0,column=0)
        b=Tk.Button(frm,text='Cancel',command=butcancel)
        b.grid(row=0,column=1)
        axes=fig.add_axes((0.1,0.1,0.8,0.8))
        
    else:
        ax=fig.add_axes((0.3,0.85,0.6,0.05));
        sl=matplotlib.widgets.Slider(ax,'',np.log10(slidermin),np.log10(slidermax),np.log10(smoothing));
    
        ax=fig.add_axes((0.1,0.85,0.1,0.05));
        fig.smoothingdone=False
        but=matplotlib.widgets.Button(ax,'Ok')
        but.on_clicked(butoff)
    
        ax=fig.add_axes((0.2,0.85,0.1,0.05));
        but=matplotlib.widgets.Button(ax,'Cancel')
        but.on_clicked(butcancel)
        axes=fig.add_axes((0.1,0.1,0.8,0.7))

    
    def fun(a,fig=fig,axes=axes,xscale=xscaling,yscale=yscaling,x=x,y=y,sl=sl,smoothmode=smoothingmode,callback=callback,thisisthefirstplot=False,tkgui=tkgui):
        if tkgui:
            if a==sl.lastvalue:
                return
            else:
                sl.lastvalue=a
        ax=axes.axis()
        if thisisthefirstplot:
            axes.cla()
            axes.plot(x,y,'b.')
        while len(axes.lines)>1:
            del axes.lines[-1]
        if tkgui:
            val=sl.get()
        else:
            val=sl.val
        fig.smoothing=pow(10.0,val);
        fig.ysmoothed=fitting.smoothcurve(x,y,fig.smoothing,mode=smoothmode)
        axes.plot(x,fig.ysmoothed,'g-',linewidth=2)
        if not thisisthefirstplot:
            axes.axis(ax)
        else:
            axes.set_xscale(xscale)
            axes.set_yscale(yscale)
        if callback is not None:
            callback(fig.smoothing,fig.ysmoothed,axes)
        plt.draw()
        #fig.show()
    if tkgui:
        sl.lastvalue=None
        sl['command']=fun
    else:
        sl.on_changed(fun)
    fun(1e-5,thisisthefirstplot=True)
    if tkgui:
        root.mainloop()
    else:
        while not fig.smoothingdone:
            plt.waitforbuttonpress()
    print "RETURNED FROM MAIN LOOP"
    if fig.cancel:
        raise RuntimeError('Smoothing was cancelled')
    if returnsmoothed:
        return fig.smoothing,fig.ysmoothed
    else:
        return fig.smoothing

def testorigin(data,orig,mask=None,dmin=None,dmax=None):
    """Shows several test plots by which the validity of the determined origin
    can  be tested.
    
    Inputs:
        data: the 2d scattering image
        orig: the origin [row,column], starting from 1.
        mask: the mask matrix. Nonzero means nonmasked
    """
    print "Creating origin testing images, please wait..."
    plt.clf()
    data=data.astype(np.double)
    if mask is None:
        mask=np.ones(data.shape,dtype=np.uint8)
    print "    plotting matrix with cross-hair..."
    plt.subplot(2,2,1)
    plot2dmatrix(data,mask=mask)
    plt.plot([0,data.shape[1]],[orig[0]-1,orig[0]-1],color='white')
    plt.plot([orig[1]-1,orig[1]-1],[0,data.shape[0]],color='white')
    plt.gca().axis('tight')
    print "    calculating and plotting slices..."
    plt.subplot(2,2,2)
    c1,nc1=utils2d.imageintC(data,orig,1-mask,35,20)
    c2,nc2=utils2d.imageintC(data,orig,1-mask,35+90,20)
    c3,nc3=utils2d.imageintC(data,orig,1-mask,35+180,20)
    c4,nc4=utils2d.imageintC(data,orig,1-mask,35+270,20)
    plt.semilogy(c1,marker='.',color='blue',markersize=3)
    plt.semilogy(c3,marker='o',color='blue',markersize=6)
    plt.semilogy(c2,marker='.',color='red',markersize=3)
    plt.semilogy(c4,marker='o',color='red',markersize=6)
    print "    calculating and plotting polar matrices..."
    plt.subplot(2,2,3)
    maxr=max([len(c1),len(c2),len(c3),len(c4)])
    pdata=utils2d.polartransform(data.astype('double'),np.arange(0,maxr,dtype=np.double),np.linspace(0,4*np.pi,600),orig[0],orig[1])
    pmask=utils2d.polartransform(mask.astype('double'),np.arange(0,maxr,dtype=np.double),np.linspace(0,4*np.pi,600),orig[0],orig[1])
    plot2dmatrix(pdata,mask=pmask)
    plt.axis('scaled')
    print "    calculating and plotting azimuthal curves..."
    plt.subplot(2,2,4)
    if dmin is None:
        dmin=maxr/4.
    if dmax is None:
        dmax=maxr/2.
    
    t,I,A=utils2d.azimintpixC(data,None,orig,(1-mask).astype(np.uint8),dmin,dmin,dmax)
    plt.plot(t,I,'b-')
    plt.ylabel('Azimuthal intensity (blue)\n(nonperiodic for 2pi)')
    plt.twinx()
    plt.plot(t,A,'g-')
    plt.ylabel('Effective area (green)\n(should be definitely flat)')
    plt.xlabel('Azimuth angle (rad)')
    plt.gcf().show()
    print "... images ready!"
def assesstransmission(fsns,titleofsample,mode='Gabriel',dirs=[]):
    """Plot transmission, beam center and Doris current vs. FSNs of the given
    sample.
    
    Inputs:
        fsns: range of file sequence numbers
        titleofsample: the title of the sample which should be investigated
        mode: 'Gabriel' if the measurements were made with the gas-detector, 
            and 'Pilatus300k' if that detector was used.            
    """
    hborder=0.08
    vborder=0.08
    hspacing=0.02
    vspacing=0.08
    axiswidth=0.6
    axisheight=(1-vborder*2-vspacing*3)/4.
    
    
    if not( isinstance(fsns,list) or isinstance(fsns,tuple)):
        fsns=[fsns]
    if mode.upper().startswith('GABRIEL'):
        header1=B1io.readheader('ORG',fsns,'.DAT',dirs=dirs)
    elif mode.upper().startswith('PILATUS'):
        header1=B1io.readheader('org_',fsns,'.header',dirs=dirs)
    else:
        raise ValueError("invalid mode argument. Possible values: 'Gabriel' or 'Pilatus'")
    params1=B1io.readlogfile(fsns,dirs=dirs)
    header=[]
    for h in header1:
        if h['Title']==titleofsample:
            header.append(h.copy())
    params=[]
    for h in params1:
        if h['Title']==titleofsample:
            params.append(h.copy())
    energies=utils.unique([h['Energy'] for h in header],(lambda a,b:abs(a-b)<2))

    axisTransm=plt.gcf().add_axes([hborder,1-vborder-axisheight,axiswidth,axisheight])
    axisBcX=plt.gcf().add_axes([hborder,1-vborder-2*axisheight-vspacing,axiswidth,axisheight])
    axisBcY=plt.gcf().add_axes([hborder,1-vborder-3*axisheight-2*vspacing,axiswidth,axisheight])
    axisDORIS=plt.gcf().add_axes([hborder,1-vborder-4*axisheight-3*vspacing,axiswidth,axisheight])

    print "Assesstransmission"
    for l in range(len(energies)):
        print "  Energy: ",energies[l]
        plt.axes(axisTransm)
        fsn=[h['FSN'] for h in params if abs(h['Energy']-energies[l])<2]
        transm1=[h['Transm'] for h in params if abs(h['Energy']-energies[l])<2]
        legend1='Energy (not calibrated) = %.1f eV\n Mean T = %.4f, std %.4f' % (energies[l],np.mean(transm1),np.std(transm1))
        print "    Transmission: mean=",np.mean(transm1),"std=",np.std(transm1)
        plt.plot(fsn,transm1,'-o',label=legend1,linewidth=1)
        plt.ylabel('Transmission')
        plt.xlabel('FSN')
        plt.grid('on')

        plt.axes(axisBcX)
        orix1=[h['BeamPosX'] for h in params if abs(h['Energy']-energies[l])<2]
        legend2='Energy (not calibrated) = %.1f eV\n Mean x = %.4f, std %.4f' % (energies[l],np.mean(orix1),np.std(orix1))
        print "    BeamcenterX: mean=",np.mean(orix1),"std=",np.std(orix1)
        plt.plot(fsn,orix1,'-o',label=legend2,linewidth=1)
        plt.ylabel('Position of beam center in X')
        plt.xlabel('FSN')
        plt.grid('on')
        
        plt.axes(axisBcY)
        oriy1=[h['BeamPosY'] for h in params if abs(h['Energy']-energies[l])<2]
        legend3='Energy (not calibrated) = %.1f eV\n Mean y = %.4f, std %.4f' % (energies[l],np.mean(oriy1),np.std(oriy1))
        print "    BeamcenterY: mean=",np.mean(oriy1),"std=",np.std(oriy1)
        plt.plot(fsn,oriy1,'-o',label=legend3,linewidth=1)
        plt.ylabel('Position of beam center in Y')
        plt.xlabel('FSN')
        plt.grid('on')
        
        plt.axes(axisDORIS)
        doris1=[h['Current1'] for h in header if abs(h['Energy']-energies[l])<2]
        legend4='Energy (not calibrated) = %.1f eV\n Mean I = %.4f' % (energies[l],np.mean(doris1))
        print "    Doris current: mean=",np.mean(doris1),"std=",np.std(doris1)
        fsnh=[h['FSN'] for h in header if abs(h['Energy']-energies[l])<2]
        plt.plot(fsnh,doris1,'o',label=legend4,linewidth=1)
        plt.ylabel('Doris current (mA)')
        plt.xlabel('FSN')
        plt.grid('on')
        
    plt.axes(axisTransm)
    plt.legend(loc='center left',bbox_to_anchor=(hspacing+hborder+axiswidth,1-vborder-0*vspacing-0.5*axisheight), bbox_transform=plt.gcf().transFigure,prop={'size':'xx-small'})
    plt.axes(axisBcX)
    plt.legend(loc='center left',bbox_to_anchor=(hspacing+hborder+axiswidth,1-vborder-1*vspacing-1.5*axisheight), bbox_transform=plt.gcf().transFigure,prop={'size':'xx-small'})
    plt.axes(axisBcY)
    plt.legend(loc='center left',bbox_to_anchor=(hspacing+hborder+axiswidth,1-vborder-2*vspacing-2.5*axisheight), bbox_transform=plt.gcf().transFigure,prop={'size':'xx-small'})
    plt.axes(axisDORIS)
    plt.legend(loc='center left',bbox_to_anchor=(hspacing+hborder+axiswidth,1-vborder-3*vspacing-3.5*axisheight), bbox_transform=plt.gcf().transFigure,prop={'size':'xx-small'})
    
def findpeak(xdata,ydata,prompt=None,mode='Lorentz',scaling='lin',blind=False,return_error=False):
    """GUI tool for locating peaks by zooming on them
    
    Inputs:
        xdata: x dataset
        ydata: y dataset
        prompt: prompt to display as a title
        mode: 'Lorentz' or 'Gauss'
        scaling: scaling of the y axis. 'lin' or 'log' 
        blind: do everything blindly (no user interaction)
        return_error: return the error of the peak position as well.
        
    Outputs:
        the peak position, and if return_error is True, the error of it
        too
        
    Usage:
        Zoom to the desired peak then press ENTER on the figure.
    """
    xdata=xdata.flatten()
    ydata=ydata.flatten()
    if not blind:
        if scaling=='log':
            plt.semilogy(xdata,ydata,'b.-')
        else:
            plt.plot(xdata,ydata,'b.-')
        if prompt is None:
            prompt='Please zoom to the peak you want to select, then press ENTER'
        plt.title(prompt)
        plt.draw()
        plt.gcf().show()
        print(prompt)
        while (plt.waitforbuttonpress() is not True):
            pass
        a=plt.axis()
        indices=((xdata<=a[1])&(xdata>=a[0]))&((ydata<=a[3])&(ydata>=a[2]))
        x1=xdata[indices]
        y1=ydata[indices]
    else:
        x1=xdata
        y1=ydata
    def gausscostfun(p,x,y):  #p: A,sigma,x0,y0
        tmp= y-p[3]-p[0]/(np.sqrt(2*np.pi)*p[1])*np.exp(-(x-p[2])**2/(2*p[1]**2))
        return tmp
    def lorentzcostfun(p,x,y):
        tmp=y-p[3]-p[0]*utils.lorentzian(p[2],p[1],x)
        return tmp
    x2=np.linspace(x1.min(),x1.max(),len(x1)*10)
    if mode=='Gauss':
        sigma0=0.25*(x1[-1]-x1[0])
        p0=((y1.max()-y1.min())/(1/np.sqrt(2*np.pi*sigma0**2)),
            sigma0,
            0.5*(x1[-1]+x1[0]),
            y1.min())
        res=scipy.optimize.leastsq(gausscostfun,p0,args=(x1,y1),maxfev=10000,full_output=True)
        p1=res[0]
        cov=res[1]
        if not blind:
            if scaling=='log':
                plt.semilogy(x2,p1[3]+p1[0]/(np.sqrt(2*np.pi)*p1[1])*np.exp(-(x2-p1[2])**2/(2*p1[1]**2)),'r-')
            else:
                plt.plot(x2,p1[3]+p1[0]/(np.sqrt(2*np.pi)*p1[1])*np.exp(-(x2-p1[2])**2/(2*p1[1]**2)),'r-')
    elif mode=='Lorentz':
        sigma0=0.25*(x1[-1]-x1[0])
        p0=((y1.max()-y1.min())/(1/sigma0),
            sigma0,
            0.5*(x1[-1]+x1[0]),
            y1.min())
        res=scipy.optimize.leastsq(lorentzcostfun,p0,args=(x1,y1),maxfev=10000,full_output=True)
        p1=res[0]
        cov=res[1]
        if not blind:
            if scaling=='log':
                plt.semilogy(x2,p1[3]+p1[0]*utils.lorentzian(p1[2],p1[1],x2),'r-')
            else:
                plt.plot(x2,p1[3]+p1[0]*utils.lorentzian(p1[2],p1[1],x2),'r-')
    else:
        raise ValueError('Only Gauss and Lorentz modes are supported in findpeak()')
    if not blind:
        plt.gcf().show()
        plt.draw()
    if return_error:
        return p1[2],np.sqrt(cov[2][2])
    else:
        return p1[2]
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
            ax=plt.gca().axis()
        plt.cla()
        plt.loglog(xdata,ydata,'.',color='blue')
        plt.loglog(xdata,modelfun(xdata,*(plt.gcf().params)),color='red')
        plt.draw()
        if keepzoom:
            plt.gca().axis(ax)
    fig=plt.figure()
    ax=[]
    sl=[]
    fig.params=[]
    for i in range(len(fitparams)):
        ax.append(plt.axes((0.1,0.1+i*0.8/len(fitparams),0.3,0.75/len(fitparams))))
        if fitparams[i]['mode']=='lin':
            sl.append(matplotlib.widgets.Slider(ax[-1],fitparams[i]['Label'],fitparams[i]['Min'],fitparams[i]['Max'],fitparams[i]['Val']))
        elif fitparams[i]['mode']=='log':
            sl.append(matplotlib.widgets.Slider(ax[-1],fitparams[i]['Label'],np.log10(fitparams[i]['Min']),np.log10(fitparams[i]['Max']),np.log10(fitparams[i]['Val'])))
        else:
            raise ValueError('Invalid mode %s in fitparams' % fitparams[i]['mode']);
        fig.params.append(fitparams[i]['Val'])
        def setfun(val,parnum=i,sl=sl[-1],mode=fitparams[i]['mode']):
            if mode=='lin':
                plt.gcf().params[parnum]=sl.val;
            elif mode=='log':
                plt.gcf().params[parnum]=pow(10,sl.val);
            else:
                pass
            redraw()
        sl[-1].on_changed(setfun)
    plt.axes((0.5,0.1,0.4,0.8))
    redraw(False)
def plotasa(asadata):
    """Plot SAXS/WAXS measurement read by readasa().
    
    Input:
        asadata: ASA dictionary (see readasa()
    
    Output:
        none, a graph is plotted.
    """
    plt.figure()
    plt.subplot(211)
    plt.plot(np.arange(len(asadata['position'])),asadata['position'],label='Intensity',color='black')
    plt.xlabel('Channel number')
    plt.ylabel('Counts')
    plt.title('Scattering data')
    plt.legend(loc='best')
    plt.subplot(212)
    x=np.arange(len(asadata['energy']))
    e1=asadata['energy'][(x<asadata['params']['Energywindow_Low'])]
    x1=x[(x<asadata['params']['Energywindow_Low'])]
    e2=asadata['energy'][(x>=asadata['params']['Energywindow_Low']) &
                         (x<=asadata['params']['Energywindow_High'])]
    x2=x[(x>=asadata['params']['Energywindow_Low']) &
         (x<=asadata['params']['Energywindow_High'])]
    e3=asadata['energy'][(x>asadata['params']['Energywindow_High'])]
    x3=x[(x>asadata['params']['Energywindow_High'])]

    plt.plot(x1,e1,label='excluded',color='red')
    plt.plot(x2,e2,label='included',color='black')
    plt.plot(x3,e3,color='red')
    plt.xlabel('Energy channel number')
    plt.ylabel('Counts')
    plt.title('Energy (pulse-area) spectrum')
    plt.legend(loc='best')
    plt.suptitle(asadata['params']['Title'])

def fitperiodicity(data,ns,mode='Lorentz',scaling='log'):
    """Determine periodicity from q values of Bragg-peaks

    Inputs:
        data: users have two possibilities:
            1) a 1D scattering dictionary
            2) a list (list, tuple, np.ndarray) of q values
        ns: a list of diffraction orders (n-s)
        mode: type of fit. See the corresponding parameter of findpeak() for
            details.
        scaling: linear ('lin') or logarithmic ('log') scale on the ordinate

    Outputs: d0, dd
        d0: the mean value for d
        dd: the standard deviation of d

    Notes:
        the way this function works depends on the format of parameter
        <data>. In case 1), a matplotlib window pops up and prompts the
        user to zoom on peaks found in <ns>. A Gaussian curve will be
        fit on the peaks, to determine the q values corresponding to the
        Bragg reflections. After this, a linear fit (with the constant
        term being 0) will be done, by taking the errors of q-s in
        account. In case 2), only the fit is done, neglecting the
        (unknown) error of q.
    """
    try:
        qs=[]
        dqs=[]
        for n in ns:
            q,dq=findpeak(data['q'],data['Intensity'],prompt='Zoom to peak %d and press ENTER' % n,scaling=scaling,return_error=True,mode=mode,)
            qs.append(q)
            dqs.append(dq)
            print "determined peak",n,"to be",q,"+/-",dq
        ns=np.array(ns)
        qs=np.array(qs)
        dqs=np.array(dqs)
        a,aerr=fitting.propfit(ns,qs,dqs)
    except KeyError:
        qs=np.array(data)
        ns=np.array(ns)
        a,aerr=fitting.propfit(ns,qs,None)
    return 2*np.pi/a, aerr*2*np.pi/a**2
    
def azimsectorgui(data,dataerr,param,mask):
    """GUI tool for sector- and azimuthal integration
    
    Inputs:
        data: 2D scattering matrix, as loaded by B1io.read2dintfile
        dataerr: error matrix corresponding to the data matrix
        param: header structure
        mask: mask matrix (1 is unmasked, 0 is masked)
    """
    fig=plt.gcf()
    fig.clf()
    fig.userdata={} # a la Matlab(R)
    def _plotting(tmp=None,rescaleazim=True,rescalerad=True):
        plt.axes(fig.userdata['ax_2dplot'])
        plt.cla()
        plot2dmatrix(fig.userdata['data'],mask=fig.userdata['lastmask'],header=fig.userdata['param'])
        plt.axes(fig.userdata['ax_sector'])
        if not fig.userdata['keeprad']:
            plt.cla()
        if not rescalerad:
            a=plt.axis()
        valididx=np.isfinite(fig.userdata['sectorintensity'])
        plt.semilogy(fig.userdata['qrange'][valididx],fig.userdata['sectorintensity'][valididx])
        plt.xlabel(u'q (1/%c)' % 197)
        plt.ylabel('Intensity')
        if not rescalerad:
            plt.axis(a)
        plt.axes(fig.userdata['ax_azim'])
        if not fig.userdata['keepazim']:
            plt.cla()
        if not rescaleazim:
            a=plt.axis()
        plt.plot(fig.userdata['thetarange'],fig.userdata['azimintensity'])
        plt.xlabel('theta (rad)')
        plt.ylabel('Intensity')
        if not rescaleazim:
            plt.axis(a)
    def _calculation_rad(tmp=None):
        print "radial integration. phi0:",fig.userdata['phi0'],"dphi:",fig.userdata['dphi']
        qmin,qmax,Nq=utils2d.qrangefrommask(fig.userdata['mask'],\
                                            fig.userdata['param']['EnergyCalibrated'],\
                                            fig.userdata['param']['Dist'],\
                                            fig.userdata['param']['PixelSize'],\
                                            fig.userdata['param']['BeamPosX'],\
                                            fig.userdata['param']['BeamPosY'])
        qrange=np.linspace(qmin,qmax,Nq)
        print "qrange before radintC:",qmin,"-",qmax
        q1,int1,err1,area1,mask1=utils2d.radintC(fig.userdata['data'],\
                                                 fig.userdata['dataerr'],\
                                                 fig.userdata['param']['EnergyCalibrated'],\
                                                 fig.userdata['param']['Dist'],\
                                                 fig.userdata['param']['PixelSize'],\
                                                 fig.userdata['param']['BeamPosX'],\
                                                 fig.userdata['param']['BeamPosY'],\
                                                 (1-fig.userdata['mask']).astype('uint8'),\
                                                 q=qrange,\
                                                 returnavgq=True,\
                                                 phi0=fig.userdata['phi0'],\
                                                 dphi=fig.userdata['dphi'],\
                                                 returnmask=True)
        print "qrange after radintC:",np.nanmin(q1),"-",np.nanmax(q1)
        fig.userdata['qrange']=q1
        fig.userdata['sectorintensity']=int1
        fig.userdata['lastmask']=1-mask1
    def _calculation_azim(tmp=None):
        print "azimuthal integration. qmin=",fig.userdata['qmin'],"qmax=",fig.userdata['qmax']
        theta1,azim1,aerr1,aarea1,amask1=utils2d.azimintqC(fig.userdata['data'],\
                                                 fig.userdata['dataerr'],\
                                                 fig.userdata['param']['EnergyCalibrated'],\
                                                 fig.userdata['param']['Dist'],\
                                                 fig.userdata['param']['PixelSize'],\
                                                 (fig.userdata['param']['BeamPosX'],\
                                                 fig.userdata['param']['BeamPosY']),\
                                                 (1-fig.userdata['mask']).astype('uint8'),\
                                                 Ntheta=fig.userdata['Ntheta'],\
                                                 qmin=fig.userdata['qmin'],\
                                                 qmax=fig.userdata['qmax'],\
                                                 returnmask=True)
        fig.userdata['thetarange']=np.hstack([theta1,theta1+2*np.pi])
        fig.userdata['azimintensity']=np.hstack([azim1,azim1])
        fig.userdata['lastmask']=1-amask1
    def dokeeprad(tmp=None):
        fig.userdata['keeprad']=not fig.userdata['keeprad']
    def dokeepazim(tmp=None):
        fig.userdata['keepazim']=not fig.userdata['keepazim']
    def dorad2azim(tmp=None):
        plt.axes(fig.userdata['ax_sector'])
        a=plt.axis()
        indices=(fig.userdata['qrange']<=a[1]) & (fig.userdata['qrange']>=a[0]) & (fig.userdata['sectorintensity']<=a[3]) & (fig.userdata['sectorintensity']>=a[2])
        fig.userdata['qmin']=np.nanmin(fig.userdata['qrange'][indices])
        fig.userdata['qmax']=np.nanmax(fig.userdata['qrange'][indices])
        print "setting qmin to",fig.userdata['qmin'],"and qmax to",fig.userdata['qmax']
        _calculation_azim()
        _plotting(rescalerad=False)
    def doazim2rad(tmp=None):
        plt.axes(fig.userdata['ax_sector'])
        a=plt.axis()
        indices=(fig.userdata['thetarange']<=a[1]) & (fig.userdata['thetarange']>=a[0]) & (fig.userdata['azimintensity']<=a[3]) & (fig.userdata['azimintensity']>=a[2])
        print fig.userdata['thetarange']
        print indices
        fig.userdata['phi0']=np.nanmin(fig.userdata['thetarange'][indices])
        fig.userdata['dphi']=np.nanmax(fig.userdata['thetarange'][indices])-fig.userdata['phi0']
        _calculation_rad()
        _plotting(rescaleazim=False)
    def doreset(tmp=None):
        fig.userdata['qmin']=0
        fig.userdata['qmax']=np.inf
        fig.userdata['phi0']=0
        fig.userdata['dphi']=3*np.pi
        _calculation_rad()
        _calculation_azim()
        fig.userdata['lastmask']=fig.userdata['mask']
        _plotting()
    def dodone(tmp=None):
        fig.userdata['done']=True

    fig.userdata['keeprad']=False
    fig.userdata['keepazim']=False
    fig.userdata['done']=False
    fig.userdata['data']=data
    fig.userdata['dataerr']=dataerr
    fig.userdata['mask']=mask
    fig.userdata['lastmask']=mask
    fig.userdata['param']=param
    fig.userdata['phi0']=0
    fig.userdata['phi0']=3*np.pi
    fig.userdata['qmin']=0
    fig.userdata['qmax']=np.inf
    fig.userdata['Ntheta']=50
    fig.userdata['Nq']=50    
    buttonarea=(.6,.1,.35,.35)
    buttons=['Radial -> Azimuthal','Azimuthal -> Radial','!Keep Radial','!Keep Azimuthal','Reset','Done']
    buttonfuncs=[dorad2azim,doazim2rad,dokeeprad,dokeepazim,doreset,dodone]
    fig.userdata['ax_2dplot']=plt.axes((.1,.6,.35,.35))
    fig.userdata['ax_sector']=plt.axes((.1,.1,.35,.35))
    fig.userdata['ax_azim']=plt.axes((.6,.6,.35,.35))
    fig.userdata['ax_btns']=[]
    fig.userdata['btns']=[]
    buttonheight=buttonarea[3]/(len(buttons)*1.5-0.5)
    for i in range(len(buttons)):
        fig.userdata['ax_btns'].append(plt.axes((buttonarea[0],\
                                   buttonarea[1]+buttonarea[3]-(i+1)*buttonheight-i*buttonheight*0.5,\
                                   buttonarea[2],buttonheight)))
        if buttons[i][0]=='!': #checkbox
            fig.userdata['btns'].append(matplotlib.widgets.CheckButtons(fig.userdata['ax_btns'][-1],[buttons[i][1:]],[0]))
            fig.userdata['btns'][-1].on_clicked(buttonfuncs[i])
        else: # button
            fig.userdata['btns'].append(matplotlib.widgets.Button(fig.userdata['ax_btns'][-1],buttons[i]))
            fig.userdata['btns'][-1].on_clicked(buttonfuncs[i])
    doreset()
    while not fig.userdata['done']:
        fig.waitforbuttonpress()
def assessmostab(fsns,dirs='.'):
    """Draw graphs on the current figure to assess the status of the monochromator stabilisator.
    
    Inputs:
        fsns: range of file sequence numbers
        dirs: list of directories to search header files in.
    """
    header=B1io.readheader('org_',fsns,'.header',dirs)
    fsn=[h['FSN'] for h in header]
    monitor=[h['Monitor']/h['MeasTime'] for h in header]
    doris=[h['MonitorDORIS']/h['MeasTime'] for h in header]
    piezo=[h['MonitorPIEZO']/h['MeasTime'] for h in header]
    energies=[h['Energy'] for h in header]
    plt.clf()
    plt.gcf().subplotpars.wspace=0.4
    plt.subplot(2,2,1);
    plt.plot(fsn,monitor,'b',label='Monitor')
    plt.ylabel('Monitor cps',color='blue')
    plt.xlabel('FSN')
    plt.twinx()
    plt.plot(fsn,doris,'r',label='DORIS counter')
    plt.ylabel('DORIS cps',color='red')
    plt.subplot(2,2,2);
    plt.plot(fsn,monitor,'b',label='Monitor')
    plt.ylabel('Monitor cps',color='blue')
    plt.xlabel('FSN')
    plt.twinx()
    plt.plot(fsn,piezo,'g',label='PIEZO counter')
    plt.ylabel('PIEZO cps',color='green')
    plt.subplot(2,2,3);
    plt.plot(fsn,doris,'r',label='DORIS counter')
    plt.ylabel('DORIS cps',color='red')
    plt.xlabel('FSN')
    plt.twinx()
    plt.plot(fsn,piezo,'g',label='PIEZO counter')
    plt.ylabel('PIEZO cps',color='green')
    plt.subplot(2,2,4);
    plt.plot(fsn,map(lambda a,b:a/b,piezo,doris),'k',label='ratio')
    for e in utils.unique(energies):
        plt.plot([f for f,e1 in zip(fsn,energies) if e1==e],
                   [p/d for p,d,e1 in zip(piezo,doris,energies) if e1==e],
                   'o',label='%.2f'%e)
    plt.ylabel('Piezo/DORIS ratio',color='black')
    plt.xlabel('FSN')
    plt.twinx()
    plt.plot(fsn,map(lambda a,b:a/b,monitor,doris),'m')
    for e in utils.unique(energies):
        plt.plot([f for f,e1 in zip(fsn,energies) if e1==e],
                   [m/d for m,d,e1 in zip(monitor,doris,energies) if e1==e],
                   'o',label='%.2f eV'%e)
    plt.ylabel('Monitor/DORIS ratio',color='magenta')
    plt.legend(loc='best',prop={'size':'x-small'})
    plt.gcf().show()
    plt.draw()
