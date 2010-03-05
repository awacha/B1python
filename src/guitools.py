#-----------------------------------------------------------------------------
# Name:        guitools.py
# Purpose:     GUI utilities
#
# Author:      Andras Wacha
#
# Created:     2010/02/22
# RCS-ID:      $Id: guitools.py $
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


import types
import numpy as np
import pylab
import matplotlib.widgets
import matplotlib.nxutils
import scipy.io
import utils
import fitting
import utils2d
import time
import B1io

HC=12398.419 #Planck's constant times speed of light, in eV*Angstrom units

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
            applies as to symboll. Default value is 1.
        gui [optional]: display graphical user interface to show/hide plotted
            lines independently. Default value is False (no gui)
    """
    if type(energies)!=types.ListType:
        energies=[energies];
    colors=['blue','green','red','black','magenta'];
    if type(samplename)==types.StringType:
        samplename=[samplename]
    if type(marker)!=types.ListType:
        marker=[marker]
    if type(mult)!=types.ListType:
        mult=[mult]
    if len(marker)==1:
        marker=marker*len(samplename)
    if len(mult)==1:
        mult=mult*len(samplename)
    if (len(marker)!=len(samplename)) or (len(mult) !=len(samplename)):
        raise ValueError
    if gui==True:
        fig=pylab.figure()
        buttonax=fig.add_axes((0.65,0.1,0.3,0.05))
        guiax=fig.add_axes((0.65,0.15,0.3,0.75))
        ax=fig.add_axes((0.1,0.1,0.5,0.8))
        btn=matplotlib.widgets.Button(buttonax,'Close GUI')
        def fun(event):
            fig=pylab.gcf()
            fig.delaxes(fig.axes[0])
            fig.delaxes(fig.axes[0])
            fig.axes[0].set_position((0.1,0.1,0.8,0.85))
        btn.on_clicked(fun)
        guiax.set_title('Visibility selector')
        texts=[]
        handles=[]
    else:
        fig=pylab.gcf()
        ax=pylab.gca()
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
            pylab.draw()
def plot2dmatrix(A,maxval=None,mask=None,header=None,qs=[],showqscale=True,contour=None,pmin=0,pmax=1,blacknegative=False):
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
    """
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
#    t0=time.time()
    nonpos=(tmp<=0)
#    t1=time.time()
    tmp[nonpos]=tmp[tmp>0].min()
#    t2=time.time()
    tmp=np.log(tmp);
#    t3=time.time()
    tmp[np.isnan(tmp)]=tmp[-np.isnan(tmp)].min();
#    t4=time.time()
#    print t1-t0
#    print t2-t1
#    print t3-t2
#    print t4-t3
    if (header is not None) and (showqscale):
        xmin=0-(header['BeamPosX']-1)*header['PixelSize']
        xmax=(tmp.shape[0]-(header['BeamPosX']-1))*header['PixelSize']
        ymin=0-(header['BeamPosY']-1)*header['PixelSize']
        ymax=(tmp.shape[1]-(header['BeamPosY']-1))*header['PixelSize']
        qxmin=4*np.pi*np.sin(0.5*np.arctan(xmin/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qxmax=4*np.pi*np.sin(0.5*np.arctan(xmax/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qymin=4*np.pi*np.sin(0.5*np.arctan(ymin/header['Dist']))*header['EnergyCalibrated']/float(HC)
        qymax=4*np.pi*np.sin(0.5*np.arctan(ymax/header['Dist']))*header['EnergyCalibrated']/float(HC)
        extent=[qymin,qymax,qxmin,qxmax]
    else:
        extent=None
    if contour is None:
        pylab.imshow(tmp,extent=extent,interpolation='nearest',vmin=tmp.min()+pmin*(tmp.max()-tmp.min()),vmax=tmp.min()+pmax*(tmp.max()-tmp.min()));
    else:
        if extent is None:
            extent1=[1,tmp.shape[0],1,tmp.shape[1]]
        else:
            extent1=extent;
        X,Y=np.meshgrid(np.linspace(extent1[2],extent1[3],tmp.shape[1]),
                           np.linspace(extent1[0],extent1[1],tmp.shape[0]))
        pylab.contour(X,Y,tmp,contour)
    if blacknegative:
        black=np.zeros((A.shape[0],A.shape[1],4))
        black[:,:,3][nonpos]=1
        pylab.imshow(black,extent=extent,interpolation='nearest')
    if mask is not None:
        white=np.ones((mask.shape[0],mask.shape[1],4))
        white[:,:,3]=np.array(1-mask).astype('float')*0.7
        pylab.imshow(white,extent=extent,interpolation='nearest')
    for q in qs:
        a=pylab.gca().axis()
        pylab.plot(q*np.cos(np.linspace(0,2*np.pi,2000)),
                   q*np.sin(np.linspace(0,2*np.pi,2000)),
                   color='white',linewidth=3)
        pylab.gca().axis(a)
    if header is not None:
        pylab.title("#%s: %s" % (header['FSN'], header['Title']))

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
        fig=pylab.gcf()
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
            pylab.gcf().toexit=True
    if mask is None:
        mask=np.ones(A.shape)
    if A.shape!=mask.shape:
        print 'The shapes of A and mask should be equal.'
        return None
    fig=pylab.gcf();
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
    conn_id=fig.canvas.mpl_connect('button_press_event',clickevent)
    fig.toexit=False
    fig.show()
    firstdraw=1;
    while fig.toexit==False:
        if fig.mydata['redrawneeded']:
            if not firstdraw:
                ax=fig.mydata['ax'].axis();
            fig.mydata['redrawneeded']=False
            fig.mydata['ax'].cla()
            pylab.axes(fig.mydata['ax'])
            plot2dmatrix(A,mask=fig.mydata['mask'])
            fig.mydata['ax'].set_title('')
            if not firstdraw:
                fig.mydata['ax'].axis(ax);
            firstdraw=0;
        pylab.draw()
        fig.waitforbuttonpress()
    #ax.imshow(maskplot)
    #pylab.show()
    mask=fig.mydata['mask']
    #pylab.close(fig)
    fig.clf()
#    pylab.title('Mask Done')
    if savefile is not None:
        print 'Saving file'
        scipy.io.savemat(savefile,{'mask':mask})
    return mask
    

def basicfittinggui(data,title='',blocking=False):
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
    fig=pylab.figure()
    pylab.clf()
    plots=['Guinier','Guinier thickness','Guinier cross-section','Porod','lin-lin','lin-log','log-lin','log-log']
    buttons=['Guinier','Guinier thickness','Guinier cross-section','Porod','Power law','Power law with c.background','Power law with l.background']
    fitfuns=[fitting.guinierfit,fitting.guinierthicknessfit,fitting.guiniercrosssectionfit,fitting.porodfit,fitting.powerfit,fitting.powerfitwithbackground,fitting.powerfitwithlinearbackground]
    for i in range(len(buttons)):
        ax=pylab.axes((leftborder,topborder-(i+1)*(0.8)/(len(buttons)+len(plots)),leftbox_end,0.7/(len(buttons)+len(plots))))
        but=matplotlib.widgets.Button(ax,buttons[i])
        def onclick(A=None,B=None,data=data,type=buttons[i],fitfun=fitfuns[i]):
            a=pylab.axis()
            plottype=pylab.gcf().plottype
            pylab.figure()
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
            if len(res)==4:
                pylab.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg' % (type,res[0],res[2],res[1],res[3]))
            elif len(res)==6:
                pylab.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg;\n %lg +/- %lg' % (type,res[0],res[3],res[1],res[4],res[2],res[5]))
            elif len(res)==8:
                pylab.title('%s fit on dataset.\nParameters: %lg +/- %lg ; %lg +/- %lg;\n %lg +/- %lg; %lg +/- %lg' % (type,res[0],res[4],res[1],res[5],res[2],res[6],res[3],res[7]))
            pylab.gcf().show()
        but.on_clicked(onclick)
    ax=pylab.axes((leftborder,topborder-(len(buttons)+len(plots))*(0.8)/(len(buttons)+len(plots)),leftbox_end,0.7/(len(buttons)+len(plots))*len(plots) ))
    pylab.title('Plot types')
    rb=matplotlib.widgets.RadioButtons(ax,plots,active=7)
    if blocking:
        ax=pylab.axes((leftborder,0.03,leftbox_end,bottomborder-0.03))
        b=matplotlib.widgets.Button(ax,"Done")
        fig=pylab.gcf()
        fig.fittingdone=False
        def onclick(A=None,B=None,fig=fig):
            fig.fittingdone=True
            blocking=False
            print "blocking should now end"
        b.on_clicked(onclick)
    pylab.axes((0.4,bottomborder,0.5,0.8))
    def onselectplottype(plottype,q=data['q'],I=data['Intensity'],title=title):
        pylab.cla()
        pylab.gcf().plottype==plottype
        if plottype=='Guinier':
            x=q**2
            y=np.log(I)
            pylab.plot(x,y,'.')
            pylab.xlabel('q^2')
            pylab.ylabel('ln I')
        elif plottype=='Guinier thickness':
            x=q**2
            y=np.log(I)*q**2
            pylab.plot(x,y,'.')
            pylab.xlabel('q^2')
            pylab.ylabel('ln I*q^2')
        elif plottype=='Guinier cross-section':
            x=q**2
            y=np.log(I)*q
            pylab.plot(x,y,'.')
            pylab.xlabel('q^2')
            pylab.ylabel('ln I*q')            
        elif plottype=='Porod':
            x=q**4
            y=I*q**4
            pylab.plot(x,y,'.')
            pylab.xlabel('q^4')
            pylab.ylabel('I*q^4')
        elif plottype=='lin-lin':
            pylab.plot(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        elif plottype=='lin-log':
            pylab.semilogx(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        elif plottype=='log-lin':
            pylab.semilogy(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        elif plottype=='log-log':
            pylab.loglog(q,I,'.')
            pylab.xlabel('q')
            pylab.ylabel('I')
        pylab.title(title)
        pylab.gcf().plottype=plottype
        pylab.gcf().show()
    rb.on_clicked(onselectplottype)
    pylab.title(title)
    pylab.loglog(data['q'],data['Intensity'],'.')
    pylab.gcf().plottype='log-log'
    pylab.gcf().show()
    pylab.draw()
    fig=pylab.gcf()
    while blocking:
        fig.waitforbuttonpress()
        print "buttonpress"
        if fig.fittingdone:
            blocking=False
            print "exiting"
    print "returning"
    return listoffits
        
#data quality tools
def testsmoothing(x,y,smoothing=1e-5,slidermin=1e-6,slidermax=1e-2):
    ax=pylab.axes((0.2,0.85,0.7,0.05));
    sl=matplotlib.widgets.Slider(ax,'',np.log10(slidermin),np.log10(slidermax),np.log10(smoothing));
    fig=pylab.gcf()
    fig.smoothingdone=False
    ax=pylab.axes((0.1,0.85,0.1,0.05));
    def butoff(a=None):
        pylab.gcf().smoothingdone=True
    but=matplotlib.widgets.Button(ax,'Ok')
    but.on_clicked(butoff)
    pylab.axes((0.1,0.1,0.8,0.7))
    pylab.cla()
    pylab.plot(x,y,'.')
    smoothing=pow(10,sl.val);
    y1=fitting.smoothcurve(x,y,smoothing,mode='spline')
    pylab.plot(x,y1,linewidth=2)
    def fun(a):
        ax=pylab.axis()
        pylab.cla()
        pylab.plot(x,y,'.')
        smoothing=pow(10,sl.val);
        y1=fitting.smoothcurve(x,y,smoothing,mode='spline')
        pylab.plot(x,y1,linewidth=2)
        pylab.axis(ax)
    sl.on_changed(fun)
    fun(1e-5)
    while not fig.smoothingdone:
        pylab.waitforbuttonpress()
    pylab.clf()
    pylab.plot(x,y,'.')
    y1=fitting.smoothcurve(x,y,pow(10,sl.val),mode='spline')
    pylab.plot(x,y1,linewidth=2)
    pylab.draw()
    return pow(10,sl.val)

def testorigin(data,orig,mask=None,dmin=0,dmax=np.inf):
    """Shows several test plots by which the validity of the determined origin
    can  be tested.
    
    Inputs:
        data: the 2d scattering image
        orig: the origin [row,column]
        mask: the mask matrix. Nonzero means nonmasked
    """
    print "Creating origin testing images, please wait..."
    if mask is None:
        mask=np.ones(data.shape)
    pylab.subplot(2,2,1)
    plot2dmatrix(data,mask=mask)
    pylab.plot([0,data.shape[1]],[orig[0],orig[0]],color='white')
    pylab.plot([orig[1],orig[1]],[0,data.shape[0]],color='white')
    pylab.gca().axis('tight')
    pylab.subplot(2,2,2)
    c1,nc1=utils2d.imageint(data,orig,1-mask,35,20)
    c2,nc2=utils2d.imageint(data,orig,1-mask,35+90,20)
    c3,nc3=utils2d.imageint(data,orig,1-mask,35+180,20)
    c4,nc4=utils2d.imageint(data,orig,1-mask,35+270,20)
    pylab.plot(c1,marker='.',color='blue',markersize=3)
    pylab.plot(c3,marker='o',color='blue',markersize=6)
    pylab.plot(c2,marker='.',color='red',markersize=3)
    pylab.plot(c4,marker='o',color='red',markersize=6)
    pylab.subplot(2,2,3)
    maxr=max([len(c1),len(c2),len(c3),len(c4)])
    pdata=utils2d.polartransform(data,np.arange(0,maxr,dtype=np.double),np.linspace(0,4*np.pi,600),orig[0],orig[1])
    pmask=utils2d.polartransform(mask,np.arange(0,maxr,dtype=np.double),np.linspace(0,4*np.pi,600),orig[0],orig[1])
    plot2dmatrix(pdata,mask=pmask)
    pylab.axis('scaled')
    pylab.subplot(2,2,4)
    t,I,E,A=utils2d.azimintpix(data,np.ones(data.shape),orig,1-mask,dmin,dmin,dmax)
    pylab.plot(t,I,'b-')
    pylab.ylabel('Azimuthal intensity\n(nonperiodic for 2pi)')
    pylab.twinx()
    pylab.plot(t,A,'g-')
    pylab.ylabel('Effective area\n(should be definitely flat)')
    pylab.gcf().show()
    print "... image ready!"
def assesstransmission(fsns,titleofsample,mode='Gabriel',dirs=[]):
    """Plot transmission, beam center and Doris current vs. FSNs of the given
    sample.
    
    Inputs:
        fsns: range of file sequence numbers
        titleofsample: the title of the sample which should be investigated
        mode: 'Gabriel' if the measurements were made with the gas-detector, 
            and 'Pilatus300k' if that detector was used.            
    """
    if type(fsns)!=types.ListType:
        fsns=[fsns]
    if mode=='Gabriel':
        header1=B1io.readheader('ORG',fsns,'.DAT',dirs=dirs)
    elif mode=='Pilatus300k':
        header1=B1io.readheader('org_',fsns,'.header',dirs=dirs)
    else:
        print "invalid mode argument. Possible values: 'Gabriel', 'Pilatus300k'"
        return
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

    doris=[h['Current1'] for h in header]
    orix=[h['BeamPosX'] for h in params]
    oriy=[h['BeamPosY'] for h in params]
    legend1=[]
    legend2=[]
    legend3=[]
    legend4=[]
    print "Assesstransmission"
    for l in range(len(energies)):
        print "  Energy: ",energies[l]
        pylab.subplot(4,1,1)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        fsn=[h['FSN'] for h in header if abs(h['Energy']-energies[l])<2]
        transm1=[h['Transm'] for h in params if abs(h['Energy']-energies[l])<2]
        print "    Transmission: mean=",np.mean(transm1),"std=",np.std(transm1)
        pylab.plot(fsn,transm1,'-o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Transmission')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend1=legend1+['Energy (not calibrated) = %.1f eV\n Mean T = %.4f, std %.4f' % (energies[l],np.mean(transm1),np.std(transm1))]
        pylab.subplot(4,1,2)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        orix1=[h['BeamPosX'] for h in params if abs(h['Energy']-energies[l])<2]
        print "    BeamcenterX: mean=",np.mean(orix1),"std=",np.std(orix1)
        pylab.plot(fsn,orix1,'-o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Position of beam center in X')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend2=legend2+['Energy (not calibrated) = %.1f eV\n Mean x = %.4f, std %.4f' % (energies[l],np.mean(orix1),np.std(orix1))]
        pylab.subplot(4,1,3)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        oriy1=[h['BeamPosY'] for h in params if abs(h['Energy']-energies[l])<2]
        print "    BeamcenterY: mean=",np.mean(oriy1),"std=",np.std(oriy1)
        pylab.plot(fsn,oriy1,'-o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Position of beam center in Y')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend3=legend3+['Energy (not calibrated) = %.1f eV\n Mean y = %.4f, std %.4f' % (energies[l],np.mean(oriy1),np.std(oriy1))]
        pylab.subplot(4,1,4)
        bbox=pylab.gca().get_position()
        pylab.gca().set_position([bbox.x0,bbox.y0,(bbox.x1-bbox.x0)*0.9,bbox.y1-bbox.y0])
        doris1=[h['Current1'] for h in header if abs(h['Energy']-energies[l])<2]
        print "    Doris current: mean=",np.mean(doris1),"std=",np.std(doris1)
        pylab.plot(fsn,doris1,'o',
                  markerfacecolor=(1/(l+1),(len(energies)-l)/len(energies),0.6),
                  linewidth=1)
        pylab.ylabel('Doris current (mA)')
        pylab.xlabel('FSN')
        pylab.grid('on')
        legend4=legend4+['Energy (not calibrated) = %.1f eV\n Mean I = %.4f' % (energies[l],np.mean(doris1))]
        
    pylab.subplot(4,1,1)
    pylab.legend(legend1,loc=(1.03,0))
    pylab.subplot(4,1,2)
    pylab.legend(legend2,loc=(1.03,0))
    pylab.subplot(4,1,3)
    pylab.legend(legend3,loc=(1.03,0))
    pylab.subplot(4,1,4)
    pylab.legend(legend4,loc=(1.03,0))
    
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
            pylab.semilogy(xdata,ydata,'b.')
        else:
            pylab.plot(xdata,ydata,'b.')
        if prompt is None:
            prompt='Please zoom to the peak you want to select, then press ENTER'
        pylab.title(prompt)
        pylab.gcf().show()
        print(prompt)
        while (pylab.waitforbuttonpress() is not True):
            pass
        a=pylab.axis()
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
    if mode=='Gauss':
        sigma0=0.25*(x1[-1]-x1[0])
        p0=((y1.max()-y1.min())/(1/np.sqrt(2*np.pi*sigma0**2)),
            sigma0,
            0.5*(x1[-1]+x1[0]),
            y1.min())
        res=scipy.optimize.leastsq(gausscostfun,p0,args=(x1,y1),maxfev=10000,full_output=True)
        p1=res[0]
        cov=res[1]
        ier=res[4]
        if not blind:
            if scaling=='log':
                pylab.semilogy(x1,p1[3]+p1[0]/(np.sqrt(2*np.pi)*p1[1])*np.exp(-(x1-p1[2])**2/(2*p1[1]**2)),'r-')
            else:
                pylab.plot(x1,p1[3]+p1[0]/(np.sqrt(2*np.pi)*p1[1])*np.exp(-(x1-p1[2])**2/(2*p1[1]**2)),'r-')
    elif mode=='Lorentz':
        sigma0=0.25*(x1[-1]-x1[0])
        p0=((y1.max()-y1.min())/(1/sigma0),
            sigma0,
            0.5*(x1[-1]+x1[0]),
            y1.min())
        res=scipy.optimize.leastsq(lorentzcostfun,p0,args=(x1,y1),maxfev=10000,full_output=True)
        p1=res[0]
        cov=res[1]
        ier=res[4]
        if not blind:
            if scaling=='log':
                pylab.semilogy(x1,p1[3]+p1[0]*utils.lorentzian(p1[2],p1[1],x1),'r-')
            else:
                pylab.plot(x1,p1[3]+p1[0]*utils.lorentzian(p1[2],p1[1],x1),'r-')
    else:
        raise ValueError('Only Gauss and Lorentz modes are supported in findpeak()')
    if not blind:
        pylab.gcf().show()
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
            ax=pylab.gca().axis()
        pylab.cla()
        pylab.loglog(xdata,ydata,'.',color='blue')
        pylab.loglog(xdata,modelfun(xdata,*(pylab.gcf().params)),color='red')
        pylab.draw()
        if keepzoom:
            pylab.gca().axis(ax)
    fig=pylab.figure()
    ax=[]
    sl=[]
    fig.params=[]
    for i in range(len(fitparams)):
        ax.append(pylab.axes((0.1,0.1+i*0.8/len(fitparams),0.3,0.75/len(fitparams))))
        if fitparams[i]['mode']=='lin':
            sl.append(matplotlib.widgets.Slider(ax[-1],fitparams[i]['Label'],fitparams[i]['Min'],fitparams[i]['Max'],fitparams[i]['Val']))
        elif fitparams[i]['mode']=='log':
            sl.append(matplotlib.widgets.Slider(ax[-1],fitparams[i]['Label'],np.log10(fitparams[i]['Min']),np.log10(fitparams[i]['Max']),np.log10(fitparams[i]['Val'])))
        else:
            raise ValueError('Invalid mode %s in fitparams' % fitparams[i]['mode']);
        fig.params.append(fitparams[i]['Val'])
        def setfun(val,parnum=i,sl=sl[-1],mode=fitparams[i]['mode']):
            if mode=='lin':
                pylab.gcf().params[parnum]=sl.val;
            elif mode=='log':
                pylab.gcf().params[parnum]=pow(10,sl.val);
            else:
                pass
            redraw()
        sl[-1].on_changed(setfun)
    pylab.axes((0.5,0.1,0.4,0.8))
    redraw(False)
def plotasa(asadata):
    """Plot SAXS/WAXS measurement read by readasa().
    
    Input:
        asadata: ASA dictionary (see readasa()
    
    Output:
        none, a graph is plotted.
    """
    pylab.figure()
    pylab.subplot(211)
    pylab.plot(np.arange(len(asadata['position'])),asadata['position'],label='Intensity',color='black')
    pylab.xlabel('Channel number')
    pylab.ylabel('Counts')
    pylab.title('Scattering data')
    pylab.legend(loc='best')
    pylab.subplot(212)
    x=np.arange(len(asadata['energy']))
    e1=asadata['energy'][(x<asadata['params']['Energywindow_Low'])]
    x1=x[(x<asadata['params']['Energywindow_Low'])]
    e2=asadata['energy'][(x>=asadata['params']['Energywindow_Low']) &
                         (x<=asadata['params']['Energywindow_High'])]
    x2=x[(x>=asadata['params']['Energywindow_Low']) &
         (x<=asadata['params']['Energywindow_High'])]
    e3=asadata['energy'][(x>asadata['params']['Energywindow_High'])]
    x3=x[(x>asadata['params']['Energywindow_High'])]

    pylab.plot(x1,e1,label='excluded',color='red')
    pylab.plot(x2,e2,label='included',color='black')
    pylab.plot(x3,e3,color='red')
    pylab.xlabel('Energy channel number')
    pylab.ylabel('Counts')
    pylab.title('Energy (pulse-area) spectrum')
    pylab.legend(loc='best')
    pylab.suptitle(asadata['params']['Title'])

def fitperiodicity(data,ns):
    """Determine periodicity from q values of Bragg-peaks

    Inputs:
        data: users have two possibilities:
            1) a 1D scattering dictionary
            2) a list (list, tuple, np.ndarray) of q values
        ns: a list of diffraction orders (n-s)

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
            q,dq=findpeak(data['q'],data['Intensity'],prompt='Zoom to peak %d and press ENTER' % n,scaling='log',return_error=True)
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
    