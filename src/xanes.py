#xanes.py
#XANES and EXAFS analysis

import fitting
import B1io
import os
import pylab
import guitools

def smoothabt(muddict,smoothing):
    """Smooth mu*d data with splines
    
    Inputs:
        muddict: mu*d dictionary
        smoothing: smoothing parameter for smoothcurve(x,y,smoothing,
            mode='spline')
        
    Outputs:
        a mud dictionary with the smoothed data.
    """
    sm=fitting.smoothcurve(muddict['Energy'],muddict['Mud'],smoothing,mode='spline')
    return {'Energy':muddict['Energy'][:],
            'Mud':sm,
            'Title':("%s_smooth%lf" % (muddict['Title'],smoothing)),
            'scan':muddict['scan']}
def execchooch(mud,element,edge,choochexecutable='/opt/chooch/chooch/bin/chooch',resolution=None):
    """Execute CHOOCH
    
    Inputs:
        mud: mu*d dictionary.
        element: the name of the element, eg. 'Cd'
        edge: the absorption edge to use, eg. 'K' or 'L1'
        choochexecutable: the path where the CHOOCH executable can be found.
        resolution: the resolution of the monochromator, if you want to take
            this into account.
    
    Outputs:
        f1f2 matrix. An exception is raised if running CHOOCH fails.
    """
    B1io.writechooch(mud,'choochin.tmp');
    if resolution is None:
        cmd='%s -v 0 -e %s -a %s -o choochout.tmp choochin.tmp' % (choochexecutable, element, edge)
    else:
        cmd='%s -v 0 -e %s -a %s -r %lf -o choochout.tmp choochin.tmp' % (choochexecutable, element, edge, resolution)
    print 'Running CHOOCH with command: ', cmd
    a=os.system(cmd);
    if (a==32512):
        raise IOError( "The chooch executable cannot be found at %s. Please supply another path." % choochexecutable)
    tmp=pylab.loadtxt('choochout.tmp');
    data=pylab.zeros((tmp.shape[0],3))
    data[:,0]=tmp[:,0];
    data[:,1]=tmp[:,2];
    data[:,2]=tmp[:,1];
    return data;
def xanes2f1f2(mud,smoothing,element,edge,title,substitutepoints=[],startpoint=-pylab.inf,endpoint=pylab.inf,postsmoothing=[],prechoochcutoff=[-pylab.inf,pylab.inf]):
    """Calculate anomalous correction factors from a XANES scan.
    
    Inputs:
        mud: mud dictionary
        smoothing: smoothing parameter for smoothabt().
        element: the short name of the element, as 'Cd'...
        edge: the absorption edge (eg. 'K', 'L1', ...)
        title: the title for saving files
        substitutepoints: a list of energy values, at which the mud value should
            be substituted by the average of the two neighbours. Use this to get
            rid of outiler points...
        startpoint: lower cutoff energy
        endpoint: upper cutoff energy.
        postsmoothing: list of 3-tuples. Each 3-tuple will be interpreted as
            (lower energy, upper energy, smoothing parameter). Apply this for
            elimination of non-physical oscillations from the curve.
        prechoochcutoff: A vector of two. It determines the interval which should
            be supplied to CHOOCH. You can use this to eliminate truncation
            effects introduced by spline smoothing.
        
    Outputs:
        the calculated anomalous scattering factors (f' and f'')
        files xanes_smoothing_<title>.png and xanes_chooch_<title>.png will be
            saved, as well as f1f2_<title>.dat with the f' and f'' values. The
            external program CHOOCH (by Gwyndaf Evans) is used to convert
            mu*d data to anomalous scattering factors.
    """
    pylab.clf()
    for p in substitutepoints:
        index=pylab.find(pylab.absolute(mud['Energy']-p)<1)
        mud['Mud'][index]=0.5*(mud['Mud'][index-1]+mud['Mud'][index+1])
    
    indices=mud['Energy']<endpoint;
    mud['Energy']=mud['Energy'][indices];
    mud['Mud']=mud['Mud'][indices];
    
    indices=mud['Energy']>startpoint;
    mud['Energy']=mud['Energy'][indices];
    mud['Mud']=mud['Mud'][indices];
    
    if smoothing is None:
        smoothing=guitools.testsmoothing(mud['Energy'],mud['Mud'],1e-5)
        print "Using %lf for smoothing parameter" % smoothing
    pylab.clf()
    pylab.plot(mud['Energy'],mud['Mud']);
    B=smoothabt(mud,smoothing)
    #pre-chooch cutoff
    indices=(B['Energy']<=prechoochcutoff[1]) & (B['Energy']>=prechoochcutoff[0])
    B['Energy']=B['Energy'][indices]
    B['Mud']=B['Mud'][indices]
    #plotting
    pylab.plot(B['Energy'],B['Mud'])
    pylab.legend(['$\mu d$ measured','$\mu d$ smoothed'],loc='best');
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('$\mu d$')
    pylab.title(title)
    #saving figure of the smoothed dataset
    pylab.savefig("xanes_smoothing_%s.svg" % title,dpi=300,papertype='a4',format='svg',transparent=True)
    pylab.gcf().show()
    pylab.figure()
    # CHOOCH-ing
    B1io.writechooch(B,'choochin.tmp')
    f1f2=execchooch(B,element,edge)
    # post-CHOOCH smoothing
    for p in postsmoothing:
        indices=(f1f2[:,0]<=p[1]) & (f1f2[:,0]>=p[0])
        x1=f1f2[indices,0]
        y1=f1f2[indices,1]
        z1=f1f2[indices,2]
        s=p[2]
        if p[2] is None:
            s=guitools.testsmoothing(x1,y1,1e-1,1e-2,1e1)
        f1f2[indices,1]=fitting.smoothcurve(x1,y1,s,mode='spline')
        f1f2[indices,2]=fitting.smoothcurve(x1,z1,s,mode='spline')
    #plotting
    pylab.plot(f1f2[:,0],f1f2[:,1:3]);
    pylab.xlabel('Energy (eV)')
    pylab.ylabel('$f^\'$ and $f^{\'\'}$')
    pylab.title(title)
    pylab.savefig("xanes_chooch_%s.svg" % title,dpi=300,papertype='a4',format='svg',transparent=True)
    B1io.writef1f2(f1f2,("f1f2_%s.dat" % title));
    return f1f2
