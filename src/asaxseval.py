#asaxseval.py

import pylab
import utils
import time
import types
import B1io

def asaxsbasicfunctions(I,Errors,f1,f2,df1=None,df2=None,element=0):
    """Calculate the basic functions (nonresonant, mixed, resonant)
    
    Inputs:
        I: a matrix of intensity (scattering cross section) data. The columns
            should contain the intensities for each energy
        Errors: a matrix of absolute errors of the intensity data. Of the same
            shape as I.
        f1: vector of the f' values for the corresponding columns of I.
        f2: vector of the f'' values for the corresponding columns of I.
        element: the atomic number of the resonant atom. If zero (default),
            derive the basic functions according to Stuhrmann. If nonzero, the
            partial structure factors of the nonresonant part (N), and the
            resonant part (R) are returned, along with the cross-term S_{NR}.
            
    Outputs:
        N: vector of the nonresonant term
        M: vector of the mixed term
        R: vector of the pure resonant term
    """
    I=pylab.array(I,dtype='float');
    f1=pylab.array(f1,dtype='float');
    f2=pylab.array(f2,dtype='float');
    Nenergies=I.shape[1];
    Ilen=I.shape[0];
    if len(f1) != Nenergies:
        print "length of the f' vector should match the number of rows in I."
        return
    if len(f2) != Nenergies:
        print "length of the f'' vector should match the number of rows in I."
        return
    N=pylab.zeros((Ilen,1));
    M=pylab.zeros((Ilen,1));
    R=pylab.zeros((Ilen,1));
    DN=pylab.zeros((Ilen,1));
    DM=pylab.zeros((Ilen,1));
    DR=pylab.zeros((Ilen,1));

    A=pylab.ones((Nenergies,3));
    A[:,1]=2*(element+f1);
    A[:,2]=(element+f1)**2+f2**2;
    DA=pylab.zeros(A.shape)
    if df1 is not None:
        DA[:,1]=2*df1;
        DA[:,2]=pylab.sqrt(4*(element+f1)**2*df1**2+4*f2**2*df2**2)
    B=pylab.dot(pylab.inv(pylab.dot(A.T,A)),A.T);
    ATA=pylab.dot(A.T,A)
    ATAerr=utils.dot_error(A.T,A,DA.T,DA)
    invATA=pylab.inv(ATA)
    invATAerr=utils.inv_error(ATA,ATAerr)
    Berror=utils.dot_error(invATA,A.T,invATAerr,DA.T)
    print Berror
    print "Condition number of inv(A'*A)*A' is ",pylab.cond(B)
    for j in range(0,Ilen):
        tmp=pylab.dot(B,I[j,:])
        N[j]=tmp[0];
        M[j]=tmp[1];
        R[j]=tmp[2];
        tmpe=utils.dot_error(B,I[j,:],Berror,Errors[j,:])
        DN[j]=tmpe[0];
        DM[j]=tmpe[1];
        DR[j]=tmpe[2];
    return N,M,R,DN,DM,DR
def asaxspureresonant(I1,I2,I3,DI1,DI2,DI3,f11,f12,f13,f21,f22,f23):
    """Calculate the pure resonant as the "difference of differences"
    
    Inputs:
        I1,I2,I3: intensity curves for the three energies
        DI1,DI2,DI3: error data for the intensity curves
        f11,f12,f13: f' values
        f21,f22,f23: f'' values
    
    Outputs:
        sep12: (I1-I2)/(f11-f12)
        dsep12: error of sep12
        sep23: (I2-I3)/(f12-f13)
        dsep23: error of I2-I3
        R: the pure resonant term
        DR: the error of the pure resonant term
    """
    factor=f11-f13+(f22**2-f21**2)/(f12-f11)-(f22**2-f23**2)/(f12-f13)
    DR=pylab.sqrt((DI1*DI1)/(f12-f11)**2+
                  (DI2*DI2)*(1/(f12-f11)**2+1/(f12-f13)**2)+
                  (DI3*DI3)/(f12-f13)**2)/pylab.absolute(factor);
    sep12=(I1-I2)/(f11-f12)
    sep23=(I2-I3)/(f12-f13)
    R=(sep12 -sep23)/factor;
    dsep12=pylab.absolute(pylab.sqrt((DI1*DI1)+(DI2*DI2))/(f11-f12))
    dsep23=pylab.absolute(pylab.sqrt((DI2*DI2)+(DI3*DI3))/(f12-f13))
    return sep12,dsep12,sep23,dsep23,R,DR
def asaxsseqeval(data,param,asaxsenergies,chemshift,fprimefile,samples=None,seqname=None,element=0):
    """Evaluate an ASAXS sequence, derive the basic functions
    
    Inputs:
        data: list of data structures as read by eg. readintnorm
        param: list of parameter structures as read by eg. readintnorm
        asaxsenergies: the UNCALIBRATED (aka. "apparent") energy values for
            the ASAXS evaluation. At least 3 should be supplied.
        chemshift: chemical shift. The difference of the calibrated edge energy
            measured on the sample (E_s) and the theoretical edge energy for an
            isolated atom (E_t). If E_s>E_t then chemshift is positive.
        fprimefile: file name (can include path) for the f' data, as created
            by Hephaestus. The file should have three columns:
            enegy<whitespace>fprime<whitespace>fdoubleprime<newline>.
            Lines beginning with # are ignored.
        samples [optional]: a string or a list of strings of samplenames to be
            treated. If omitted, all samples are evaluated.
        seqname [optional]: if given, the following files will be created:
            seqname_samplename_ie.txt : summarized intensities and errors
            seqname_samplename_basicfun.txt: the asaxs basic functions with
                their errors
            seqname_samplename_separation.txt: I_0, (I_1-I_2)/(f1_1-f1_2),
                (I_2-I_3)/(f1_2-f1_3) and the pure resonant term, with their
                errors
            seqname_f1f2.eps: f' and f'' diagram
            seqname_samplename_basicfun.eps: basic functions displayed
            seqname_samplename_separation.eps: separated curves, I_0 and pure
                resonant displayed
            seqname.log: logging
        element [optional]: if nonzero, this is the atomic number of the
            resonant element. If zero (default), the evaluation is carried out
            according to Stuhrmann. Nonzero yields the PSFs.
    """
    if samples is None:
        samples=utils.unique([param[i]['Title'] for i in range(0,len(data))]);
        print "Found samples: ", samples
    if type(samples)!=types.ListType:
        samples=[samples];
    if seqname is not None:
        logfile=open('%s.log' % seqname,'wt')
        logfile.write('ASAXS sequence name: %s\n' % seqname)
        logfile.write('Time: %s' % time.asctime())
    asaxsecalib=[];
    #asaxsenergies=pylab.array(utils.unique(asaxsenergies,lambda a,b:(abs(a-b)<2)))
    asaxsenergies=pylab.array(asaxsenergies);
    for j in range(0,len(asaxsenergies)):
        asaxsecalib.append([param[i]['EnergyCalibrated']
                             for i in range(0,len(data)) 
                             if abs(param[i]['Energy']-asaxsenergies[j])<2][0]);
    asaxsecalib=pylab.array(asaxsecalib);
    
    print "Calibrated ASAXS energies:", asaxsecalib
    fprimes=B1io.readf1f2(fprimefile);
    pylab.plot(fprimes[:,0],fprimes[:,1],'b-');
    pylab.plot(fprimes[:,0],fprimes[:,2],'r-');
    asaxsf1=pylab.interp(asaxsecalib-chemshift,fprimes[:,0],fprimes[:,1]);
    asaxsf2=pylab.interp(asaxsecalib-chemshift,fprimes[:,0],fprimes[:,2]);
    print "f' values", asaxsf1
    print "f'' values", asaxsf2
    if seqname is not None:
        logfile.write('Calibrated ASAXS energies:\n')
        for i in range(len(asaxsenergies)):
            logfile.write("%f -> %f\tf1=%f\tf2=%f\n" % (asaxsenergies[i],asaxsecalib[i],asaxsf1[i],asaxsf2[i]))
        logfile.write('Chemical shift (eV): %f\n' % chemshift)
        logfile.write('Atomic number supplied by the user: %d\n' % element)
        logfile.write('fprime file: %s\n' % fprimefile)
    pylab.plot(asaxsecalib-chemshift,asaxsf1,'b.',markersize=10);
    pylab.plot(asaxsecalib-chemshift,asaxsf2,'r.',markersize=10);
    pylab.legend(['f1','f2'],loc='upper left');
    pylab.xlabel('Photon energy (eV)');
    pylab.ylabel('Anomalous corrections (e.u.)');
    pylab.title('Anomalous correction factors')
    if seqname is not None:
        pylab.savefig('%s_f1f2.eps' % seqname,dpi=300,transparent='True',format='eps')
    if len(asaxsenergies)<3:
        print "At least 3 energies should be given!"
        return
    for s in samples:
        print "Evaluating sample %s" % s
        if seqname is not None:
            logfile.write('Sample: %s\n' % s)
        q=None;
        I=None;
        E=None;
        counter=None;
        fsns=None
        for k in range(0,len(data)): #collect the intensities energy-wise.
            if param[k]['Title']!=s:
                continue
            if q is None:
                q=pylab.array(data[k]['q']);
                NQ=len(q);
                Intensity=pylab.zeros((len(q),len(asaxsenergies)))
                Errors=pylab.zeros((len(q),len(asaxsenergies)))
                counter=pylab.zeros((1,len(asaxsenergies)))
                fsns=[[] for l in range(len(asaxsenergies))]
            if pylab.sum(q-pylab.array(data[k]['q']))>0:
                print "Check the datasets once again: different q-scales!"
                continue;
            energyindex=pylab.absolute(asaxsenergies-param[k]['Energy'])<2
            Intensity[:,energyindex]=Intensity[:,energyindex]+pylab.array(data[k]['Intensity']).reshape(NQ,1);
            Errors[:,energyindex]=Intensity[:,energyindex]+(pylab.array(data[k]['Error']).reshape(NQ,1))**2;
            counter[0,energyindex]=counter[0,energyindex]+1;
            if pylab.find(len(energyindex))>0:
                print pylab.find(energyindex)[0]
                fsns[pylab.find(energyindex)[0]].append(param[k]['FSN']);
        Errors=pylab.sqrt(Errors)
        Intensity=Intensity/pylab.kron(pylab.ones((NQ,1)),counter)
        if seqname is not None:
            for i in range(0,len(asaxsenergies)):
                logfile.write('FSNs for energy #%d:' % i)
                for j in fsns[i]:
                    logfile.write('%d' % j)
                logfile.write('\n')
            datatosave=pylab.zeros((len(q),2*len(asaxsenergies)+1))
            datatosave[:,0]=q;
            for i in range(len(asaxsenergies)):
                datatosave[:,2*i+1]=Intensity[:,i]
                datatosave[:,2*i+2]=Errors[:,i]
            pylab.savetxt('%s_%s_ie.txt' % (seqname, s),datatosave,delimiter='\t')
        # now we have the Intensity and Error matrices fit to feed to asaxsbasicfunctions()
        N,M,R,DN,DM,DR=asaxsbasicfunctions(Intensity,Errors,asaxsf1,asaxsf2,element=element);
        sep12,dsep12,sep23,dsep23,R1,dR1=asaxspureresonant(Intensity[:,0],Intensity[:,1],Intensity[:,2],
                                                           Errors[:,0],Errors[:,1],Errors[:,2],
                                                           asaxsf1[0],asaxsf1[1],asaxsf1[2],
                                                           asaxsf2[0],asaxsf2[1],asaxsf2[2])
        Ireconst=N+M*2*asaxsf1[0]+R*(asaxsf1[0]**2+asaxsf2[0]**2)
        if seqname is not None:
            datatosave=pylab.zeros((len(q),7))
            datatosave[:,0]=q;
            datatosave[:,1]=N.flatten();  datatosave[:,2]=DN.flatten();
            datatosave[:,3]=M.flatten();  datatosave[:,4]=DM.flatten();
            datatosave[:,5]=R.flatten();  datatosave[:,6]=DR.flatten();
            pylab.savetxt('%s_%s_basicfun.txt' % (seqname, s),datatosave,delimiter='\t')
            datatosave[:,1]=sep12.flatten(); datatosave[:,2]=dsep12.flatten();
            datatosave[:,3]=sep23.flatten(); datatosave[:,4]=dsep23.flatten();
            datatosave[:,5]=R1.flatten(); datatosave[:,6]=dR1.flatten();
            pylab.savetxt('%s_%s_separation.txt' % (seqname, s),datatosave,delimiter='\t')
        pylab.figure()
        #pylab.errorbar(q,Intensity[:,0],Errors[:,0],label='I_0',marker='.')
        #pylab.errorbar(q,N.flatten(),DN.flatten(),label='Nonresonant',marker='.')
        #pylab.errorbar(q,M.flatten(),DM.flatten(),label='Mixed',marker='.')
        #pylab.errorbar(q,R.flatten(),DR.flatten(),label='Resonant',marker='o')
        pylab.plot(q,Intensity[:,0],label='I_0',marker='.')
        pylab.plot(q,N.flatten(),label='Nonresonant',marker='.')
        pylab.plot(q,M.flatten(),label='Mixed',marker='.')
        pylab.plot(q,R.flatten(),label='Resonant',marker='o')
        pylab.plot(q,Ireconst.flatten(),label='I_0_reconstructed',marker='.')
        pylab.title("ASAXS basic functions for sample %s" % s)
        pylab.xlabel(u"q (1/%c)" % 197)
        pylab.ylabel("Scattering cross-section (1/cm)")
        pylab.gca().set_xscale('log');
        pylab.gca().set_yscale('log');
        pylab.legend();
        pylab.savefig('%s_%s_basicfun.eps'%(seqname,s),dpi=300,format='eps',transparent=True)
        pylab.figure()
        #pylab.errorbar(q,Intensity[:,0],Errors[:,0],label='I_0',marker='.')
        #pylab.errorbar(q,sep12,dsep12,label='(I_0-I_1)/(f1_0-f1_1)',marker='.')
        #pylab.errorbar(q,sep23,dsep23,label='(I_1-I_2)/(f1_1-f1_2)',marker='.')
        #pylab.errorbar(q,R1.flatten(),dR1.flatten(),label='Pure resonant',marker='.')
        pylab.plot(q,Intensity[:,0],label='I_0',marker='.')
        pylab.plot(q,sep12,label='(I_0-I_1)/(f1_0-f1_1)',marker='.')
        pylab.plot(q,sep23,label='(I_1-I_2)/(f1_1-f1_2)',marker='.')
        pylab.plot(q,R1.flatten(),label='Pure resonant',marker='.')
        
        pylab.title("ASAXS separated and pure resonant terms for sample %s" % s)
        pylab.xlabel(u"q (1/%c)" % 197)
        pylab.ylabel("Scattering cross-section (1/cm)")
        pylab.gca().set_xscale('log');
        pylab.gca().set_yscale('log');
        pylab.legend();
        pylab.savefig('%s_%s_separation.eps'%(seqname,s),dpi=300,format='eps',transparent=True)
    logfile.close()
    pylab.show()
