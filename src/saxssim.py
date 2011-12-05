from c_saxssim import theorsaxs2D,theorsaxs2D_fast,Ctheorsphere2D,Ctheorspheregas,Ctheorspheres,maxdistance,unidir,grf_saxs2D,grf_realize,ddistcylinder,ddistsphere,ddistellipsoid,ddistbrick,ftddist,ddistgrf,charfuncgrf,packspheres,structurefactor,ellipsoid_scatter,Ctheorspheres_azim,CtheorchainPBC
import numpy as np

def grf_savevtk(file,grfmatrix,origin=None,spacing=None):
    if type(file)==type(''):
        fileopened=True
        file=open(file,"w+")
    else:
        fileopened=False
    if origin is None:
        origin=[0,0,0]
    try:
        len(origin)
    except:
        origin=[origin]*3
    if len(origin)!=3:
        raise ValueError,"origin should be None or a scalar or an array with length==3"
    if spacing is None:
        spacing=[1,1,1]
    try:
        len(spacing)
    except:
        spacing=[spacing]*3
    if len(spacing)!=3:
        raise ValueError,"spacing should be None or a scalar or an array with length==3"
    file.write("# vtk DataFile Version 3.0\n")
    file.write("created by grf.py by awacha\n")
    file.write("ASCII\n\n")
    file.write("DATASET STRUCTURED_POINTS\n")
    file.write("DIMENSIONS %lu %lu %lu\n" % grfmatrix.shape)
    file.write("ORIGIN %lf %lf %lf\n" % tuple(origin))
    file.write("SPACING %lf %lf %lf\n\n" % tuple(spacing))
    file.write("POINT_DATA %lu\n" % grfmatrix.size)
    file.write("SCALARS grfdata float\n")
    file.write("LOOKUP_TABLE default\n")
    np.savetxt(file,grfmatrix.flatten())
    if fileopened:
        file.close()

def spheres2povray(file,spheredata,targetx=None,targety=None,targetz=None,
                   backgroundcolor="White",lightcolor="White",show_axes=True,
                   axislen=None,axiswid=None,conelen=None,conewid=None,
                   lightinfty=None,rcamera=None,phicamera=0,thetacamera=0,
                   create_movie=False, boxsize=None, box1=None, box2=None):
    if type(file)==type(''):
        fileopened=True
        file=open(file,"w+")
    else:
        fileopened=False
    if targetx is None:
        targetx=np.mean(spheredata[:,0]);
    if targety is None:
        targety=np.mean(spheredata[:,1]);
    if targetz is None:
        targetz=np.mean(spheredata[:,2]);
    if rcamera is None:
        rcamera=maxdistance(spheredata)
    if axislen is None:
        axislen=rcamera
    if axiswid is None:
        axiswid=spheredata[:,3].mean()/3.0
    if conelen is None:
        conelen=axislen/20.0
    if conewid is None:
        conewid=axiswid*1.5
    if lightinfty is None:
        lightinfty=rcamera*3
    if boxsize is not None:
        if not isinstance(boxsize,list) and not isinstance(boxsize,tuple):
            boxsize=tuple([boxsize]*3);
        if box1 is None:
            box1=(0,0,0)
        if box2 is None:
            box2=boxsize;
    file.write("#include \"colors.inc\"\n")
    file.write("#declare ThetaCamera=%g;\n" % thetacamera)
    if create_movie:
        file.write("#declare PhiCamera=%g+clock;\n"%phicamera)
    else:
        file.write("#declare PhiCamera=%g;\n"%phicamera)
    file.write("#declare RCamera=%g;\n" % rcamera)
    file.write("#declare TargetX=%g;\n" % targetx)
    file.write("#declare TargetY=%g;\n" % targety)
    file.write("#declare TargetZ=%g;\n" % targetz)
    if show_axes:
        file.write("#declare Axislen=%g;\n"%axislen)
        file.write("#declare Axiswid=%g;\n"%axiswid)
        file.write("#declare Conelen=%g;\n"%conelen)
        file.write("#declare Conewid=%g;\n"%conewid)
    file.write("#declare Lightinfty=%g;\n"%lightinfty)
    file.write("#declare Lightcolor=%s;\n"%lightcolor)
    file.write("background { color %s }\n"%backgroundcolor)
    file.write("camera {\n");
    file.write("  orthographic\n")
    file.write("  location <TargetX+RCamera*sin(ThetaCamera)*cos(PhiCamera),\n")
    file.write("            TargetZ+RCamera*cos(ThetaCamera),\n")
    file.write("            TargetY+RCamera*sin(ThetaCamera)*sin(PhiCamera)>\n")
    file.write("  look_at <TargetX, TargetZ, TargetY>\n")
    file.write("}\n")
    if show_axes:
        file.write("union {\n")
        file.write("  cylinder {\n")
        file.write("    <0, 0, 0>, <Axislen , 0 , 0>, Axiswid\n")
        file.write("  }\n")
        file.write("  cone {\n")
        file.write("    <Axislen ,0,0>, Conewid, <Axislen+Conelen ,0, 0>,0\n")
        file.write("  }\n")
        file.write("  texture { pigment { color Green } }\n")
        file.write("}\n")
        
        file.write("union {\n")
        file.write("  cylinder {\n")
        file.write("    <0, 0, 0>, <0, Axislen , 0>, Axiswid\n")
        file.write("  }\n")
        file.write("  cone {\n")
        file.write("    <0, Axislen ,0>, Conewid, <0, Axislen+Conelen ,0>,0\n")
        file.write("  }\n")
        file.write("  texture { pigment { color Red } }\n")
        file.write("}\n")
        
        file.write("union {\n")
        file.write("  cylinder {\n")
        file.write("    <0, 0, 0>, <0, 0, Axislen>, Axiswid\n")
        file.write("  }\n")
        file.write("  cone {\n")
        file.write("    <0, 0, Axislen>, Conewid, <0,0, Axislen+Conelen>,0\n")
        file.write("  }\n")
        file.write("  texture { pigment { color Cyan } }\n")
        file.write("}\n")
    if boxsize is not None:
        file.write("union {\n")
        
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box1[0],box1[1],box1[2],box1[0],box1[1],box2[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box1[0],box1[1],box1[2],box1[0],box2[1],box1[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box1[0],box1[1],box1[2],box2[0],box1[1],box1[2]))
        file.write("  }\n")

        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box2[0],box2[1],box1[2],box2[0],box1[1],box1[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box2[0],box2[1],box1[2],box2[0],box2[1],box2[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box2[0],box2[1],box1[2],box1[0],box2[1],box1[2]))
        file.write("  }\n")

        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box2[0],box1[1],box2[2],box2[0],box1[1],box1[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box2[0],box1[1],box2[2],box1[0],box1[1],box2[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box2[0],box1[1],box2[2],box2[0],box2[1],box2[2]))
        file.write("  }\n")

        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box1[0],box2[1],box2[2],box2[0],box2[1],box2[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box1[0],box2[1],box2[2],box1[0],box2[1],box1[2]))
        file.write("  }\n")
        file.write("  cylinder {\n")
        file.write("    <%lg, %lg, %lg>, <%lg, %lg, %lg>, Axiswid\n"%(
                       box1[0],box2[1],box2[2],box1[0],box1[1],box2[2]))
        file.write("  }\n")
        file.write("  texture { pigment { color Black } }\n")
        file.write("}\n")
        
    file.write("light_source { < -Lightinfty, 0, 0 > color Lightcolor }\n")
    file.write("light_source { <0, -Lightinfty, 0 > color Lightcolor }\n")
    file.write("light_source { <0,0, -Lightinfty > color Lightcolor }\n")
    file.write("light_source { < Lightinfty, 0, 0 > color Lightcolor }\n")
    file.write("light_source { <0, Lightinfty, 0 > color Lightcolor }\n")
    file.write("light_source { <0,0, Lightinfty > color Lightcolor }\n")
    
    file.write("union {\n")
    for i in range(spheredata.shape[0]):
        if spheredata.shape[1]>5 and spheredata[i,5]>0:
            file.write("  cylinder {\n")
            file.write("    <%g, %g, %g>, <%g, %g, %g>, %g\n" % (spheredata[i,0]-0.5*spheredata[i,5]*spheredata[i,6],
                                                                 spheredata[i,1]-0.5*spheredata[i,5]*spheredata[i,7],
                                                                 spheredata[i,2]-0.5*spheredata[i,5]*spheredata[i,8],
                                                                 spheredata[i,0]+0.5*spheredata[i,5]*spheredata[i,6],
                                                                 spheredata[i,1]+0.5*spheredata[i,5]*spheredata[i,7],
                                                                 spheredata[i,2]+0.5*spheredata[i,5]*spheredata[i,8],
                                                                 spheredata[i,3]))
            file.write("    texture {\n")
            file.write("      pigment { color Blue }\n")
            file.write("    }\n")
            file.write("  }\n")
        else:
            file.write("  sphere {\n")
            file.write("    <%g, %g, %g>, %g\n" % (spheredata[i,0],spheredata[i,1],spheredata[i,2],spheredata[i,3]))
            file.write("    texture {\n")
            file.write("      pigment { color Blue }\n")
            file.write("    }\n")
            file.write("  }\n")
    file.write('}\n')
    if fileopened:
        file.close()
