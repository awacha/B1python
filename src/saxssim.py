from c_saxssim import *

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

def spheres2povray(file,spheredata,targetx=0,targety=0,targetz=0,
                   backgroundcolor="White",lightcolor="White",show_axes=True,
                   axislen=None,axiswid=None,conelen=None,conewid=None,
                   lightinfty=None,rcamera=None,phicamera=0,thetacamera=0,
                   create_movie=False):
    if type(file)==type(''):
        fileopened=True
        file=open(file,"w+")
    else:
        fileopened=False
    
    if rcamera is None:
        rcamera=maxdistance(spheredata)
    if axislen is None:
        axislen=rcamera
    if axiswid is None:
        axiswid=spheredata[:,4].mean()/10.0
    if conelen is None:
        conelen=axislen/20.0
    if conewid is None:
        conewid=axiswid*1.5
    if lightinfty is None:
        lightinfty=rcamera*3

    file.write("#include \"colors.inc\"\n")
    file.write("#declare ThetaCamera=%f;\n" % thetacamera)
    if create_movie:
        file.write("#declare PhiCamera=%f+clock;\n"%phicamera)
    else:
        file.write("#declare PhiCamera=%f;\n"%phicamera)
    file.write("#declare RCamera=%f;\n" % rcamera)
    file.write("#declare TargetX=%f;\n" % targetx)
    file.write("#declare TargetY=%f;\n" % targety)
    file.write("#declare TargetZ=%f;\n" % targety)
    if show_axes:
        file.write("#declare Axislen=%f;\n"%axislen)
        file.write("#declare Axiswid=%f;\n"%axiswid)
        file.write("#declare Conelen=%f;\n"%conelen)
        file.write("#declare Conewid=%f;\n"%conewid)
    file.write("#declare Lightinfty=%f;\n"%lightinfty)
    file.write("#declare Lightcolor=%s;\n"%lightcolor)
    file.write("background { color %s }\n"%backgroundcolor)
    file.write("camera {\n");
    file.write("  orthographic\n")
    file.write("  location <RCamera*sin(ThetaCamera)*cos(PhiCamera),\n")
    file.write("            RCamera*cos(ThetaCamera),\n")
    file.write("            RCamera*sin(ThetaCamera)*sin(PhiCamera)>\n")
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
            file.write("    <%f, %f, %f>, <%f, %f, %f>, %f\n" % (spheredata[i,0]-0.5*spheredata[i,5]*spheredata[i,6],
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
            file.write("    <%f, %f, %f>, %f\n" % (spheredata[i,0],spheredata[i,1],spheredata[i,2],spheredata[i,3]))
            file.write("    texture {\n")
            file.write("      pigment { color Blue }\n")
            file.write("    }\n")
            file.write("  }\n")
    file.write('}\n')
    if fileopened:
        file.close()