import os
import pickle
funcs=[]

files=os.walk('src').next()[2]

files=[f for f in files if (f[-3:]=='.py' or f[-4:]=='.pyx')]
#print 
for fn in files:
    f=open(os.path.join('src',fn),'rt')
#    print "processing file %s"%fn
    x=f.readline()
    infunction=False
    while len(x)>0:
        if x[:3]=='def':
            funcname=x.strip()[4:].split('(')[0].strip()
            funcs.append({'name':funcname,'definedin':fn,'req':[],'type':'PY'})
        elif x[:4]=='cdef':
            funcname=x.strip()[5:].split('(')[0].strip()
            if funcname[:6]=='extern':
                break
            if funcname[:6]=='inline':
                funcname=funcname.split()[-1]
            funcs.append({'name':funcname,'definedin':fn,'req':[],'type':'CY'})
        else:
#            print "Nofun: x=%s"%x
            pass
        x=f.readline()
    f.close()

for fn in files:
    f=open(os.path.join('src',fn),'rt')
    infunction=None
    x=f.readline()
    stringmode=False
#    print "Processing file: ",fn
    while len(x)>0:
#        print "Current line (INFUNC: %s):"%infunction, x[:-1]
        if len(x.strip())<=0:
            x=f.readline()
            continue
        if x.strip()[0]=='#':
            pass
        else:
            for i in xrange(x.count('"""')):
                stringmode=not stringmode
            if stringmode:
 #               print "STRING: %s" % x[:-1]
                pass
            elif x[:3]=='def':
                funcname=x.strip()[4:].split('(')[0].strip()
                infunction=funcname
            elif x[:4]=='cdef':
                funcname=x.strip()[5:].split('(')[0].strip()
                if funcname[:6]=='extern':
                    x=f.readline()
                    continue
                elif funcname[:6]=='inline':
                    funcname=funcname.split()[-1]
                infunction=funcname
            else:
                for i  in xrange(len(funcs)):
                    if x.count(funcs[i]['name'])>0:
#                        print "Found dep (%s): %s" % (funcs[i]['name'],x)
                        funcs[i]['req'].append(infunction)
        x=f.readline()
    f.close()

#print "Functions: ",[f['name'] for f in funcs]
for fun in funcs:
    fun['dep']=[f['name'] for f in funcs if fun['name'] in f['req']]
    print "Function: %s" % fun['name']
    print "  Defined in: %s" % fun['definedin']
    print "  Type: %s" % fun['type']
    print "  Required by:",fun['req']
    print "  Depends on:",fun['dep']
    print ""

f=open('finddeps.pickle',"wt")
pickle.dump(funcs,f)
f.close()

#print [f for f in funcs if f['name']=='trimq']
#print [f for f in funcs if f['name']=='powerfit']

