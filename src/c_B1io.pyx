import numpy as np
cimport numpy as np

def cbfdecompress(data, Py_ssize_t dim1, Py_ssize_t dim2):
    cdef Py_ssize_t index_input
    cdef Py_ssize_t index_output
    cdef double value_current
    cdef double value_diff
    cdef Py_ssize_t nbytes
    cdef np.ndarray[ndim=1,dtype=np.double_t] outarray
    cdef Py_ssize_t npixels
    
    index_input=0
    index_output=0
    value_current=0
    value_diff=0
    nbytes=len(data)
    npixels=dim1*dim2
    outarray=np.zeros(npixels,dtype=np.double)
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
                    value_diff=value_diff-4294967296.0
                index_input+=4
        value_current+=value_diff
        if index_output<dim1*dim2:
            outarray[index_output]=value_current
        else:
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
    return outarray.reshape((dim2,dim1))
