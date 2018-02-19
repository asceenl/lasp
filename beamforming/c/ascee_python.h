// ascee_python.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Some routines to generate numpy arrays from matrices and vectors.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef ASCEE_PYTHON_H
#define ASCEE_PYTHON_H
#include <numpy/ndarrayobject.h>
#ifdef ASCEE_DOUBLE_PRECISION
#define ASCEE_NUMPY_FLOAT_TYPE NPY_FLOAT64
#define ASCEE_NUMPY_COMPLEX_TYPE NPY_COMPLEX128
#else
#define ASCEE_NUMPY_FLOAT_TYPE NPY_FLOAT32
#endif


static inline PyObject* dmat_to_ndarray(dmat* mat,bool transfer_ownership) {
    fsTRACE(15);
    dbgassert(mat,NULLPTRDEREF);

    import_array();
    
    // Dimensions given in wrong order, as mat is
    // Fortran-contiguous. Later on we transpose the result. This is
    // more easy than using the PyArray_New syntax.
    npy_intp dims[] = {mat->n_cols,mat->n_rows};
    PyObject* arr_t = PyArray_SimpleNewFromData(2,dims,
                                              ASCEE_NUMPY_FLOAT_TYPE,
                                              mat->_data);
    if(!arr_t) {
        WARN("Array creation failure");
        feTRACE(15);
        return NULL;
    }

    if(transfer_ownership) {
        mat->_foreign_data = true;
        PyArray_ENABLEFLAGS(arr_t, NPY_OWNDATA);
    }

    // Transpose the array
    PyObject* arr = PyArray_Transpose(arr_t,NULL);
    if(!arr) {
        WARN("Array transpose failure");
        feTRACE(15);
        return NULL;
    }
    Py_DECREF(arr_t);

                                              
    feTRACE(15);
    return arr;
}


#endif // ASCEE_PYTHON_H
//////////////////////////////////////////////////////////////////////
