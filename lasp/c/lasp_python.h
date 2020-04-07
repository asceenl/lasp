// ascee_python.h
//
// Author: J.A. de Jong - ASCEE
//
// Description:
// Some routines to generate numpy arrays from matrices and vectors.
//////////////////////////////////////////////////////////////////////
#pragma once
#ifndef LASP_PYTHON_H
#define LASP_PYTHON_H
#define TRACERPLUS (-10)
#include <numpy/ndarrayobject.h>
#ifdef LASP_DOUBLE_PRECISION
#define LASP_NUMPY_FLOAT_TYPE NPY_FLOAT64
#define LASP_NUMPY_COMPLEX_TYPE NPY_COMPLEX128
#else
#define LASP_NUMPY_FLOAT_TYPE NPY_FLOAT32
#endif

#ifdef MS_WIN64
/** 
 * Function passed to Python to use for cleanup of
 * foreignly obtained data.
 **/
static inline void capsule_cleanup(void* capsule) {
	void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
	}

#endif
/** 
 * Create a numpy array from an existing dmat.
 *
 * @param mat dmat struccture containing array data and metadata.
 * @param transfer_ownership If set to true, Numpy array will be responsible
 * for freeing the data.
 *
 * @return Numpy array
 */
static inline PyObject* dmat_to_ndarray(dmat* mat,bool transfer_ownership) {
    fsTRACE(15);
    dbgassert(mat,NULLPTRDEREF);

    import_array();
    
    // Dimensions given in wrong order, as mat is
    // Fortran-contiguous. Later on we transpose the result. This is
    // more easy than using the PyArray_New syntax.
    npy_intp dims[] = {mat->n_cols,mat->n_rows};
    PyObject* arr_t = PyArray_SimpleNewFromData(2,dims,
                                              LASP_NUMPY_FLOAT_TYPE,
                                              mat->_data);
    if(!arr_t) {
        WARN("Array creation failure");
        feTRACE(15);
        return NULL;
    }

    if(transfer_ownership) {
        mat->_foreign_data = true;
#ifdef MS_WIN64
		// The default destructor of Python cannot free the data, as it is allocated
		// with malloc. Therefore, with this code, we tell Numpy/Python to use
		// the capsule_cleanup constructor. See:
		// https://stackoverflow.com/questions/54269956/crash-of-jupyter-due-to-the-use-of-pyarray-enableflags/54278170#54278170
		// Note that in general it was disadvised to build all C code with MinGW on Windows.
		// We do it anyway, see if we find any problems on the way.
		void* capsule = PyCapsule_New(mat->_data, NULL, capsule_cleanup);
		PyArray_SetBaseObject( arr_t, capsule);
#else		
        PyArray_ENABLEFLAGS(arr_t, NPY_OWNDATA);
#endif
    }

    // Transpose the array
    PyObject* arr = PyArray_Transpose((PyArrayObject*) arr_t,NULL);
    if(!arr) {
        WARN("Array transpose failure");
        feTRACE(15);
        return NULL;
    }
    Py_DECREF(arr_t);
 
    feTRACE(15);
    return arr;
}


#endif // LASP_PYTHON_H
//////////////////////////////////////////////////////////////////////
