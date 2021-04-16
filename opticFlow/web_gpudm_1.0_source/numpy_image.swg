// -*- c++ -*- 
/*
Copyright (C) 2015 Jerome Revaud

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
*/

%{
#include "numpy_image.h"
%}


%{
#define SWIG_FILE_WITH_INIT

#include <numpy/arrayobject.h>

#define CHECK_NUMPY_ARRAY(expected_npy)                                 \
  if(!a) {                                                              \
    fprintf(stderr,"error in %s(): NULL input\n",__PRETTY_FUNCTION__);             \
    return NULL;                                                        \
  }                                                                     \
  if(!PyArray_Check(a)) {                                               \
    fprintf(stderr,"error in %s(): input not numpy array\n",__PRETTY_FUNCTION__);  \
    return NULL;                                                        \
  }                                                                     \
  if(!PyArray_ISCONTIGUOUS((PyArrayObject*)a)) {                                        \
    fprintf(stderr,"error in %s(): array is not C-contiguous\n",__PRETTY_FUNCTION__);  \
    return NULL;                                                        \
  }                                                                     \
  if(PyArray_TYPE((PyArrayObject*)a)!=expected_npy) {                                   \
    fprintf(stderr,"error in %s(): input has bad type (type id %d != " #expected_npy " %d)\n",__PRETTY_FUNCTION__, \
            PyArray_TYPE((PyArrayObject*)a),expected_npy);                              \
    return NULL;                                                        \
  }

// append to current function result as a tuple
PyObject* APPEND_TUPLE(PyObject* o, PyObject* result) {
  PyObject *o2, *o3;
  
  if ((!result) || (result == Py_None)) {
      result = o;
  } else {
      if (!PyTuple_Check(result)) {
          o2 = result;
          result = PyTuple_New(1);
          PyTuple_SetItem(result,0,o2); // SetItem steals a ref
      }
      o3 = PyTuple_New(1);
      PyTuple_SetItem(o3,0,o); // SetItem steals a ref
      o2 = result;
      result = PySequence_Concat(o2,o3);
      Py_CLEAR(o2);
      Py_CLEAR(o3);
  }
  
  return result;
}

%}

%init %{
/* needed, else crash at runtime */
    import_array();
%}


/* ~~~~ 1D arrays ~~~~~~~~~~~~~~~ */

%define TYPEMAP_ARRAY_1D(type,NPY_TYPE)
%typemap(in) 
  (type ## _array* arr) 
  (type ## _array array) {
  
  PyObject* a = $input;
  if(a==Py_None)
    $1 = NULL;
  else {
    CHECK_NUMPY_ARRAY(NPY_TYPE)
    array.pixels = (type*) PyArray_DATA((PyArrayObject*)a);
    
    a = PyObject_GetAttrString($input,"shape");
    assert(PyTuple_Size(a)==1);
    array.tx = PyInt_AsLong(PyTuple_GetItem(a,0));
    Py_CLEAR(a);
    
    $1=&array;
  }
}

%apply (type##_array* arr) {(type##_array* )};
%enddef

TYPEMAP_ARRAY_1D(int,NPY_INT)
TYPEMAP_ARRAY_1D(float,NPY_FLOAT)


/* ~~~~ 2D images ~~~~~~~~~~~~~~~ */

%define TYPEMAP_IMAGE(type,NPY_TYPE)
%typemap(in) 
  (type ## _image* img) 
  (type ## _image image) {
  
  PyObject* a = $input;
  if(a==Py_None)
    $1 = NULL;
  else {
    CHECK_NUMPY_ARRAY(NPY_TYPE)
    image.pixels = (type*) PyArray_DATA((PyArrayObject*)a);
    
    a = PyObject_GetAttrString($input,"shape");
    assert(PyTuple_Size(a)==2);
    image.ty = PyInt_AsLong(PyTuple_GetItem(a,0));
    image.tx = PyInt_AsLong(PyTuple_GetItem(a,1));
    Py_CLEAR(a);
    
    $1=&image;
  }
}

// special case where we want to output an image

%apply (type##_image* img) {(type##_image* )};
%enddef

TYPEMAP_IMAGE(int, NPY_INT)
TYPEMAP_IMAGE(float, NPY_FLOAT)


/* ~~~~ Cubes = 3D images ~~~~~~~~~~~~~~~ */

%define TYPEMAP_CUBE_3D(type,NPY_TYPE)
%typemap(in) 
  (type ## _cube* cube) 
  (type ## _cube cube) {
  
  PyObject* a = $input;
  if(a==Py_None)
    $1 = NULL;
  else {
    CHECK_NUMPY_ARRAY(NPY_TYPE)
    cube.pixels = (type*) PyArray_DATA((PyArrayObject*)a);
    
    a = PyObject_GetAttrString($input,"shape");
    assert(PyTuple_Size(a)==3);
    cube.ty = PyInt_AsLong(PyTuple_GetItem(a,0));
    cube.tx = PyInt_AsLong(PyTuple_GetItem(a,1));
    cube.tz = PyInt_AsLong(PyTuple_GetItem(a,2));
    Py_CLEAR(a);
    
    $1=&cube;
  }
}

// special case where we want to output something

%apply (type##_cube* cube) {(type##_cube* )};
%enddef

TYPEMAP_CUBE_3D(int, NPY_INT)
TYPEMAP_CUBE_3D(float, NPY_FLOAT)





















