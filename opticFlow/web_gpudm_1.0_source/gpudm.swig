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

%module gpudm;

%{

#include "caffe/caffe.hpp"
#include "extra_layers.hpp"

using namespace caffe; 
using namespace std; 

%}

%include "std_string.i"
%include "std_pair.i"
%include "std_set.i"
%include "std_map.i"
%include "std_vector.i"
%include "boost_shared_ptr.i"


%shared_ptr(caffe::Blob< float >); 
%shared_ptr(caffe::Layer< float >); 
%shared_ptr(caffe::NeuronLayer<float>);
%shared_ptr(caffe::ReLULayer< float >); 
%shared_ptr(caffe::PowerLayer< float >); 
%shared_ptr(caffe::RectifiedSigmoidLayer< float >); 
%shared_ptr(caffe::ReshapeLayer< float >); 
%shared_ptr(caffe::PixelNormLayer< float >);
%shared_ptr(caffe::BaseConvolutionLayer< float >); 
%shared_ptr(caffe::ConvolutionLayer< float >); 
%shared_ptr(caffe::PatchConvolutionLayer<float>);
%shared_ptr(caffe::CSR_SparseConvolutionLayer<float>);
%shared_ptr(caffe::BorderRectifyLayer< float >);
%shared_ptr(caffe::PoolingLayer<float>); 
%shared_ptr(caffe::InnerProductLayer<float>); 
%shared_ptr(caffe::DeepMatchingArgMaxLayer<float>); 


%include "caffe/caffe.hpp"
%include "caffe/common.hpp"
%include "caffe/blob.hpp"
%include "caffe/filler.hpp"
%include "caffe/layer.hpp"

using namespace std;
using namespace boost;
%template(BlobVector) vector< shared_ptr< caffe::Blob< float > > >; 
%template(BlobPtrVector) vector< caffe::Blob< float >* >; 
%template(LayerVector) vector< shared_ptr< caffe::Layer< float > > >; 
%template(FloatVector) vector< float >; 

%include "caffe/net.hpp"
//%include "caffe/common_layers.hpp"
//%include "caffe/neuron_layers.hpp"
//%include "caffe/vision_layers.hpp"

%include "caffe/layers/absval_layer.hpp"
%include "caffe/layers/accuracy_layer.hpp"
%include "caffe/layers/argmax_layer.hpp"
%include "caffe/layers/base_conv_layer.hpp"
%include "caffe/layers/base_data_layer.hpp"
%include "caffe/layers/batch_norm_layer.hpp"
%include "caffe/layers/batch_reindex_layer.hpp"
%include "caffe/layers/bias_layer.hpp"
%include "caffe/layers/bnll_layer.hpp"
//%include "caffe/layers/clip_layer.hpp"
%include "caffe/layers/concat_layer.hpp"
%include "caffe/layers/contrastive_loss_layer.hpp"
%include "caffe/layers/conv_layer.hpp"
%include "caffe/layers/crop_layer.hpp"
%include "caffe/layers/cudnn_conv_layer.hpp"
//%include "caffe/layers/cudnn_deconv_layer.hpp"
%include "caffe/layers/cudnn_lcn_layer.hpp"
%include "caffe/layers/cudnn_lrn_layer.hpp"
%include "caffe/layers/cudnn_pooling_layer.hpp"
%include "caffe/layers/cudnn_relu_layer.hpp"
%include "caffe/layers/cudnn_sigmoid_layer.hpp"
%include "caffe/layers/cudnn_softmax_layer.hpp"
%include "caffe/layers/cudnn_tanh_layer.hpp"
%include "caffe/layers/data_layer.hpp"
%include "caffe/layers/deconv_layer.hpp"
%include "caffe/layers/dropout_layer.hpp"
%include "caffe/layers/dummy_data_layer.hpp"
%include "caffe/layers/eltwise_layer.hpp"
%include "caffe/layers/elu_layer.hpp"
%include "caffe/layers/embed_layer.hpp"
%include "caffe/layers/euclidean_loss_layer.hpp"
%include "caffe/layers/exp_layer.hpp"
%include "caffe/layers/filter_layer.hpp"
%include "caffe/layers/flatten_layer.hpp"
%include "caffe/layers/hdf5_data_layer.hpp"
%include "caffe/layers/hdf5_output_layer.hpp"
%include "caffe/layers/hinge_loss_layer.hpp"
%include "caffe/layers/im2col_layer.hpp"
%include "caffe/layers/image_data_layer.hpp"
%include "caffe/layers/infogain_loss_layer.hpp"
%include "caffe/layers/inner_product_layer.hpp"
%include "caffe/layers/input_layer.hpp"
%include "caffe/layers/log_layer.hpp"
%include "caffe/layers/loss_layer.hpp"
%include "caffe/layers/lrn_layer.hpp"
%include "caffe/layers/lstm_layer.hpp"
%include "caffe/layers/memory_data_layer.hpp"
%include "caffe/layers/multinomial_logistic_loss_layer.hpp"
%include "caffe/layers/mvn_layer.hpp"
%include "caffe/layers/neuron_layer.hpp"
%include "caffe/layers/parameter_layer.hpp"
%include "caffe/layers/pooling_layer.hpp"
%include "caffe/layers/power_layer.hpp"
%include "caffe/layers/prelu_layer.hpp"
%include "caffe/layers/python_layer.hpp"
%include "caffe/layers/recurrent_layer.hpp"
%include "caffe/layers/reduction_layer.hpp"
%include "caffe/layers/relu_layer.hpp"
%include "caffe/layers/reshape_layer.hpp"
%include "caffe/layers/rnn_layer.hpp"
%include "caffe/layers/scale_layer.hpp"
%include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
%include "caffe/layers/sigmoid_layer.hpp"
%include "caffe/layers/silence_layer.hpp"
%include "caffe/layers/slice_layer.hpp"
%include "caffe/layers/softmax_layer.hpp"
%include "caffe/layers/softmax_loss_layer.hpp"
%include "caffe/layers/split_layer.hpp"
%include "caffe/layers/spp_layer.hpp"
//%include "caffe/layers/swish_layer.hpp"
%include "caffe/layers/tanh_layer.hpp"
%include "caffe/layers/threshold_layer.hpp"
%include "caffe/layers/tile_layer.hpp"
%include "caffe/layers/window_data_layer.hpp"

%include "caffe/util/math_functions.hpp"
%include "extra_layers.hpp"

#define GOOGLE_PROTOBUF_VERSION 3000000
#define GOOGLE_PROTOBUF_MIN_PROTOC_VERSION 3000000

namespace google {
  namespace protobuf {
    class Message {
    private: 
      Message();
      int x;
    };
    typedef unsigned int uint32;
    typedef int int32;
    typedef long long int64;
  }
}

%include "caffe/proto/caffe.pb.h"


%template(NetFloat) caffe::Net<float>; 
%template(BlobFloat) caffe::Blob<float>; 
%template(LayerFloat) caffe::Layer<float>;
%template(NeuronLayerFloat) caffe::NeuronLayer<float>;
%template(ReLULayerFloat) caffe::ReLULayer<float>;
%template(PowerLayerFloat) caffe::PowerLayer<float>;
%template(ReshapeLayerFloat) caffe::ReshapeLayer< float >; 
%template(RectifiedSigmoidLayerFloat) caffe::RectifiedSigmoidLayer<float>;
%template(PixelNormLayerFloat) caffe::PixelNormLayer< float >;
%template(BaseConvolutionLayerFloat) caffe::BaseConvolutionLayer< float >; 
%template(ConvolutionLayerFloat) caffe::ConvolutionLayer< float >; 
%template(PatchConvolutionLayerFloat) caffe::PatchConvolutionLayer<float>;
%template(CSR_SparseConvolutionLayerFloat) caffe::CSR_SparseConvolutionLayer<float>;
%template(BorderRectifyLayerFloat) caffe::BorderRectifyLayer< float >;
%template(PoolingLayerFloat) caffe::PoolingLayer<float>; 
%template(DeepMatchingArgMaxLayerFloat) caffe::DeepMatchingArgMaxLayer<float>; 

// function releases GIL while doing slow action
%exception {
  Py_BEGIN_ALLOW_THREADS
  $action
  Py_END_ALLOW_THREADS
}

%exception;


%{
#define SWIG_FILE_WITH_INIT
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

PyObject * floats_to_numpy_ref(float *src, long size) {
  npy_intp shp = size; 
  return PyArray_SimpleNewFromData(1, &shp, NPY_FLOAT32, src);
}

%}

PyObject * floats_to_numpy_ref(float *src, long size);

%init %{
/* needed, else crash at runtime */
    import_array();
%}


%pythoncode %{

import numpy

def BlobFloat_get_shape(self): 
  return (self.num(), self.channels(), self.height(), self.width())

def BlobFloat_to_numpy_ref(self): 
  b = floats_to_numpy_ref(self.cpu_data(), self.count())
  return b.reshape(self.get_shape())

def BlobFloat_diff_to_numpy_ref(self): 
  b = floats_to_numpy_ref(self.cpu_diff(), self.count())
  return b.reshape(self.get_shape())

def BlobFloat_mutable_to_numpy_ref(self): 
  b = floats_to_numpy_ref(self.mutable_cpu_data(), self.count())
  return b.reshape(self.get_shape())

def BlobFloat_mutable_diff_to_numpy_ref(self): 
  b = floats_to_numpy_ref(self.mutable_cpu_diff(), self.count())
  return b.reshape(self.get_shape())

BlobFloat.get_shape = BlobFloat_get_shape
BlobFloat.to_numpy_ref = BlobFloat_to_numpy_ref
BlobFloat.diff_to_numpy_ref = BlobFloat_diff_to_numpy_ref
BlobFloat.mutable_to_numpy_ref = BlobFloat_mutable_to_numpy_ref
BlobFloat.mutable_diff_to_numpy_ref = BlobFloat_mutable_diff_to_numpy_ref

%}


/*-------- DeepMatching functions ------------*/
%{

#include "numpy_image.h"
#include <algorithm>
using std::min;
using std::max;
static inline float pow2(float x) {return x*x;}
#define nullptr 0

static inline int retrieve_children( const int x, const int y, const int_cube* child_grid ) {
  const int size0_div2 = child_grid->pixels[0];
  const int step0 = child_grid->tx==1 && child_grid->ty==1 ? 1 : 
                                        max( child_grid->pixels[2]-child_grid->pixels[0], 
                                             child_grid->pixels[1+2*child_grid->tx]-child_grid->pixels[1] );
  int i = (x-size0_div2)/step0;
  int j = (y-size0_div2)/step0;
  assert( x==(i*step0+size0_div2) || !"error: child_grid does not match current grid" );
  assert( y==(j*step0+size0_div2) || !"error: child_grid does not match current grid" );
  if( i<0 || i>=child_grid->tx )  return -1;
  if( j<0 || j>=child_grid->ty )  return -1;
  return i+j*child_grid->tx;
}

static inline void prepare_gaps( const int parent_psize, const int nc, int gaps[3] ) {
  const int hs = parent_psize/2;
  if(nc==2) { // 4 children per parent patch
    gaps[0] = hs/2 - hs;
    gaps[1] = hs/2;
  } else if(nc==3) {  // 9 children per parent patch
    gaps[0] = hs/2 - hs;
    gaps[1] = 0;
    gaps[2] = hs/2;
  } else assert(0);
}

/* Prepare a grid of cell positions in the first image for a given scale. Big cells inherit the cell at the previous scale.
    size = size of cells at current scale
    offset, step = grid generator: (offset + i*step, offset + j*step)
    child_grid = grid of the previous layer (or None if first layer)
    child_norms = image containing the norms of the patch at the previous level
    grid = result center positions of cells in current scale
    children = index of cells in previous scale used to construct big cells
    norms = norms of the cells of this level
*/
void _prepare_big_cells( int size, int offset, int step, 
                         int_cube* child_grid, float_image* child_norms,
                         int_cube* grid, int_cube* children, float_image* norms ) {
  assert(grid->tz==2);
  const int ntx = grid->tx; // should be == 1+(tx-size)/step so that patches do not pass the border
  const int nty = grid->ty; // should be == 1+(ty-size)/step so that patches do not pass the border
  
  /* grid[i,j] = ( offset + i*step, offset + j*step )
    
    connection between two scales:
    x cell position in lower scale == x position of children in upper scale
    child_offset + child_i*child_step = offset + i*step + U*size/4
                                                     with U = (2*u/(nc-1)-1) \in {-1,0,1}
  */
  
  int i,j,u,v;
  int* r = grid->pixels;
  
  if( !child_grid ) {
    // this is the first scale: 
    // we just return a grid of step size*(1-overlap/2) in [0, tx[ x [0, ty[
    
    for(j=0; j<nty; j++)
      for(i=0; i<ntx; i++) {
        *r++ = offset + i*step;
        *r++ = offset + j*step;
      }
  } else {
    assert(child_grid->tz==2);
    assert( (child_norms!=NULL) == (norms!=NULL) || !"both must be null or non-null at the same time" );
    if(norms) ASSERT_SAME_SIZE( child_grid, child_norms );
    assert( children );
    const int nc = sqrt(children->tz); // number of children per row or col
    assert( children->tz==pow2(nc) );
    ASSERT_SAME_SIZE( grid, children );
    if(norms) ASSERT_SAME_SIZE( grid, norms );
    // this is at least second scale
    // we return a grid of step size*(1-overlap/2) in [0, tx[ x [0, ty[
    
    int gaps[3];
    prepare_gaps( size, nc, gaps ); // usually, returns [-q,q] for nc==2 with q=size/4
    
    int* c = children->pixels; 
    float *n = norms ? norms->pixels : nullptr;
    if(n) memset(n,0,ntx*nty*sizeof(float));
    for(j=0; j<nty; j++)
      for(i=0; i<ntx; i++) {
        int x = offset + i*step;
        int y = offset + j*step;
        *r++ = x;
        *r++ = y;
        
        // accumulate norms from 2x2 or 3x3 neighbors        
        for(v=0; v<nc; v++)
          for(u=0; u<nc; u++,c++) {
            // we want to index the children at position:
            // ( center_x + gaps[u], center_y + gaps[v] )
            *c = retrieve_children( x+gaps[u], y+gaps[v], child_grid );
            if(n && *c>=0) *n += child_norms->pixels[*c];
          }
        if(n) n++;
      }
  }
}

#define NEWA(type,n) (type*)malloc(sizeof(type)*(n))

static float** get_list_corres( const float_cube* map, int* nb ) {
  const int tz = map->tz;
  float* m = map->pixels;
  const long npix = map->tx*map->ty;
  float** res = NEWA(float*,npix);
  
  int i,n=0;
  for(i=0; i<npix; i++,m+=tz)
    if(m[4]) { // if score non-null
      res[n++] = m; // remember pointer
    }
  
  *nb = n;
  return res;
}

static inline int cmp_corres( const void* a, const void* b) {
  return memcmp(*(float**)a,*(float**)b,4*sizeof(float));
}

/* Intersect 2 mappings: erase all correspondences that are not reciprocal 
*/
int _intersect_corres( const float_cube* map0, const float_cube* map1, float_image* corres ) {
  const int tz = 6;
  assert( map0->tz==tz && map1->tz==tz );
  
  // build the list of triplets
  int n0,n1;
  float** const corres0 = get_list_corres(map0,&n0);
  float** const corres1 = get_list_corres(map1,&n1);
  
  // arg-sort the lists
  qsort( corres0, n0, sizeof(float*), cmp_corres );
  qsort( corres1, n1, sizeof(float*), cmp_corres );
  
  // remove all correspondences from map0/map1 that is not shared
  float** c0 = corres0;
  float** c1 = corres1;
  float** const c0max = corres0 + n0;
  float** const c1max = corres1 + n1;
  float* res = corres->pixels;
  float* r = res;
  while(c0<c0max && c1<c1max) {
    int d = memcmp(*c0,*c1,5*sizeof(float));
    if(d<0) { // corres0 < corres1
      c0++;
    } else 
    if(d>0) { // corres0 > corres1
      c1++;
    } else { // corres0 == corres1
      if( r==res || memcmp( r-tz, *c0, tz*sizeof(float) ) ) { // if not already copied
        memcpy( r, *c0, tz*sizeof(float) );
        r += tz;
      }
      c0++;
      c1++;
    }
  }
  
  free(corres0);
  free(corres1);
  return (r-res)/tz;
}

%}

%include <numpy_image.swg>

void _prepare_big_cells( int size, int offset, int step, 
                         int_cube* child_grid, float_image* child_norms,
                         int_cube* grid, int_cube* children, float_image* norms );

int _intersect_corres( const float_cube* map0, const float_cube* map1, float_image* corres );


%pythoncode %{

def prepare_big_cells( imshape, cell_size, overlap, child_overlap, child_grid, child_norms, dense_step=0, offset=None ):
  if offset is None:  offset = cell_size/2 if not dense_step else 0
  step = cell_size/(overlap+1) if not dense_step else dense_step
  grid_size = lambda imsize: 1+max(0,imsize-2*offset)/step
  gtx = grid_size(imshape[1])
  gty = grid_size(imshape[0])
  grid = numpy.empty((gty,gtx,2),numpy.int32)
  norms = numpy.zeros((gty,gtx),numpy.float32) if child_norms is not None else None
  
  assert overlap in {0,1}
  nc = (2+child_overlap)**2  # number of children per cell
  children = numpy.empty((gty,gtx,nc),numpy.int32) if child_grid!=None else None
  
  _prepare_big_cells( cell_size, offset, step, child_grid, child_norms, grid, children, norms )
  
  if norms is None:
    return step, grid, children
  else:
    return step, grid, children, norms

def intersect_corres( c0, c1 ):
  n = min(c0.size, c1.size) / 6
  res = numpy.empty((n,6), numpy.float32)
  n = _intersect_corres( c0, c1, res )
  return res[:n]


%}




















