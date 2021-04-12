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
#include "caffe/proto/caffe.pb.h"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "my_im2col.hpp"
#include "extra_layers.hpp"


#include <stdio.h>
#include <algorithm>
using std::min;
using std::max;
using std::swap;


namespace caffe {

  ////////////////////////////////////////////////////////////
  //        CONVOLUTION WITH CSR-SPARSE WEIGHTS            //
  //////////////////////////////////////////////////////////

/* Computes matrix transpose: B = A.T + beta * B
  
   A = mxn dense matrix (row-major order)
   B = nxm dense matrix (row-major order)
*/
template<typename Dtype>
void jerome_gpu_transpose( int m, int n, Dtype* A, Dtype* B, Dtype beta=0 );
template<>
void jerome_gpu_transpose<float>( int m, int n, float* A, float* B, float beta ) {
  float one=1;
  cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);
  cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, 
              m, n, &one, A, n, &beta, B, m, B, m);
}
template<>
void jerome_gpu_transpose<double>( int m, int n, double* A, double* B, double beta ) {
  double one=1;
  cublasSetPointerMode(Caffe::cublas_handle(), CUBLAS_POINTER_MODE_HOST);
  cublasDgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, 
              m, n, &one, A, n, &beta, B, m, B, m);
}


/* Computes:  C = alpha * A * B + beta * C
   
   A = mxk sparse matrix (CSR storage, row r contains val[rowptr[r]:rowptr[r+1]] 
                                        at columns colind[rowptr[r]:rowptr[r+1]] )
   B = kxn dense matrix  (row-major order)
   C = mxn dense matrix  (row-major order)
   
   if transA: use A.T instead, i.e A = kxm sparse matrix in CSR format
   if transB: use B.T instead, i.e B = nxk dense matrix in row-major order
   if transC: return C.T = nxm dense matrix in row-major order  
*/
template<typename Dtype>
void jerome_gpu_csrmm(int m, int n, int k, 
      Dtype alpha, CBLAS_TRANSPOSE transA, int nnz, const Dtype* valA, const int* rowptrA, const int* colindA,
      CBLAS_TRANSPOSE transB, const Dtype* B, 
      Dtype beta, CBLAS_TRANSPOSE transC, Dtype* C, Dtype* C_tmp);
template<>
void jerome_gpu_csrmm(int m, int n, int k, 
      float alpha, CBLAS_TRANSPOSE transA, int nnz, const float* valA, const int* rowptrA, const int* colindA,
      CBLAS_TRANSPOSE transB, const float* B, 
      float beta, CBLAS_TRANSPOSE transC, float* C, float* C_tmp) {
  
  const cusparseOperation_t trA = transA==CblasNoTrans ? 
             CUSPARSE_OPERATION_NON_TRANSPOSE : CUSPARSE_OPERATION_TRANSPOSE;
  
  const cusparseOperation_t trB = transB==CblasNoTrans ? // B in row-major
             CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int ldb = transB==CblasNoTrans ? n : k; // B in row-major
  
  const int ldc = m; // if transC==N we will transpose it afterwards
  float beta_csrmm = beta;
  if (transC == CblasNoTrans ) {
    CHECK_NE( C_tmp, (float*)0 ) << "Error: need a m*n buffer for transC=CNlasNoTrans";
    CHECK_NE( C_tmp, C )    << "Error: need a m*n buffer for transC=CNlasNoTrans";
    swap(C,C_tmp);
    beta_csrmm = 0; // real beta is used when we transpose
  }
  
  cusparseMatDescr_t descrA = 0; 
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );  // A is not symmetric or anything
  
  cusparseSetPointerMode(CusparseHandle::cusparse_handle(), CUSPARSE_POINTER_MODE_HOST);
  CUSPARSE_CHECK(cusparseScsrmm2(CusparseHandle::cusparse_handle(), trA, trB, m, n, k, nnz, 
                                 &alpha, descrA, valA, rowptrA, colindA, B, ldb, &beta_csrmm, C, ldc));
  cusparseDestroyMatDescr(descrA); 
  
  if (transC == CblasNoTrans ) {  // row-major demanded
    // transpose C because it is column-major (out-of-place)
    jerome_gpu_transpose( n, m, C, C_tmp, beta );  // fine as (C_tmp,C) have been swaped
  }
}
template<>
void jerome_gpu_csrmm(int m, int n, int k, 
      double alpha, CBLAS_TRANSPOSE transA, int nnz, const double* valA, const int* rowptrA, const int* colindA,
      CBLAS_TRANSPOSE transB, const double* B,
      double beta, CBLAS_TRANSPOSE transC, double* C, double* C_tmp) {
  NOT_IMPLEMENTED;
}

/* Load a CSR matrix stored in a blob.
   The csr matrix is concatenated as {nr, nc, rowptr, colind, data} (all integers, except data = Dtype)
*/
template<typename Dtype>
void CSR_SparseConvolutionLayer<Dtype>::load_csr_gpu( 
        const int** weight_rowptr, const int** weight_colind, 
        const Dtype** weight_data, Dtype** weight_diff ) {
  
  *weight_rowptr = recast<const int*>(weight_rowptr_->gpu_data());
  *weight_colind = recast<const int*>(weight_colind_->gpu_data());
  *weight_data = this->blobs_[0]->gpu_data();
  if (weight_diff)  *weight_diff = this->blobs_[0]->mutable_gpu_diff();
}

/* Convolution layer but with sparse filters.
   This layer computes:   Out = Filters x im2col( In )
   
   In and Out are dense matrices (NCHW fully-packed).
   Filters are given as a CSR matrix instead as dense matrix.
*/
template<typename Dtype>
void CSR_SparseConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* col_data = do_im2col_ ? this->col_buffer_.mutable_gpu_data() : NULL;
  Dtype* tmp_res = recast<Dtype*>(transpose_res_->mutable_gpu_data());
  
  // load sparse weights = concatenated {nr, nc, rowptr, colind, data} 
  const int *weight_rowptr, *weight_colind; const Dtype *weight_data; 
  load_csr_gpu( &weight_rowptr, &weight_colind, &weight_data, NULL );
  
  for (int n = 0; n < this->NUM_; ++n) {
    // First, im2col
    if (do_im2col_)
      my_im2col_gpu(bottom_data + bottom[0]->offset(n), this->CHANNELS_, this->HEIGHT_,
                      this->WIDTH_, this->KSIZE_, this->PAD_, this->STRIDE_, col_data);
    else
      col_data = const_cast<Dtype*>( bottom_data + bottom[0]->offset(n) );  // safe, won't be modified
    
    // Second, innerproduct 
    jerome_gpu_csrmm<Dtype>(this->M_, this->N_, this->K_, 
      1, CblasNoTrans, nnz_, weight_data, weight_rowptr, weight_colind, 
         CblasNoTrans, col_data,
      0, CblasNoTrans, top_data + top[0]->offset(n), tmp_res );
    
    // third, add bias
    if (this->biasterm_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, this->NUM_OUTPUT_,
          this->N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
          recast<const Dtype*>(this->bias_multiplier_->gpu_data()),
          (Dtype)1., top_data + top[0]->offset(n));
    }
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(CSR_SparseConvolutionLayer);


  ////////////////////////////////////////////////////////////
  //               BORDER RECTIFICATION                    //
  //////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void kernel_border(const long total, const Dtype* input_data, const int num, const int nchannels, 
                              const int height, const int width, const Dtype* weights, const int ks,
                              Dtype* res_data ) {
  // to be launched on total = nchannels * height * width
  CUDA_KERNEL_LOOP(index, total) {
    input_data += index;
    res_data += index;
    const int i = index % width;
    index /= width;
    const int j = index % height;
    const int chan = index / height;
    const int H = ks/2;
    
    // find multiplier for this particular pixel
    const int y = (j<H) ? j : (H>ks-height+j ? H : ks-height+j);
    const int x = (i<H) ? i : (H>ks-width+i ? H : ks-width+i);
    
    const long img_size = nchannels * height * width;
    const Dtype mul = weights[x + y*ks + chan*ks*ks];
    for(int n=0; n<num; ++n) 
      res_data[n*img_size] = mul * input_data[n*img_size];
  }
}

template <typename Dtype>
__global__ void kernel_border_fixed(const long total, const int height, const int width, const int ks,
                                    Dtype* data ) {
  const int H = ks/2;
  // to be launched on total = num * nchannels * height * width
  CUDA_KERNEL_LOOP(index, total) {
    data += index;
    const int i = index % width;
    index /= width;
    const int j = index % height;
    
    // find multiplier for this particular pixel
    const int x = min(i, max(H, ks-width+1 +i));
    const int y = min(j, max(H, ks-height+1+j));
    if (x==H && y==H) return;
    
    const Dtype mul = ks*ks / (Dtype(ks - abs(H-x)) * Dtype(ks - abs(H-y)));
    (*data) *= mul;
  }
}

template <typename Dtype, bool avoid_center>
__global__ void kernel_border_offsets(const long total, const int nchannels, 
                              const int true_height, const int true_width, 
                              const int height, const int width, const int grid_width, const int ngh_rad, //const Dtype* offsets, //
                              const Dtype* weights, const int ks, const int crop, Dtype* data ) {
  // to be launched on total = nchannels * height * width
  CUDA_KERNEL_LOOP(index, total) {
    data += index;
    if (*data) {  // if null => nothing to do
      int i = index % width;
      index /= width;
      int j = index % height;
      const int chan = index / height;
//      i += int(offsets[2*chan+0]);  // now expressed in full image coordinates
//      j += int(offsets[2*chan+1]);    
      const int H = ks/2;
      i += crop + H + (chan % grid_width) * ks - ngh_rad; // for some reason, slower than using offsets[]
      j += crop + H + (chan / grid_width) * ks - ngh_rad;
      
      // find multiplier for this particular pixel
      const int x = min(i, max(H, ks-true_width +i));
      const int y = min(j, max(H, ks-true_height+j));
      if (avoid_center && x==H && y==H ) return;
      
      if (x<0 || y<0 || x>=ks || y>=ks) {
        Dtype mul = weights[x + y*ks + chan*ks*ks];
        *data *= mul;
      }
    }
  }
}

template <typename Dtype, bool ori>
__global__ void kernel_border_fixed_offsets(const long total, const int nchannels, 
                              const int true_height, const int true_width, 
                              const int width, const int grid_width, const int ngh_rad, //const Dtype* offsets, //
                              const int ksize, const int crop, Dtype* data ) {
  // to be launched on total = num * nchannels * 2 * (ks-1) * width
  // assuming width == height !!! (MANDATORY)
  CUDA_KERNEL_LOOP(index, total) {
    int i = index % width;
    index /= width;
    int j = index % (ksize-1);
    index /= (ksize-1);
    int chan = index % nchannels;
    long num = index / nchannels;
    
    const int height = width;
    const int H = ksize/2;
    
    Dtype mul;
    if (ori) {
      // pattern = horizontal bar of (ksize-1, width)
      const int offy = crop + H + (chan / grid_width) * ksize - ngh_rad;
      if (offy <= H-1) { // patch overlaps top border
        mul = ksize / Dtype(1+j);
        j += (1-H) - offy;
      
      } else if (offy+height > true_height-H) {// patch overlaps bottom border
        mul = ksize / Dtype(ksize-1-j);
        j += true_height-H - offy;  // actually: true_height = true_height+1 actually
        
      } else return;
      if (j<0 || j>=height)  return;
      
    } else {
      {int t=i+width*j; i=t%(ksize-1); j=t/(ksize-1);}  // swap i and j contiguously
      // pattern = vertical bar of (height, ksize-1)
      const int offx = crop + H + (chan % grid_width) * ksize - ngh_rad; // for some reason, slower than using offsets[]
      if (offx <= H-1) { // patch overlaps left border
        mul = ksize / Dtype(1+i);
        i += (1-H) - offx;
      
      } else if (offx+width > true_width-H) {// patch overlaps right border
        mul = ksize / Dtype(ksize-1-i);
        i += true_width-H - offx;
        
      } else return;
      if (i<0 || i>=width)  return;
    }
    
    data[((num*nchannels + chan)*height + j)*width + i] *= mul;
  }
}

template <typename Dtype>
void BorderRectifyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  const Dtype* mask = this->blobs_[0]->gpu_data();
  // cannot verify mask weight because it's on the GPU, so we just trust user
  
  const long num = bottom[0]->num();
  const long nchan = bottom[0]->channels();
  const long height = bottom[0]->height();
  const long width = bottom[0]->width();
  
  // could be well accelerated if all center pixels were just not processed at all
  
  const long total = nchan * height * width;
  kernel_border<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, bottom_data, 
               bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), 
               mask, KSIZE_, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FORWARD(BorderRectifyLayer);


  ////////////////////////////////////////////////////////////
  //                   RECTIFIED SIGMOID                   //
  //////////////////////////////////////////////////////////

template <typename Dtype>
__device__ inline Dtype rectified_sigmoid_gpu(Dtype x) {
  return Dtype(2.) / (Dtype(1.) + exp(-x)) - Dtype(1.);
}

template <typename Dtype>
__global__ void kernel_rectified_sigmoid(const long total, const Dtype* input_data, Dtype* res_data ) {
  CUDA_KERNEL_LOOP(index, total) {
    res_data[index] = rectified_sigmoid_gpu( input_data[index] );
  }
}

template <typename Dtype>
void RectifiedSigmoidLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  const long total = bottom[0]->count();
  kernel_rectified_sigmoid<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FORWARD(RectifiedSigmoidLayer);


  ////////////////////////////////////////////////////////////
  //                PIXEL NORMALIZATION                    //
  //////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void kernel_pixel_norm(const long total, const Dtype* input_data, int num, int channels, 
                                  const int npix, float alpha, Dtype* res_data ) {
  CUDA_KERNEL_LOOP(index, total) {
    int img = index / npix;
    index = (index % npix) + img * channels * npix;
    
    Dtype sq_sum = 1e-16;
    for(int i=0; i<channels; ++i)
      sq_sum += input_data[index + i*npix] * input_data[index + i*npix];
    Dtype norm = sqrt(alpha/sq_sum);
    
    for(int i=0; i<channels; ++i)
      res_data[index + i*npix] = norm * input_data[index + i*npix];
  }
}

template <typename Dtype>
void PixelNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num_ = bottom[0]->num();
  const int channels_ = bottom[0]->channels();
  const int height_ = bottom[0]->height();
  const int width_ = bottom[0]->width();
  
  const long npix = height_ * width_;
  const long total = num_ * npix; // one thread for each image pixel
  
  kernel_pixel_norm<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, 
      bottom_data, num_, channels_, height_ * width_, alpha_, top_data );
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FORWARD(PixelNormLayer);


  ////////////////////////////////////////////////////////////
  //                  PATCH CONVOLUTION                    //
  //////////////////////////////////////////////////////////

/* Dense to sparse copy
  theoretical padding (e.g. 128) >> eff padding (e.g. 3)
  virtual padding = theoretical padding - effective padding
  everything beyond effective_padding is zeros, so we don't need these values explicitly
*/
template <typename Dtype>
__global__ void kernel_select_nghrad_autopad(const long total, const long index_offset, 
                                     const int height_out, const int width_out, const int grid_width, 
                                     const Dtype* dense_res, const int full_width, const int full_height, 
                                     const int virtual_pad, Dtype* sparse_res ) {
  CUDA_KERNEL_LOOP(index, total) {
    index += index_offset;
    sparse_res += index;
    const int i_out = index % width_out;
    index /= width_out;
    const int j_out = index % height_out;
    index /= height_out;
    const int c_out = index;  // patch index
    
    const int step = 4;
    const int x = i_out + (c_out % grid_width) * step - virtual_pad;
    const int y = j_out + (c_out / grid_width) * step - virtual_pad;
    *sparse_res = (x < 0 || y < 0 || x >= full_width || y >= full_height ) ? 0 :
                  dense_res[ x + full_width*(y +  full_height*c_out) ];
  }
}

template <typename Dtype>
__global__ void sqr_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index]*a[index];
  }
}
template <typename Dtype>
void caffe_gpu_sqr(const int N, const Dtype* a, Dtype* y) {
  sqr_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void inv_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = Dtype(1)/a[index];
  }
}
template <typename Dtype>
void caffe_gpu_inv(const int N, const Dtype* a, Dtype* y) {
  inv_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void PatchConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int second = bottom[0]->offset(1)/2;  // offset for second image
  
  Dtype* patches_data = patches_.mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  //const Dtype* offsets_data = this->blobs_[0]->gpu_data();
  CHECK_EQ( width_ % KSIZE_, 0 );
  const int grid_width = (width_ - 2*CROP_) / KSIZE_;
  
  const int M_ = channels_out_;
  const int N_ = col_buffer_.width() * col_buffer_.height();
  const int N_ONES_ = ones_col_.count() / K_; 
  const int height_out = top[0]->height();
  const int width_out = top[0]->width();
  CHECK_EQ( KSIZE_, 4 );  // see step == 4 defined in the kernel
  const int true_height = bottom[0]->height()/STRIDE_ + 1;
  const int true_width  = bottom[0]->width() /STRIDE_ + 1;
  
  // go through the images
  for (int n = 0; n < num_; n++) {
    const int contiguous_patches = 1;
    
    // first, transform first image into consecutive patches
    if (contiguous_patches)
      my_im2col_gpu_T( bottom_data + bottom[0]->offset(n), channels_, height_,
                width_, KSIZE_, -CROP_, KSIZE_, patches_data);
    else
      my_im2col_gpu( bottom_data + bottom[0]->offset(n)+second, channels_, height_,
                width_, KSIZE_, -CROP_, KSIZE_, patches_data);
    
    // then, prepare second image for dense convolution
    my_im2col_gpu( bottom_data + bottom[0]->offset(n)+second, channels_, height_,
                width_, KSIZE_, PAD_, STRIDE_, col_data);
    
    // finally, perform dense dotproduct
    if (ngh_rad_ < 0 ) { 
      caffe_gpu_gemm<Dtype>(contiguous_patches?CblasNoTrans:CblasTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1.,  patches_data, col_data,
              (Dtype)0, top_data + top[0]->offset(n));
    } else {
      CHECK_GE( virtual_pad_, 0 );
      CHECK_EQ( contiguous_patches, 1 ) << "error: patches have to be contiguous here";
      const long step_m = dense_buf_.channels();
      Dtype* dense_buf = dense_buf_.mutable_gpu_data();
      
      for (long m = 0; m < M_; m += step_m) {
        // compute full convolutions for subset of patches because GEMM is hyper-fast
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, min(step_m,M_-m), N_, K_,
              (Dtype)1., patches_data + m * K_, col_data,
              (Dtype)0, dense_buf );
        const long total_m = min(step_m,M_-m) * height_out * width_out;
        // retrieve just the values that we are interested in
        kernel_select_nghrad_autopad<Dtype><<<CAFFE_GET_BLOCKS(total_m), CAFFE_CUDA_NUM_THREADS>>>(total_m,
              m * height_out * width_out, height_out, width_out, grid_width, 
              dense_buf - m*N_, col_buffer_.width(), col_buffer_.height(), virtual_pad_, 
              top_data + top[0]->offset(n) );
        CUDA_POST_KERNEL_CHECK;
      }
    }
    
    if (normalize_borders_ == 'd') {
      // dynamic normalization of image borders
      const Dtype* ones_data = ones_col_.gpu_data();
      Dtype* mask_data = masks_.mutable_gpu_data();
      
      caffe_gpu_sqr<Dtype>(patches_.count(), patches_data, patches_data); // square values
      caffe_gpu_gemm<Dtype>(contiguous_patches?CblasNoTrans:CblasTrans, CblasNoTrans, M_, N_ONES_, K_,
              (Dtype)1., patches_data, ones_data,
              (Dtype)0, mask_data );  // sum on image borders
      caffe_gpu_inv<Dtype>(masks_.count(), mask_data, mask_data); // compute inverse of squared norm
      const long total = top[0]->count() / top[0]->num();
      if (ngh_rad_<0) {
        kernel_border<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, 
               top_data + top[0]->offset(n), 1, channels_out_, height_out, width_out, 
               mask_data, ones_col_.width(), top_data + top[0]->offset(n));
      } else {
        NOT_IMPLEMENTED; // looks buggy from python test_extra_layers3.py corr
        kernel_border_offsets<Dtype,true><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, 
               channels_out_, true_height, true_width, height_out, width_out, grid_width, ngh_rad_,
               mask_data, ones_col_.width(), CROP_, top_data + top[0]->offset(n));
      }
      CUDA_POST_KERNEL_CHECK;
    }
  }
  
  if (normalize_borders_ == 's') {
    if (ngh_rad_<0) {
      const long total = top[0]->count();
      kernel_border_fixed<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, 
                height_out, width_out, KSIZE_, top_data );
      CUDA_POST_KERNEL_CHECK;
    } else {
      CHECK_EQ( width_out, height_out );
      CHECK_EQ( STRIDE_, 1 );
      const long total = long(num_) * 
                         channels_out_ * (KSIZE_-1) * width_out;
      kernel_border_fixed_offsets<Dtype,false><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, 
              channels_out_, true_height, true_width, width_out, grid_width, ngh_rad_, 
              KSIZE_, CROP_, top_data);
      CUDA_POST_KERNEL_CHECK;
      kernel_border_fixed_offsets<Dtype,true><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, 
              channels_out_, true_height, true_width, width_out, grid_width, ngh_rad_, 
              KSIZE_, CROP_, top_data);
      CUDA_POST_KERNEL_CHECK;
    }
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(PatchConvolutionLayer);



  ////////////////////////////////////////////////////////////
  //                DeepMatching ArgMax layer              //
  //////////////////////////////////////////////////////////

/* Some function to embed 2 floating points into 1.
  This assume that the 2 floats are in [0,1] and [0,10], respectively,
  and only allows to save 3-4 digits for each float.
*/
template <typename Dtype>
__device__ inline Dtype combine_2_floats( Dtype first, Dtype second ) {
  // combine 2 floats into a single one
  return 2*floor(1000.f*second) + first;
}
template <typename Dtype>
__device__ inline Dtype retrieve_2_floats( Dtype v, Dtype* second ) {
  // retrieve 2 floats from a single one
  v /= 2;
  float i = floor(v);
  *second = i/1000.f;
  return 2*(v-i);
}
template <typename Dtype>
__device__ inline Dtype retrieve_first_float( Dtype v ) {
  v /= 2;
  return 2*(v-floor(v)); 
}
template <typename Dtype>
__device__ inline Dtype retrieve_second_float( Dtype v ) {
  return floor(v/2)/1000.f; 
}

// find best score in a 3x3 local window around (i,j)
template <typename Dtype>
__device__ inline Dtype local_argmax_first(const Dtype* data, int* j, int* i, int height, int width) {
  const int mini = max(0, *i-1); 
  const int maxi = min(*i+2, width);
  const int minj = max(0, *j-1); 
  const int maxj = min(*j+2, height);
  
  Dtype res = 0;
  int besti, bestj;
  for(int v=minj; v<maxj; ++v)
    for(int u=mini; u<maxi; ++u) {
      Dtype s = data[v*width + u];
      s = retrieve_first_float(s);  // insensitive to multiplexed floats
      if (s>res)  {res=s; besti=u; bestj=v;}
    }
  *i = besti;
  *j = bestj;
  return res;
}

// retrieve patch at pixel position (x,y)
// offset = step/2 if level==0 else 0
__device__ inline int retrieve_patch( const int x, const int y, const int step, const int offset,
                                      const int grid_width, const int grid_height ) {
  const int i = (x - offset) / step;
  const int j = (y - offset) / step;
  return ( i<0 || i>=grid_width || j<0 || j>=grid_height ) ? -1 : i + j * grid_width;
}

typedef unsigned long long ULONG;
//#define ULONG unsigned long long

__device__ inline ULONG multiplex_score_pos0_pos1(float score, int i0, int j0, int i1, int j1) {
  ULONG res = ((__float_as_int(score)+0x80000000) & 0xFFFFFFF0);  // center 0 and erase last 4 bits
  res <<= 32; // lower 36 bits are 0
  res |= (j0 & 0xFFL) << (36-8);          // i0 in 4*f*[0,255]
  res |= (i0 & 0xFFL) << (36-8-8);        // j0 in 4*f*[0,255]
  res |= (j1 & 0x3FF) << (36-8-8-10);     // i1 in f*[0,1023]
  res |= (i1 & 0x3FF) << (36-8-8-10-10);  // j1 in f*[0,1023]
  return res;
}
__device__ inline float demultiplex_score_pos(ULONG plex, int* i0, int* j0, int *i1, int* j1) {
  *j0 = (plex >> 28) & 0xFFL;
  *i0 = (plex >> 20) & 0xFFL;
  *j1 = (plex >> 10) & 0x3FF;
  *i1 = (plex >> 00) & 0x3FF;
  return __int_as_float(((plex>>32) & 0xFFFFFFF0) - 0x80000000);
}


template <typename Dtype, bool use_nghrad, bool safe_write>
__device__ inline void write_corres( const int n, const int c, int i1, int j1, const Dtype score, 
                                     const int img_width, const int img_height,
                                     const int grid_width, const int grid_height, const int step,
                                     const int nghrad, void* _corres ) {
  const int i0 = step/2 + (c%grid_width) * step;
  const int j0 = step/2 + (c/grid_width) * step;
  const int f = step/4; 
  
  // quantized coordinates in first image
  const int qi0 = i0/step;
  const int qj0 = j0/step;
  // coordinates in second image
  if (use_nghrad) {
    // assuming that offsets = (grid/f) - (ngh_rad/f)
    i1 = f*i1 - nghrad + i0;  // add offset to cancel ngh_rad
    j1 = f*j1 - nghrad + j0;
  } else {
    i1 *= f;
    j1 *= f;
  }
  
  if( 0<=i1 && 0<=j1 && i1<img_width && j1<img_height ) {
    const int qi1 = i1/step;
    const int qj1 = j1/step;
    
    if (safe_write) {
      // compact everything into a long integer !
      ULONG res = multiplex_score_pos0_pos1(score, i0/step, j0/step, i1/f, j1/f);
      
      // goto correct image
      ULONG* corres = ((ULONG*)_corres) + n * 2 * grid_height * grid_width; 
      atomicMax( corres + (qj0*grid_width + qi0), res );
      
      corres += grid_height * grid_width; 
      atomicMax( corres + (qj1*grid_width + qi1), res );
    } else {
      // this part is problematic: 
      // execution order is random hence results are not predictable
      // even using atomicMax, nothing is really guarranted...
      
      // goto correct image
      Dtype* corres = ((Dtype*)_corres) + n * 12 * grid_height * grid_width;  
      
      Dtype* corres0 = corres + 6*(qj0*grid_width + qi0);
      float score0 = __int_as_float(atomicMax((int*)(corres0+4), __float_as_int(score)));
      if (score0 < score ) {
        corres0[0] = i0;
        corres0[1] = j0;
        corres0[2] = i1;
        corres0[3] = j1;
      }
      const int qi1 = i1/step;
      const int qj1 = j1/step;
      Dtype* corres1 = corres + 6*((grid_height + qj1)*grid_width + qi1);
      float score1 = __int_as_float(atomicMax((int*)(corres1+4), __float_as_int(score)));
      if (score1 < score ) {
        corres1[0] = i0;
        corres1[1] = j0;
        corres1[2] = i1;
        corres1[3] = j1;
      }
    }
  }
}

// purpose of this kernel: propagate the scores from parents to all children
// in a single wavefront (all parent cells are examined once, i.e. 1 thread per parent cell)
template <typename Dtype, bool use_nghrad, bool safe_write>
__global__ void kernel_parent_children( const long total, const int ch_level, const bool top_level,
                   const Dtype* parent_data, const int par_channels, const int par_height, const int par_width,
                   Dtype* child_data, const int ch_channels, const int ch_height, const int ch_width, 
                   const int img_width, const int img_height, const int step, const int nghrad, void* corres ) {
  CUDA_KERNEL_LOOP(index, total) {
    Dtype par_score = parent_data[index];
    
    // check if parent is valid (i.e. on an existing path)
    if (top_level || par_score>=2) {
      par_score = retrieve_second_float(par_score); // retrieve parent score
      
      int par_i1 = index % par_width;
      index /= par_width;
      int par_j1 = index % par_height;
      index /= par_height;
      int c = index % par_channels;
      int n = index / par_channels;
      
      const int grid_width =  img_width  / step;
      const int grid_height = img_height / step;
      
      // coordinates (center of patch) in first image (due to the regular grid)
      const int i0 = (c%(grid_width+1)) * step;
      const int j0 = (c/(grid_width+1)) * step;
      
      // update every child
      for(int ch = 0; ch < 4; ch++) {
        const int u = 2*(ch%2) - 1;  // in {-1,1}
        const int v = 2*(ch/2) - 1;  // in {-1,1}
        c = retrieve_patch( i0 + u*((step/2)<<ch_level), j0 + v*((step/2)<<ch_level), 
                            step, ch_level?0:step/2, grid_width+(ch_level>0), grid_height+(ch_level>0) );
        if (c>=0) {
          int i1, j1;
          if (use_nghrad) {
            i1 = 2*par_i1;
            j1 = 2*par_j1;
          } else {
            i1 = 2*(par_i1+u);
            j1 = 2*(par_j1+v);
          }
          
          // do a local argmax to refine the current position
          Dtype* ch_layer = child_data + (n*ch_channels + c) * ch_height * ch_width;  // goto correct layer
          Dtype s = local_argmax_first(ch_layer, &j1, &i1, ch_height, ch_width);  // best local score
          
          if (s>0) {
            if (ch_level==0) {  // bottom level -> directly write result
              write_corres<Dtype,use_nghrad, safe_write>( n, c, i1, j1, par_score + s, 
                   img_width, img_height, grid_width, grid_height, step, nghrad, corres );
            } else {
              s = combine_2_floats(s, par_score + s);
              atomicMax( (int*)(ch_layer + j1 * ch_width + i1), __float_as_int(s) );
            }
          }
        }
      }
    }
  }
}

template< typename Dtype >
__global__ void kernel_demultiplex_corres(const long total, ULONG* data, const int f, Dtype* corres) {
  const int step = 4;
  CUDA_KERNEL_LOOP(index, total) {
    // read value
    int i0,j0,i1,j1;
    float score = demultiplex_score_pos( data[index/6], &i0, &j0, &i1, &j1 );
    
    // write output
    const int n = index % 6;
    if (n < 2) 
      corres[index] = f*(step/2 + step*((n==0) ? i0 : j0));
    else if (n < 4)
      corres[index] = (n==2) ? f*i1 : f*j1;
    else
      corres[index] = (n==4) ? score : 0;
  }
}


template <typename Dtype>
void DeepMatchingArgMaxLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 
  CHECK_EQ( bottom[0], this->blobs_.back().get()) << "error: previous activation_blobs must be supplied manually";
  CHECK_EQ( top[0]->height()*step_, img_height_ ) << "error: corres0 and corres1 height must mutiple of step";
  CHECK_EQ( top[0]->width()*step_, img_width_ ) << "error: corres0 and corres1 width must mutiple of step";
  CHECK_EQ( img_width_*img_height_, step_*step_*this->blobs_[0]->channels() ) << "error: grid doesn't match lowest pyramid blob";
  
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  const int n_levels = this->blobs_.size();
  const int grid_size = img_width_ * img_height_ / (step_*step_);
  
  // if activated, slightly slower but results are more stable and also denser
  const bool safe_write = true;
  
  // init synced mem to 0
  void* corres_data = top_data;
  shared_ptr<SyncedMemory> buf;
  if (safe_write) {
    buf.reset( new SyncedMemory( bottom[0]->num() * 2 * grid_size * sizeof(ULONG)) );
    corres_data = buf->mutable_gpu_data();
    CUDA_CHECK(cudaMemset(corres_data, 0, buf->size()));
    
    CHECK_LE( img_width_ / step_, 256 ) << "image is too large (see multiplex_score())";
    CHECK_LE( img_height_ / step_, 256 ) << "image is too large (see multiplex_score())";
  }
  else
    CUDA_CHECK(cudaMemset( top_data, 0, top[0]->count()*sizeof(Dtype) ));  // init to 0
  
  for(int ch_level=n_levels-2; ch_level>=0; ch_level--) {
    // load children and parent maps
    const Blob<Dtype>* par = this->blobs_[ch_level+1].get();
    Blob<Dtype>* ch = this->blobs_[ch_level].get();
    const long total = par->count();
    
    if(ngh_rad_<0) {
      const bool use_nghrad = false;
      kernel_parent_children<Dtype, use_nghrad, safe_write><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>( total, 
                 ch_level, ch_level==n_levels-2, 
                 par->gpu_data(), par->channels(), par->height(), par->width(),
                 ch->mutable_gpu_data(), ch->channels(), ch->height(), ch->width(), 
                 img_width_, img_height_, step_, ngh_rad_, corres_data );
    } else {
      const bool use_nghrad = true;
      kernel_parent_children<Dtype, use_nghrad, safe_write><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>( total, 
                 ch_level, ch_level==n_levels-2, 
                 par->gpu_data(), par->channels(), par->height(), par->width(),
                 ch->mutable_gpu_data(), ch->channels(), ch->height(), ch->width(), 
                 img_width_, img_height_, step_, ngh_rad_, corres_data );
    }
    CUDA_POST_KERNEL_CHECK;
  }
  
  // decompresse correspondences from ULONG to (float,int,int,int,int)
  if (safe_write) { 
    const long total = top[0]->count();
    CHECK_EQ( total, 6 * bottom[0]->num() * 2 * grid_size );  // size of corres_data
    kernel_demultiplex_corres<Dtype><<<CAFFE_GET_BLOCKS(total), CAFFE_CUDA_NUM_THREADS>>>(total, 
        (ULONG*)corres_data, step_/4, top_data);
    CUDA_POST_KERNEL_CHECK;
  }
}

//INSTANTIATE_LAYER_GPU_FORWARD(DeepMatchingArgMaxLayer); // no double
template class DeepMatchingArgMaxLayer<float>;


}// namespace caffe























