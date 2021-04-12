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
#include <stdio.h>
#include "caffe/proto/caffe.pb.h"
#include "my_im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"
#include "extra_layers.hpp"

#include <algorithm>
using std::min;
using std::max;

namespace caffe{


  ////////////////////////////////////////////////////////////
  //        CONVOLUTION WITH CSR-SPARSE WEIGHTS            //
  //////////////////////////////////////////////////////////

// init cusparse handle
CusparseHandle CusparseHandle::instance_ = CusparseHandle();

template<typename Dtype>
void CSR_SparseConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";
  
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  
  // Set the parameters
  CHECK   ( conv_param.has_kernel_h() && conv_param.has_kernel_w());
  CHECK_EQ( conv_param.has_kernel_h() ,  conv_param.has_kernel_w());
  KSIZE_ = conv_param.kernel_h();
  CHECK   ( conv_param.has_stride_h() && conv_param.has_stride_w());
  CHECK_EQ( conv_param.has_stride_h() ,  conv_param.has_stride_w());
  STRIDE_ = conv_param.stride_h();
  CHECK   ( conv_param.has_pad_h() && conv_param.has_pad_w());
  CHECK_EQ( conv_param.has_pad_h() ,  conv_param.has_pad_w());
  PAD_ = conv_param.pad_h();
  
  do_im2col_ = !(this->KSIZE_==1 && this->PAD_==0 && this->STRIDE_==1);
  
  biasterm_ = conv_param.bias_term();
  
  NUM_OUTPUT_ = conv_param.num_output();
  CHECK_GT(NUM_OUTPUT_, 0);
  
  // at least one blob for sparse weight CSR matrix
  this->blobs_.resize(1);
}
template<typename Dtype>
void CSR_SparseConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionParameter conv_param = this->layer_param_.convolution_param();
  
  NUM_ = bottom[0]->num();
  CHANNELS_ = bottom[0]->channels();
  HEIGHT_ = bottom[0]->height();
  WIDTH_ = bottom[0]->width();
  
  // The im2col result buffer would only hold one image at a time to avoid
  // overly large memory usage.
  int height_out = (HEIGHT_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  int width_out = (WIDTH_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  if (do_im2col_)
    col_buffer_.Reshape(1, CHANNELS_ * KSIZE_ * KSIZE_, height_out, width_out);
  
    // Figure out the dimensions for individual gemms.
  M_ = NUM_OUTPUT_;
  K_ = CHANNELS_ * KSIZE_ * KSIZE_;
  N_ = height_out * width_out;
  top[0]->Reshape(bottom[0]->num(), NUM_OUTPUT_, height_out, width_out);
  
  if (biasterm_ && this->blobs_.size()<=1) {
    this->blobs_.resize(2);
    
    // Intialize the bias term
    this->blobs_[1].reset(new Blob<Dtype>(1, 1, 1, NUM_OUTPUT_));
    GetFiller<Dtype>(conv_param.bias_filler())->Fill( this->blobs_[1].get() );
  }
  
  if (biasterm_) {
    bias_multiplier_.reset(new SyncedMemory(N_ * sizeof(Dtype)));
    Dtype* bias_multiplier_data = reinterpret_cast<Dtype*>(bias_multiplier_->mutable_cpu_data());
    for (int i = 0; i < N_; ++i) bias_multiplier_data[i] = 1.;
  }
  
  transpose_res_.reset( new SyncedMemory( max(this->M_, this->K_) * this->N_ * sizeof(Dtype)) );
}


template<typename Dtype>
void CSR_SparseConvolutionLayer<Dtype>::SetSparsityPattern( const int nnz, 
          const Blob<float>* row_ptr, const Blob<float>* col_ind, const Blob<float>* data ) {
  CHECK_GE( nnz, 0 );
  CHECK_GT( this->M_, 0 ) << "Error: SetUp() must be call before";
  CHECK_EQ( row_ptr->count(), this->M_+1 );
  CHECK_EQ( col_ind->count(), nnz );
  const int* indptr  = recast<const int*>(row_ptr->cpu_data());
  for(int i=0; i<this->M_; i++) {
    CHECK_GE( indptr[i], 0 ) << "bad raw_ptr index";
    CHECK_LE( indptr[i], nnz ) << "bad raw_ptr index";
  }
  CHECK_EQ( indptr[this->M_], nnz ) << "bad raw_ptr index";
  const int* indices = recast<const int*>(col_ind->cpu_data());
  for(int i=0; i<nnz; i++) {
    CHECK_GE( indices[i], 0 ) << "bad col index";
    CHECK_LT( indices[i], this->K_ ) << "bad col index";
  }
  
  nnz_ = nnz;
  
  // copy sparsity pattern
  weight_rowptr_.reset( new SyncedMemory( (this->M_+1)*sizeof(int) ) );
  weight_colind_.reset( new SyncedMemory( nnz*sizeof(int) ) );
  
  memcpy( weight_colind_->mutable_cpu_data(), col_ind->cpu_data(), weight_colind_->size() );
  memcpy( weight_rowptr_->mutable_cpu_data(), row_ptr->cpu_data(), weight_rowptr_->size() );
  
  // fill weights
  this->blobs_[0].reset(new Blob<Dtype>(1, 1, 1, nnz));
  if (data) {
    CHECK_EQ( data->count(), nnz );
    memcpy( this->blobs_[0]->mutable_cpu_data(), data->cpu_data(), nnz*sizeof(Dtype) );
  } else {
    GetFiller<Dtype>(this->layer_param_.convolution_param().weight_filler())
        ->Fill(this->blobs_[0].get());
  }
}

INSTANTIATE_CLASS(CSR_SparseConvolutionLayer);


  ////////////////////////////////////////////////////////////
  //               BORDER RECTIFICATION                    //
  //////////////////////////////////////////////////////////

template <typename Dtype>
void BorderRectifyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "BorderRectifyLayer Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Transpose Layer takes a single blob as output.";
  
  CHECK_EQ( KSIZE_ % 2,  1) << "KSIZE must always be symmetric";
  
  // parameter = same as kernel size
  this->blobs_.resize(1);
  // the weights must be initialized by user
  this->blobs_[0].reset(new Blob<Dtype>(1, bottom[0]->channels(), KSIZE_, KSIZE_));
}

template <typename Dtype>
void BorderRectifyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape( bottom[0]->num(), bottom[0]->channels(), 
                   bottom[0]->height(), bottom[0]->width() );
}

template <typename Dtype>
inline void rectify_border( const int nchan, const int height, const int width, const int ms,
                            const Dtype* mask, const Dtype* input_data, Dtype* res_data ) {
  CHECK_EQ( ms % 2, 1 );
  const int H = ms/2;
  
  for(int c=0; c<nchan; ++c) {
    for(int j=0; j<height; ++j) { 
      // find current position in mask
      const int y = (j<H) ? j : (H>ms-height+j ? H : ms-height+j);
      for(int i=0; i<width; ++i) {
        // find current position in mask
        const int x = (i<H) ? i : (H>ms-width+i ? H : ms-width+i);
        *res_data++ = (*input_data++) * mask[x + ms*(y + c*ms)];
      }
    }
  }
}

template <typename Dtype>
inline void rectify_border_fixed(const long total, const int height, const int width, 
                                 const int ks, const int ms, Dtype* data ) {
  CHECK_EQ( ms % 2, 1 );
  const int H = ms/2;
  const int channels = total / height / width;
  
  for (int c = 0; c < channels; c++, data+=width*height)
    for (int j = 0; j < height; j++) {
      for (int i = 0; i < width; i++) {
        if (i==H && (j>=H && j<height-1-H)) 
          i = max(i, (width-1)-H); // fast foward
        
        // find multiplier for this particular pixel
        const int x = min(i, max(H, ms-width +i));
        const int y = min(j, max(H, ms-height+j));
        
        const Dtype mul = ks*ks / (Dtype(ks - abs(H-x)) * Dtype(ks - abs(H-y)));
        data[i + j*width] *= mul;
      }
    }
}

template <typename Dtype>
inline void rectify_border_offsets( const int nchan, const int true_height, const int true_width, 
                            const int height, const int width, const int grid_width, 
                            const int ngh_rad, const int KSIZE_, const int ms, const int CROP_, 
                            const Dtype* mask, const Dtype* input_data, Dtype* res_data ) {
  const int H = ms/2;
  
  for(int c=0; c<nchan; ++c) {
    //const int ox = int(offsets[2*c+0]);
    //const int oy = int(offsets[2*c+1]);
    const int ox = CROP_ + H + (c % grid_width) * KSIZE_ - ngh_rad;
    const int oy = CROP_ + H + (c / grid_width) * KSIZE_ - ngh_rad;
    
    for(int j_=0; j_<height; ++j_) { 
      const int j = j_+oy;  // true y position on full image
      // find current position in mask
      const int y = min(j, max(H, ms-true_height+j));
      
      for(int i_=0; i_<width; ++i_) {
        const int i = i_+ox;  // true x position on full image
        // find current position in mask
        const int x = min(i, max(H, ms-true_width +i));
        
        *res_data++ = (x<0 || y<0 || x>=ms || y>=ms) ? 0 :
                      (*input_data) * mask[x + ms*(y + c*ms)];
        input_data++;
      }
    }
  }
}

template <typename Dtype>
inline void rectify_border_fixed_offsets( const int nchan, const int true_height, const int true_width, 
                            const int height, const int width, const int grid_width, 
                            const int ngh_rad, const int ks, const int ms, const int CROP_, 
                            const Dtype* input_data, Dtype* res_data ) {
  //CHECK_EQ( ms % 2, 1 );
  const int H = ms/2;
  
  for(int c=0; c<nchan; ++c) {
    //const int ox = int(offsets[2*c+0]);
    //const int oy = int(offsets[2*c+1]);
    const int ox = CROP_ + H + (c % grid_width) * ks - ngh_rad;
    const int oy = CROP_ + H + (c / grid_width) * ks - ngh_rad;
    
    for(int j_=0; j_<height; ++j_) { 
      const int j = j_+oy;  // true y position on full image
      // find current position in mask
      const int y = min(j, max(H, ms-true_height+j));
      
      for(int i_=0; i_<width; ++i_) {
        const int i = i_+ox;  // true x position on full image
        // find current position in mask
        const int x = min(i, max(H, ms-true_width +i));
        
        *res_data++ = (x<0 || y<0 || x>=ms || y>=ms) ? 0 :
                      (*input_data) * ks*ks / (Dtype(ks - abs(H-x)) * Dtype(ks - abs(H-y)));
        input_data++;
      }
    }
  }
}

template <typename Dtype>
void BorderRectifyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  const Dtype* mask = this->blobs_[0]->cpu_data();
  
  const int num = bottom[0]->num();
  const int nchan = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  
  for(int n=0; n<num; ++n) 
    rectify_border( nchan, height, width, KSIZE_, mask, 
                    bottom_data + bottom[0]->offset(n), 
                    top_data + top[0]->offset(n) );
}

INSTANTIATE_CLASS(BorderRectifyLayer);



  ////////////////////////////////////////////////////////////
  //                   RECTIFIED SIGMOID                   //
  //////////////////////////////////////////////////////////

template <typename Dtype>
inline Dtype rectified_sigmoid(Dtype x) {
  return 2. / (1. + exp(-x)) - 1.;
}

template <typename Dtype>
void RectifiedSigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  
  for (int i = 0; i < count; ++i)
    top_data[i] = rectified_sigmoid( bottom_data[i] );
}

INSTANTIATE_CLASS(RectifiedSigmoidLayer);


  ////////////////////////////////////////////////////////////
  //                PIXEL NORMALIZATION                    //
  //////////////////////////////////////////////////////////

template <typename Dtype>
void PixelNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num_ = bottom[0]->num();
  const int channels_ = bottom[0]->channels();
  const int height_ = bottom[0]->height();
  const int width_ = bottom[0]->width();
  
  Blob<Dtype> squared_(1, channels_, height_, width_);
  Dtype* squared_data = squared_.mutable_cpu_data();
  
  Blob<Dtype> pix_norm_(1, 1, height_, width_);
  Dtype* pix_norm_data = pix_norm_.mutable_cpu_data();
  
  const int npix = height_ * width_;
  
  // go through the images
  for (int n = 0; n < num_; ++n) {
    // compute the squared data
    caffe_sqr(channels_ * npix,
        bottom_data + bottom[0]->offset(n),
        squared_data );
    
    // Sum squared values for all layers (could be well optimized with gemv)
    memset(pix_norm_data, 0, sizeof(Dtype)*pix_norm_.count());
    for (int c = 0; c < channels_; ++c)
      caffe_axpy<Dtype>(npix, 1,
          squared_data + squared_.offset(0, c),
          pix_norm_data);
    
    // compute pixel norms
    caffe_powx<Dtype>(npix, pix_norm_data, -0.5, pix_norm_data);  // square-root
    if (alpha_!=1)
      caffe_scal<Dtype>(npix, sqrt(alpha_), pix_norm_data); // multiply by constant
    
    // multiply each channel
    for (int c = 0; c < channels_; ++c) 
      caffe_mul<Dtype>(npix, bottom_data + bottom[0]->offset(n,c),
          pix_norm_data, top_data + top[0]->offset(n,c) );
  }
}

INSTANTIATE_CLASS(PixelNormLayer);


  ////////////////////////////////////////////////////////////
  //                  PATCH CONVOLUTION                    //
  //////////////////////////////////////////////////////////

template <typename Dtype>
void PatchConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 1) << "PatchConvolutionLayer Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "PatchConvolutionLayer takes a single blob as output.";
  
  CHECK_GT( KSIZE_, 0 );
  if (PAD_ >= 0) {
    // positive padding means extending the second image
    virtual_pad_ = max( 0, PAD_ - (KSIZE_-1) ); // virtual padding, handled by kernel
    PAD_ -= virtual_pad_;
    CROP_ = 0;
  } else {  
    // negative padding means cropping the first image
    CHECK_EQ( ngh_rad_, -PAD_ );
    CROP_ = -PAD_;  // CROP_ > 0
    virtual_pad_ = PAD_ = 0;
  }
  CHECK_GE( CROP_, 0 );
  CHECK_EQ( STRIDE_, 1) << "not implemented otherwise";
}
template <typename Dtype>
void PatchConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // image pairs are already concatenated on the channel dimension
  CHECK_EQ( channels_ % 2, 0) << "error: number of channels must be even";
  channels_ /= 2;
  CHECK_EQ( (height_ - 2*CROP_) % KSIZE_, 0 ) << "error: images height is not multiple of patch size";
  CHECK_EQ( (width_  - 2*CROP_) % KSIZE_, 0 ) << "error: images width is not multiple of patch size";
  
  // temporary matrix to host patches
  K_ = channels_ * KSIZE_ * KSIZE_;
  patches_.Reshape(1, K_, (height_ - 2*CROP_) / KSIZE_, (width_ - 2*CROP_) / KSIZE_);
  channels_out_ = patches_.height() * patches_.width();
  
  // temporary matrix for convolve-ready second image
  const int height_out = (height_ + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  const int width_out  = (width_  + 2 * PAD_ - KSIZE_) / STRIDE_ + 1;
  col_buffer_.Reshape(1, K_, height_out, width_out );
  
  if (ngh_rad_ < 0) { // correlation with full img1
    CHECK_EQ( PAD_, KSIZE_/2 );
    top[0]->Reshape(num_, channels_out_, height_out, width_out);
  } else {
    const int output_size = 2*ngh_rad_ + 1;
    CHECK_EQ( ngh_rad_, PAD_+virtual_pad_+CROP_ );
    top[0]->Reshape(num_, channels_out_, output_size, output_size);
    
    // temporary buffer to transfer results from dense to sparse 
    dense_buf_.Reshape(1, min(channels_out_, 128), height_out, width_out );
  }
  
  if (normalize_borders_ == 'd') {
    // create virtual image of ones for dynamic patch normalization
    Blob<Dtype> ones; // temp variable
    ones.Reshape(1, channels_, KSIZE_, KSIZE_);
    Dtype* ones_data = ones.mutable_cpu_data();
    for(int n=0; n<ones.count(); ++n) ones_data[n] = 1;  // fill it with ones
    const int onew_out = KSIZE_ / STRIDE_ + 1;  // because PAD_ == KSIZE_/2 in this case
    ones_col_.Reshape(1, K_, onew_out, onew_out);
    my_im2col_cpu( ones_data, channels_, KSIZE_, KSIZE_, KSIZE_, KSIZE_/2, 
                STRIDE_, ones_col_.mutable_cpu_data());
    // preallocate masks' memory
    masks_.Reshape( channels_out_, 1, onew_out, onew_out);
  }
}


/* Matrix-Vector product.
   It computes: y = alpha*A*x  + beta*y
   
   A is a MxN dense matrix (row-major or col-major order)
   x is a Nx1 vector
   y is a Mx1 vector
*/
template<typename Dtype>
void caffe_cpu_gemv_lda(const CBLAS_ORDER orderA, const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const Dtype alpha, const Dtype* A, int lda, const Dtype* x,
    const Dtype beta, Dtype* y);
template <>
void caffe_cpu_gemv_lda<float>(const CBLAS_ORDER orderA, const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, int lda, const float* x,
    const float beta, float* y) {
  cblas_sgemv(orderA, TransA, M, N, alpha, A, lda, x, 1, beta, y, 1);
}
template <>
void caffe_cpu_gemv_lda<double>(const CBLAS_ORDER orderA, const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, int lda, const double* x,
    const double beta, double* y) {
  cblas_dgemv(orderA, TransA, M, N, alpha, A, lda, x, 1, beta, y, 1);
}


template <typename Dtype>
void PatchConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int second = bottom[0]->offset(1)/2;  // offset for second image
  
  Dtype* patches_data = patches_.mutable_cpu_data();
  Dtype* col_data = col_buffer_.mutable_cpu_data();
  //const Dtype* offsets_data = this->blobs_[0]->cpu_data();
  
  const int M_ = channels_out_;
  const int N_ = col_buffer_.height() * col_buffer_.width();
  const int N_ONES_ = ones_col_.count() / K_; 
  const int mask_size = KSIZE_ / STRIDE_ + 1;
  const int height_out = top[0]->height();
  const int width_out = top[0]->width();
  const int grid_width = (width_ - 2*CROP_) / KSIZE_;
  const int true_height = bottom[0]->height()/STRIDE_ + 1;
  const int true_width  = bottom[0]->width() /STRIDE_ + 1;
  
  // go through the images
  for (int n = 0; n < num_; n++) {
    // first, transform first image into consecutive patches
    my_im2col_cpu_T( bottom_data + bottom[0]->offset(n), channels_, height_,
                width_, KSIZE_, -CROP_, KSIZE_, patches_data);
    // possible optimization: add a transpose kernel here instead of calling im2col_T
    // don't forget that if no transpose is used at all, CblasNoTrans => CblasTrans below
    
    // then, prepare second image for dense convolution
    my_im2col_cpu( bottom_data + bottom[0]->offset(n)+second, channels_, height_,
                width_, KSIZE_, PAD_, STRIDE_, col_data);
    
    // finally, perform dense dotproduct
    if (ngh_rad_ < 0 ) { 
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., patches_data, col_data,
              (Dtype)0, top_data + top[0]->offset(n));
    } else {
      // perform mutliple gemv
      for(int i=0; i<channels_out_; ++i) {
        // res_map offset for this patch in col_ coordinates
        const int lx = CROP_ + (i % grid_width) * KSIZE_; // top-left corner of patch
        const int ly = CROP_ + (i / grid_width) * KSIZE_;
        const int ox = lx + PAD_ - ngh_rad_; //int(offsets_data[2*i+0]) - KSIZE_/2 + PAD_;
        const int oy = ly + PAD_ - ngh_rad_;
        
        // some sanity checks
        CHECK_GE( ox+virtual_pad_, 0 ) << "problem for patch["<<i<<"] with (lx,ly) = ("<<lx<<","<<ly<<")";
        CHECK_GE( oy+virtual_pad_, 0 ) << "problem for patch["<<i<<"] with (lx,ly) = ("<<lx<<","<<ly<<")";
        CHECK_LE( ox+virtual_pad_ + width_out, col_buffer_.width()+2*virtual_pad_ ) 
          << "problem for patch["<<i<<"] with (ox,oy) = ("<<ox<<","<<oy<<")";
        CHECK_LE( oy+virtual_pad_ + height_out, col_buffer_.height()+2*virtual_pad_ ) 
          << "problem for patch["<<i<<"] with (ox,oy) = ("<<ox<<","<<oy<<")";
        
        // y = A.x, A is MxN matrix, x is a N-dim vector
        // here A = col.T
        for(int y=0; y<height_out; ++y)
          if( oy+y >= 0 && oy+y < col_buffer_.height() ) {
            int begx = max(ox, 0); // begining 
            int endx = min(ox+width_out, col_buffer_.width()); // ending
            caffe_cpu_gemv_lda(CblasColMajor, CblasNoTrans, endx-begx, K_,
                (Dtype)1., col_data + begx + (y+oy)*col_buffer_.width(), N_, patches_data + i*K_, 
                (Dtype)0., top_data + top[0]->offset(n, i, y, begx-ox) );
          }
      }
    }
    
    if (normalize_borders_ == 'd') {
      // dynamic normalization of image borders
      const Dtype* ones_data = ones_col_.cpu_data();
      Dtype* mask_data = masks_.mutable_cpu_data();
      
      caffe_sqr<Dtype>(patches_.count(), patches_data, patches_data); // square values
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_ONES_, K_,
              (Dtype)1., patches_data, ones_data,
              (Dtype)0, mask_data );  // sum on image borders
      caffe_powx<Dtype>(masks_.count(), mask_data, -1, mask_data); // compute invsere of squared norm
      if (ngh_rad_ < 0 ) { 
        rectify_border( channels_out_, height_out, width_out, mask_size, mask_data,
                top_data+ top[0]->offset(n), 
                top_data+ top[0]->offset(n) );  // mutliply borders accordingly
      } else {
        rectify_border_offsets( channels_out_, true_height, true_width, 
                height_out, width_out, grid_width, 
                ngh_rad_, KSIZE_, mask_size, CROP_, mask_data, 
                top_data+ top[0]->offset(n), 
                top_data+ top[0]->offset(n) );  // mutliply borders accordingly
      }
    }
  }
  
  if (normalize_borders_ == 's') {
    if (ngh_rad_<0) {
      const long total = top[0]->count();
      rectify_border_fixed(total, height_out, width_out, KSIZE_, mask_size, top_data );
    } else {
      CHECK_EQ( width_out, height_out );
      CHECK_EQ( STRIDE_, 1 );
      rectify_border_fixed_offsets<Dtype>( 
              num_ * channels_out_, true_height, true_width, 
              height_out, width_out, grid_width, 
              ngh_rad_, KSIZE_, mask_size, CROP_, top_data, top_data);
    }
  }
}

INSTANTIATE_CLASS(PatchConvolutionLayer);


  ////////////////////////////////////////////////////////////
  //                DeepMatching ArgMax layer              //
  //////////////////////////////////////////////////////////

template <typename Dtype>
void DeepMatchingArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 
  CHECK_EQ(bottom.size(), 1) << "Conv Layer takes a single blob as input.";
  CHECK_EQ(top.size(), 1) << "Conv Layer takes a single blob as output.";
  
  CHECK_EQ( (step_/4)*4, step_ ) << "step must be a multiple of 4 (otherwise retrieve_patch() won't work)";
  
  CHECK_GT( img_width_, step_ );
  CHECK_GT( img_height_, step_ );
  CHECK_EQ( img_width_ % step_, 0 );
  CHECK_EQ( img_height_ % step_, 0 );
  CHECK_GE( ngh_rad_, -1 );
}
template <typename Dtype>
void DeepMatchingArgMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 
  const int num_ = bottom[0]->num();
  const int output_height = (img_height_ + step_ - 1) / step_;
  const int output_width = (img_width_ + step_ - 1) / step_;
  
  top[0]->Reshape(num_, 2*6, output_height, output_width);
}

// retrieve patch at pixel position (x,y)
// offset = step/2 if level==0 else 0
static inline int retrieve_patch( const int x, const int y, const int step, const int offset,
                                  const int grid_width, const int grid_height ) {
  const int i = (x - offset) / step;
  const int j = (y - offset) / step;
  CHECK_EQ( x, i*step + offset ) << "error: child_grid does not match current grid";
  CHECK_EQ( y, j*step + offset ) << "error: child_grid does not match current grid";
  if( i<0 || i>=grid_width )  return -1;
  if( j<0 || j>=grid_height )  return -1;
  return i + j * grid_width;
}

// find best score in a 3x3 local window around (i,j)
template <typename Dtype>
inline Dtype local_argmax(const Dtype* data, int* j, int* i, int height, int width) {
  const int mini = max(0, *i-1); 
  const int maxi = min(*i+2, width);
  const int minj = max(0, *j-1); 
  const int maxj = min(*j+2, height);
  
  Dtype res = 0;
  int besti, bestj;
  for(int v=minj; v<maxj; ++v)
    for(int u=mini; u<maxi; ++u) {
      Dtype s = data[v*width + u];
      if (s>res)  {res=s; besti=u; bestj=v;}
    }
  *i = besti;
  *j = bestj;
  return res;
}

const int bigprime = 1000003;
static int n_bottom = 0;

template <typename Dtype>
inline void write_corres( const int n, const int c, int i1, int j1, const Dtype score, 
                          const int img_width, const int img_height,
                          const int grid_width, const int grid_height, const int step,
                          const int nghrad, bool use_nghrad, void* _corres ) {
  
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
    
    // goto correct image
    Dtype* corres = ((Dtype*)_corres) + n * 12 * grid_height * grid_width;  
    
    Dtype* corres0 = corres + 6*(qj0*grid_width + qi0);
    if (corres0[4] < score ) {
      corres0[0] = i0;
      corres0[1] = j0;
      corres0[2] = i1;
      corres0[3] = j1;
      corres0[4] = score;
      corres0[5] = 0; // unused
    }
    Dtype* corres1 = corres + 6*((grid_height + qj1)*grid_width + qi1);
    if (corres1[4] < score ) {
      corres1[0] = i0;
      corres1[1] = j0;
      corres1[2] = i1;
      corres1[3] = j1;
      corres1[4] = score;
      corres1[5] = 0; // unused
    }
  }
}

template <typename Dtype, bool use_nghrad>
inline void argmax_correspondences_rec( const int num, const Blob<Dtype>** pyr, const int top_level,
                  const int level, const int c, int j1, int i1, Dtype score, Dtype* hashtable_scores,
                  Dtype* corres, const int img_width, const int img_height, const int step, const int nghrad ) {
  const Dtype* data = pyr[level]->cpu_data() + pyr[level]->offset(num, c);
  CHECK_GE(c, 0); CHECK_LT(c, pyr[level]->channels());
  const int height = pyr[level]->height();
  const int width = pyr[level]->width();
  CHECK_EQ(width, height);
  CHECK_EQ(width % 2, 1);
  
  // first, do a local argmax to refine the current position
  Dtype s = (level==top_level) ? data[ i1 + j1*width ] :
            local_argmax(data, &j1, &i1, height, width);
  if(!s)  return; // bad coordinates
  score += s;
  
  // check if better path has already been established at this place 
  const long offset = level*bigprime + (pyr[level]->offset(0,c,j1,i1) % bigprime);
  bool not_better = (score <= hashtable_scores[offset]);
  if(not_better) return; // this maximum was already investigated with a better score
  else hashtable_scores[offset] = score;  // set new record
  
  // coordinates (center of patch) in first image (due to the regular grid)
  const int g = (level>=1); // grid is one pixel wider from level>=1
  const int grid_width =  img_width  / step + g;
  const int grid_height = img_height / step + g;
  const int i0 = (level?0:step/2) + (c%grid_width)*step;
  const int j0 = (level?0:step/2) + (c/grid_width)*step;
  
  if (level) { // pyramid level >= 1
    
    // recursive calls
    const int lower_grid_width  = grid_width  - (level<=1);
    const int lower_grid_height = grid_height - (level<=1);
    const int lower_grid_offset = level==1? step/2 : 0;
    for(int ch=0; ch<4; ch++) {
      // find children of this patch
      const int u = 2*(ch%2) - 1;  // in {-1,1}
      const int v = 2*(ch/2) - 1;  // in {-1,1}
      int child = retrieve_patch( i0 + u*((step/4)<<level), j0 + v*((step/4)<<level), 
                                  step, lower_grid_offset, lower_grid_width, lower_grid_height );
      
      if (child>=0)
        // position of children in child1 = parent1 - (parent0-child0)
        argmax_correspondences_rec<Dtype, use_nghrad>( num, pyr, top_level, level-1, child, 
                                    use_nghrad ? 2*j1 : 2*(j1+v), use_nghrad ? 2*i1 : 2*(i1+u), 
                                    score, hashtable_scores, corres, img_width, img_height, step, nghrad );
    }
    
  } else {
    write_corres( num, c, i1, j1, score, img_width, img_height, grid_width, grid_height, 
                  step, nghrad, use_nghrad, corres );
  }
}


template <typename Dtype>
void DeepMatchingArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) { 
  CHECK_EQ( bottom[0], this->blobs_.back().get()) << "error: previous activation_blobs must be supplied manually";
  CHECK_EQ( top[0]->height()*step_, img_height_ ) << "error: corres0 and corres1 height must mutiple of step";
  CHECK_EQ( top[0]->width()*step_, img_width_ ) << "error: corres0 and corres1 width must mutiple of step";
  CHECK_EQ( img_width_*img_height_, step_*step_*this->blobs_[0]->channels() ) << "error: grid doesn't match lowest pyramid blob";
  
  Dtype* corres = top[0]->mutable_cpu_data();
  memset( corres, 0, top[0]->count()*sizeof(Dtype) );  // init to 0
  
  // gather all activation_blobs in a simple vector
  const int n_levels = this->blobs_.size();
  
  const Blob<Dtype>* blob_pyr[10];  // faster access than shared_ptrs
  for(int i=0; i<n_levels; ++i)  blob_pyr[i] = this->blobs_[i].get();
  
  // hashtable to avoid redundant computations (it stores best paths in a compressed representations)
  SyncedMemory hashed_scores( n_levels * bigprime * sizeof(Dtype) );
  Dtype* hashtable_data = reinterpret_cast<Dtype*>(hashed_scores.mutable_cpu_data());
  
  for(int n=0; n<bottom[0]->num(); ++n) {
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int level = n_levels - 1;
    
    // reset hash table
    memset(hashtable_data, 0, hashed_scores.size());
    
    if(ngh_rad_<0) {
      const bool use_nghrad = false;
      for(int c=0; c<channels; ++c)
        for(int j=0; j<height; ++j) 
          for(int i=0; i<width; ++i) {
            //LOG(INFO)<<"ready";getchar();
            argmax_correspondences_rec<Dtype, use_nghrad>( n, blob_pyr, level, level, c, j, i, Dtype(0), hashtable_data,
                                                      corres, img_width_, img_height_, step_, ngh_rad_);
          }
    } else {
      const bool use_nghrad = true;
      for(int c=0; c<channels; ++c)
        for(int j=0; j<height; ++j) 
          for(int i=0; i<width; ++i) {
            //LOG(INFO)<<"ready";getchar();
            argmax_correspondences_rec<Dtype, use_nghrad>( n, blob_pyr, level, level, c, j, i, Dtype(0), hashtable_data,
                                                      corres, img_width_, img_height_, step_, ngh_rad_);
          }
    }
  }
}

//INSTANTIATE_CLASS(DeepMatchingArgMaxLayer);
template class DeepMatchingArgMaxLayer<float>;


}  // namespace caffe




















