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
#ifndef __EXTA_LAYERS_H__
#define __EXTA_LAYERS_H__

#include "caffe/layer.hpp"
//#include "caffe/vision_layers.hpp"

#include "caffe/layers/absval_layer.hpp"
#include "caffe/layers/accuracy_layer.hpp"
#include "caffe/layers/argmax_layer.hpp"
#include "caffe/layers/base_conv_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/layers/batch_reindex_layer.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/layers/bnll_layer.hpp"
//#include "caffe/layers/clip_layer.hpp"
#include "caffe/layers/concat_layer.hpp"
#include "caffe/layers/contrastive_loss_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/crop_layer.hpp"
#include "caffe/layers/cudnn_conv_layer.hpp"
//#include "caffe/layers/cudnn_deconv_layer.hpp"
#include "caffe/layers/cudnn_lcn_layer.hpp"
#include "caffe/layers/cudnn_lrn_layer.hpp"
#include "caffe/layers/cudnn_pooling_layer.hpp"
#include "caffe/layers/cudnn_relu_layer.hpp"
#include "caffe/layers/cudnn_sigmoid_layer.hpp"
#include "caffe/layers/cudnn_softmax_layer.hpp"
#include "caffe/layers/cudnn_tanh_layer.hpp"
#include "caffe/layers/data_layer.hpp"
#include "caffe/layers/deconv_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/dummy_data_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/elu_layer.hpp"
#include "caffe/layers/embed_layer.hpp"
#include "caffe/layers/euclidean_loss_layer.hpp"
#include "caffe/layers/exp_layer.hpp"
#include "caffe/layers/filter_layer.hpp"
#include "caffe/layers/flatten_layer.hpp"
#include "caffe/layers/hdf5_data_layer.hpp"
#include "caffe/layers/hdf5_output_layer.hpp"
#include "caffe/layers/hinge_loss_layer.hpp"
#include "caffe/layers/im2col_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/layers/infogain_loss_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/log_layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/lstm_layer.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/layers/multinomial_logistic_loss_layer.hpp"
#include "caffe/layers/mvn_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/layers/parameter_layer.hpp"
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/prelu_layer.hpp"
#include "caffe/layers/python_layer.hpp"
#include "caffe/layers/recurrent_layer.hpp"
#include "caffe/layers/reduction_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include "caffe/layers/reshape_layer.hpp"
#include "caffe/layers/rnn_layer.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/layers/sigmoid_layer.hpp"
#include "caffe/layers/silence_layer.hpp"
#include "caffe/layers/slice_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/layers/split_layer.hpp"
#include "caffe/layers/spp_layer.hpp"
//#include "caffe/layers/swish_layer.hpp"
#include "caffe/layers/tanh_layer.hpp"
#include "caffe/layers/threshold_layer.hpp"
#include "caffe/layers/tile_layer.hpp"
#include "caffe/layers/window_data_layer.hpp"

#include "caffe/proto/caffe.pb.h"
#include <stdio.h>

#include "cusparse.h"
#define CUSPARSE_CHECK(condition) CHECK_EQ((condition), CUSPARSE_STATUS_SUCCESS)

#define recast reinterpret_cast

namespace caffe {

  ////////////////////////////////////////////////////////////
  //        CONVOLUTION WITH CSR-SPARSE WEIGHTS            //
  //////////////////////////////////////////////////////////

  class CusparseHandle {
   public:
    static cusparseHandle_t cusparse_handle() {
      if( instance_.cusparse_handle_ == NULL ) { // not yet initalized
        CUSPARSE_CHECK(cusparseCreate(&instance_.cusparse_handle_));
      }
      return instance_.cusparse_handle_; 
    }
    
   private:
    static CusparseHandle instance_;
    cusparseHandle_t cusparse_handle_;
    
    CusparseHandle() : 
      cusparse_handle_(NULL) {}
    ~CusparseHandle() {
      if (cusparse_handle_)
        CUSPARSE_CHECK(cusparseDestroy(cusparse_handle_));
    }
  };

  /* Convolution layer but with sparse filters.
     This layer computes:   Out = Filters x im2col( In )
     
     In and Out are dense matrices (NCHW fully-packed).
     Filters are given as a CSR matrix instead as dense matrix.
  */
  template <typename Dtype>
  class CSR_SparseConvolutionLayer : public Layer<Dtype> {
   public:
    explicit CSR_SparseConvolutionLayer(const LayerParameter& param )
        : Layer<Dtype>(param) {}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    
    // Blob<float> is cast to Blob<int> 
    void SetSparsityPattern( const int nnz, 
        const Blob<float>* row_ptr, const Blob<float>* col_ind, const Blob<float>* data = NULL );
    
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
      {NOT_IMPLEMENTED;}
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    
    int KSIZE_, STRIDE_, PAD_, NUM_OUTPUT_;
    int NUM_, CHANNELS_, HEIGHT_, WIDTH_;
    bool biasterm_;
    int M_, N_, K_; // matrix multiplication sizes
    Blob<Dtype> col_buffer_;
    
    int nnz_; // number of non-zero elements
    shared_ptr<SyncedMemory> weight_rowptr_, weight_colind_;
    
    void load_csr_gpu( const int** weight_rowptr, const int** weight_colind, 
                       const Dtype** weight_data, Dtype** weight_diff );
    
    bool do_im2col_;
    shared_ptr<SyncedMemory> bias_multiplier_, transpose_res_;
  };


  ////////////////////////////////////////////////////////////
  //               BORDER RECTIFICATION                    //
  //////////////////////////////////////////////////////////
  
  template <typename Dtype>
  class BorderRectifyLayer : public Layer<Dtype> {
   public:
    explicit BorderRectifyLayer(const LayerParameter& param, int ksize)
        : Layer<Dtype>(param), KSIZE_(ksize) {}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    
    const int KSIZE_;
  };


  ////////////////////////////////////////////////////////////
  //                   RECTIFIED SIGMOID                   //
  //////////////////////////////////////////////////////////

  template <typename Dtype>
  class RectifiedSigmoidLayer : public NeuronLayer<Dtype> {
   public:
    explicit RectifiedSigmoidLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}
    
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
  };


  ////////////////////////////////////////////////////////////
  //                PIXEL NORMALIZATION                    //
  //////////////////////////////////////////////////////////

  template <typename Dtype>
  class PixelNormLayer : public NeuronLayer<Dtype> {
   public:
    explicit PixelNormLayer(const LayerParameter& param, Dtype norm)
        : NeuronLayer<Dtype>(param), alpha_(norm) {}
    
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    
    const Dtype alpha_;
  };


  ////////////////////////////////////////////////////////////
  //                  PATCH CONVOLUTION                    //
  //////////////////////////////////////////////////////////

  template <typename Dtype>
  class PatchConvolutionLayer : public Layer<Dtype> {
   public:
    explicit PatchConvolutionLayer(const LayerParameter& param, int ksize, int pad, int ngh_rad, char normalize_borders)
        : Layer<Dtype>(param), KSIZE_(ksize), PAD_(pad), STRIDE_(1), ngh_rad_(ngh_rad), normalize_borders_(normalize_borders) {}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    
    const char normalize_borders_;  // if 'd': general case (dynamic) border normalization
                                    // if 's': static border normalization => faster 
                                    //         (DeepMatching-like with normalized pixel descriptors)
    int num_, channels_, height_, width_;
    int channels_out_;
    const int KSIZE_, STRIDE_;  
    int PAD_; // if PAD_ is negative, then it is applied to the first image (subsampling center patches only)
    int K_;
    const int ngh_rad_; // 0 if no offsets, else 2*nghrad+1
    int virtual_pad_; // PAD_ + virtual_pad_ = theoretical_pad
    int CROP_; // crop of the first image in case of negative padding
    
    Blob<Dtype> patches_;     // first image transformed into consecutive patches
    Blob<Dtype> col_buffer_;  // temporary matrix for convolve-ready second image (4x4 bigger)
    Blob<Dtype> ones_col_;
    Blob<Dtype> masks_;
    Blob<Dtype> dense_buf_;   // buffer to transfer results from dense GEMM to sparse nghrad
  };


  ////////////////////////////////////////////////////////////
  //                DeepMatching ArgMax layer              //
  //////////////////////////////////////////////////////////

  template <typename Dtype>
  class DeepMatchingArgMaxLayer : public Layer<Dtype> {
   public:
    explicit DeepMatchingArgMaxLayer(const LayerParameter& param, 
                   const int img_height, const int img_width, const int step, const int ngh_rad)
        : Layer<Dtype>(param), img_height_(img_height), img_width_(img_width), step_(step),
          ngh_rad_(ngh_rad) {}
    
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    
   protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
      {NOT_IMPLEMENTED;}
    
    const int img_height_, img_width_, step_, ngh_rad_;
  };



};



#endif


























