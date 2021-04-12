// Copyright 2013 Yangqing Jia

//#include <algorithm>
//#include <cmath>
//#include <cstdlib>
//#include <cstring>

#include "caffe/common.hpp"
#include "my_im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void my_im2col_gpu_kernel(const int n, const Dtype* data_im,
    const int height, const int width, const int ksize, const int pad,
    const int stride, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(_index, n) {
    int index = _index;
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && w < width && h < height) ?
            data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

template <typename Dtype>
void my_im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  my_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, height, width, ksize, pad, stride, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void my_im2col_gpu<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void my_im2col_gpu<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);


template <typename Dtype>
__global__ void my_im2col_gpu_kernel_T(const int n, const Dtype* data_im,
    const int channels, const int height, const int width, const int ksize, const int pad,
    const int stride, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(_index, n) {
    int index = _index;
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize * ksize;
    int channels_col = channels * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    data_col += channel_out + channels_col * (w_out + width_col * h_out);
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && w < width && h < height) ?
            data_im[i * width + j] : 0;
        data_col++;
      }
    }
  }
}

template <typename Dtype>
void my_im2col_gpu_T(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  my_im2col_gpu_kernel_T<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, channels, height, width, ksize, pad, stride, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void my_im2col_gpu_T<float>(const float* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void my_im2col_gpu_T<double>(const double* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);

}  // namespace caffe
