#ifndef __MY_COL2IM_H__
#define __MY_COL2IM_H__

/* just some wrappers to map to the "old" versions of im2col functions in caffe
*/

namespace caffe {

template <typename Dtype>
void my_im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad, 
    const int stride, Dtype* data_col);

// Transposed version (output = channels, num, height, width)
template <typename Dtype>
void my_im2col_cpu_T(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad, 
    const int stride, Dtype* data_col);


template <typename Dtype>
void my_im2col_gpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad, 
    const int stride, Dtype* data_col);

// Transposed version (output = channels, num, height, width)
template <typename Dtype>
void my_im2col_gpu_T(const Dtype* data_im, const int channels,
    const int height, const int width, const int ksize, const int pad, 
    const int stride, Dtype* data_col);

}  // namespace caffe

#endif























