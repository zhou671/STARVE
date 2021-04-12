"""
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
"""
import pdb
import numpy as np
import gpudm

###########################################
# basic functions for blob <-> numpy transfer

def set_GPU(GPU=0):
  ''' if GPU == -1 => use CPU
      otherwise GPU device number
  '''
  if GPU>=0:
    gpudm.Caffe_set_mode(gpudm.Caffe.GPU)
    gpudm.Caffe.SetDevice(GPU)
  else:
    gpudm.Caffe_set_mode(gpudm.Caffe.CPU)


def blob_shape(blob):
  return (blob.num(), blob.channels(), blob.height(), blob.width())

def set_blob_to(blob, a):
  assert a.shape == blob_shape(blob), pdb.set_trace()
  assert a.flags.c_contiguous
  b = gpudm.floats_to_numpy_ref(blob.mutable_cpu_data(), a.size)
  b.ravel()[:] = a.ravel()[:]



###########################################
# Deep Network

class Net (object):
  """ a neural net. Simplification of the Net in C++ version """
  
  def __init__(self, input_blob):
    if type(input_blob) == tuple:
      # just a size
      size = input_blob
      input_blob = gpudm.BlobFloat()
      input_blob.Reshape(size[0], size[1], size[2], size[3])
    self.input_blob = input_blob
    self.activation_blobs = [("data", input_blob)]
    self.layers = []

  @staticmethod
  def new_BlobPtrVector(*v):
    bv = gpudm.BlobPtrVector()
    for x in v:
      if x is not None:
        bv.push_back(x)
    return bv

  @staticmethod
  def set_filler_params(dest, src):
    for k, v in src.items():
      getattr(dest, "set_" + k)(v)


  ####################################################3
  # Training and testing 

  def forward(self, start=0, end=0):
    end = end or len(self.layers)
    for i, (layer_name, layer) in enumerate(self.layers[start:end],start):
      bottom = self.activation_blobs[i][1]
      top = self.activation_blobs[i + 1][1]
      if top is not None: 
        layer.Forward(self.new_BlobPtrVector(bottom),
                      self.new_BlobPtrVector(top))
      else:
        # last layer for train
        layer.Forward(self.new_BlobPtrVector(bottom, self.labels_blob),
                      self.new_BlobPtrVector())

  def test(self, input_data = None):
    if input_data is not None:
      self.input_blob.mutable_to_numpy_ref()[:] = input_data
    self.forward()
    res = self.activation_blobs[-1][1]
    return res.to_numpy_ref()


  ####################################################3
  # I/O     

  def set_parameters(self, params, verbose=0, create_blobs=False):
    for layer_name, layer in self.layers:
      if layer_name in params:
        layer_blobs = layer.blobs()
        if create_blobs and len(layer_blobs)==0:  
          for i, npdata in enumerate(params[layer_name]):
            assert npdata.ndim == 4, 'error: paramter is not a blob'
            blob = gpudm.BlobFloat(*npdata.shape)
            blob.mutable_to_numpy_ref()[:] = npdata
            layer_blobs.push_back(blob)
        elif len(layer_blobs) > 0: 
          assert len(params[layer_name]) == len(layer_blobs), "expected %d blobs for layer %s, received %d" % (len(layer_blobs), layer_name, len(params[layer_name]))
          for i, npdata in enumerate(params[layer_name]):
            #print 'Setting param #%d in layer %s' % (i, layer_name)
            if type(npdata) == np.ndarray:
              blob = layer_blobs[i].mutable_to_numpy_ref()
              assert blob.shape == npdata.shape, "Error: parameters shapes differ: blob=%s vs weights=%s" %(
                      str(blob.shape), str(npdata.shape))
              blob[:] = npdata
            else:
              set_blob_to(layer_blobs[i], npdata)
      elif verbose:
        print "Layer %s not found in parameters"%(layer_name)

  def get_parameters(self, diff=False):
    params = {}
    for layer_name, layer in self.layers:
      layer_params = params[layer_name] = []
      if diff:  diff_params = params[layer_name+'_diff'] = []
      for blob in layer.blobs(): 
        layer_params.append(blob.to_numpy_ref().copy())
        if diff:  diff_params.append(blob.diff_to_numpy_ref().copy())
    return params

  def describe(self):
    print "input:", self.input_blob.get_shape()
    for i, (layer_name, layer) in enumerate(self.layers):
      print "layer", layer_name, layer.__class__
      print "  parameters:"
      for j, blob in enumerate(layer.blobs()):
        print "    ", blob.get_shape()
      bottom_name, bottom = self.activation_blobs[i]
      top_name, top = self.activation_blobs[i + 1]
      print "  in:", bottom_name, bottom.get_shape()
      print "  out:", top_name, top.get_shape()


  ####################################################3
  # Build layers

  def check_name(self, name):
    used = set(zip(*([(None,None)]+self.layers))[0])
    assert name not in used, 'error: layer name already used: '+name

  def get_layer(self, name, ret_index=False, list=False, layers=None, re=False):
    rname = name if re else name.replace('*','.*')
    import re
    res = []
    for i,(n,l) in enumerate(layers or self.layers):
      if re.match(rname,n):
        r = i if ret_index else l
        if list:
          res.append(r)
        else:
          return r
    assert list or res, "error: no layer matching '%s' found"%name
    return res

  def get_activation(self, name, **kwargs):
    return self.get_layer(name, layers=self.activation_blobs, **kwargs)

  def add_layer(self, name, layer_class, args, inplace=False): 
    self.check_name(name)
    # single output
    if inplace:
      top_blob = self.activation_blobs[-1][1]
      if hasattr(top_blob,'ShareData'):
        blob = gpudm.BlobFloat(*blob_shape(top_blob))
        blob.ShareData(top_blob)
        blob.ShareDiff(top_blob)
        top_blob = blob
    else:
      top_blob = gpudm.BlobFloat()
    if type(args) is not tuple: args = (args,)
    
    layer = layer_class(*args)
    self.layers.append((name, layer))
    
    layer.SetUp(self.new_BlobPtrVector(self.activation_blobs[-1][1]), self.new_BlobPtrVector(top_blob))
    
    self.activation_blobs.append((name, top_blob))    

  def add_convolution(self, name,                      
                      kernelsize,
                      num_output,
                      stride = 1, 
                      group = 1,
                      pad = 0,
                      biasterm = True,
                      weight_filler = {},
                      bias_filler = {}, 
                      weights=None):
    cp = gpudm.ConvolutionParameter()
    cp.set_kernel_h(kernelsize)
    cp.set_kernel_w(kernelsize)
    cp.set_num_output(num_output)
    assert stride>0, 'error: stride is 0 for '+name
    cp.set_stride_h(stride )
    cp.set_stride_w(stride )
    cp.set_group(group)
    cp.set_pad_h(pad)
    cp.set_pad_w(pad)
    cp.set_bias_term(biasterm)
    self.set_filler_params(cp.mutable_weight_filler(), weight_filler)
    self.set_filler_params(cp.mutable_bias_filler(), bias_filler)
    lp = gpudm.LayerParameter()
    lp.set_allocated_convolution_param(cp)
    cp.this.disown()  # otherwise it will be freed 2 times
    
    self.add_layer(name, gpudm.ConvolutionLayerFloat, lp)
    if weights: self.set_parameters({name:weights}, verbose=0)

  def add_border_rectification(self, name, 
                          kernelsize, 
                          weights,
                          inplace = True ):
    args = (gpudm.LayerParameter(), kernelsize)
    self.add_layer(name, gpudm.BorderRectifyLayerFloat, args, inplace=inplace)
    self.set_parameters({name:weights}, verbose=0) # computed based on a convolution mask

  def add_relu(self, name, inplace = True ):
    lp = gpudm.LayerParameter()
    self.add_layer(name, gpudm.ReLULayerFloat, lp, inplace=inplace)

  def add_rectified_sigmoid(self, name):
    lp = gpudm.LayerParameter()
    self.add_layer(name, gpudm.RectifiedSigmoidLayerFloat, lp)

  def add_pixel_norm(self, name, norm=1.0, inplace=True):
    lp = gpudm.LayerParameter()
    self.add_layer(name, gpudm.PixelNormLayerFloat, (lp, norm), inplace=inplace)

  def add_reshape_layer(self, name, dims, inplace=True):
    shape = gpudm.BlobShape()
    for d in dims:
      shape.add_dim(d)
    rp = gpudm.ReshapeParameter()
    rp.set_allocated_shape(shape)
    shape.this.disown()  # otherwise it will be freed 2 times
    lp = gpudm.LayerParameter()
    lp.set_allocated_reshape_param(rp)
    rp.this.disown()  # otherwise it will be freed 2 times
    self.add_layer(name, gpudm.ReshapeLayerFloat, lp, inplace=inplace)

  def add_patch_correlation(self, name, kernelsize = 1, 
                                  pad = None,
                                  nghrad = -1,
                                  normalize_borders = 'dynamic' ):
    if nghrad>=0 and pad is None: pad = nghrad  # smart default
    lp = gpudm.LayerParameter()
    norm_modes = {'dynamic':'d', 'd':'d', 'static':'s', 's':'s', 'none':0}
    self.add_layer(name, gpudm.PatchConvolutionLayerFloat, 
                    (lp, kernelsize, pad, nghrad, norm_modes[normalize_borders]) )

  def add_power_law(self, name, power, scale=1, shift=0, inplace=False):
    pp = gpudm.PowerParameter()
    pp.set_power(power)
    pp.set_scale(scale)
    pp.set_shift(shift)
    lp = gpudm.LayerParameter()
    lp.set_allocated_power_param(pp)
    pp.this.disown()  # otherwise it will be freed 2 times
    self.add_layer(name, gpudm.PowerLayerFloat, lp, inplace=inplace)

  # pool methods
  MAX = gpudm.PoolingParameter_PoolMethod_MAX
  AVE = gpudm.PoolingParameter_PoolMethod_AVE
  STOCHASTIC = gpudm.PoolingParameter_PoolMethod_STOCHASTIC
  
  def add_pooling(self, name,
                  kernelsize,
                  stride = 1,
                  pool = MAX,
                  pad = 0):
    if kernelsize=='full': 
      last_blob = blob_shape(self.activation_blobs[-1][1])
      assert last_blob[2] == last_blob[3]
      kernelsize = last_blob[-1]
    pp = gpudm.PoolingParameter()
    pp.set_kernel_h(kernelsize)
    pp.set_kernel_w(kernelsize)
    pp.set_stride_h(stride)
    pp.set_stride_w(stride)
    pp.set_pad_h(pad)
    pp.set_pad_w(pad)
    pp.set_pool(pool)
    lp = gpudm.LayerParameter()
    lp.set_allocated_pooling_param(pp)
    pp.this.disown()  # otherwise it will be freed 2 times
    self.add_layer(name, gpudm.PoolingLayerFloat, lp)

  def add_sparse_convolution(self, name, sp_pattern, 
                                  use_sp_data = True,
                                  kernelsize = 1, 
                                  stride = 1, 
                                  pad = 0,
                                  biasterm = False,
                                  weight_filler = {},
                                  bias_filler = {}, 
                                  blobs_lr=[1.0,2.0],
                                  weight_decays=[1.0,1.0],
                                  weights = None ):
    from scipy import sparse
    assert sparse.isspmatrix(sp_pattern)
    num_output = sp_pattern.shape[0]  # sparsity pattern is given in input
    
    cp = gpudm.ConvolutionParameter()
    cp.set_kernel_h(kernelsize)
    cp.set_kernel_w(kernelsize)
    cp.set_num_output(num_output)
    assert stride>0, 'error: stride is 0 for '+name
    cp.set_stride_h(stride)
    cp.set_stride_w(stride)
    cp.set_pad_h(pad)
    cp.set_pad_w(pad)
    cp.set_bias_term(biasterm)
    self.set_filler_params(cp.mutable_weight_filler(), weight_filler)
    self.set_filler_params(cp.mutable_bias_filler(), bias_filler)
    lp = gpudm.LayerParameter()
    lp.set_allocated_convolution_param(cp)
    cp.this.disown()  # otherwise it will be freed 2 times
    
    def arrToBlob(arr): # dirty function but simpler for now
        bb = gpudm.BlobFloat(1,1,1,arr.size)
        bb.mutable_to_numpy_ref().view(arr.dtype)[:] = arr.ravel()
        return bb
    
    if sparse.isspmatrix_csr(sp_pattern):
      sparsity_args = (sp_pattern.nnz, 
                       arrToBlob(sp_pattern.indptr), arrToBlob(sp_pattern.indices), 
                       arrToBlob(sp_pattern.data) if use_sp_data else None)
      self.add_layer(name, gpudm.CSR_SparseConvolutionLayerFloat, lp)
    elif sparse.isspmatrix_bsr(sp_pattern):
      br,bc = sp_pattern.blocksize
      assert br == bc, "error: not implemented for non-square blocks"
      sparsity_args = (sp_pattern.nnz/(br*bc), br,
                       arrToBlob(sp_pattern.indptr), arrToBlob(sp_pattern.indices), 
                       arrToBlob(sp_pattern.data) if use_sp_data else None)
      self.add_layer(name, gpudm.BSR_SparseConvolutionLayerFloat, lp)
    else:
      assert False, "This sparse matrix type is not implemented"
    
    # define sparsity pattern now
    self.layers[-1][1].SetSparsityPattern( *sparsity_args )
    
    self.layers[-1][1].blobs_lr = blobs_lr
    self.layers[-1][1].weight_decays = weight_decays
    if weights: self.set_parameters({name:weights}, verbose=0)

  def add_dm_argmax(self, name, shape, nlevels, nghrad, tag='pow%d', step=4):
    lp = gpudm.LayerParameter()
    self.add_layer(name, gpudm.DeepMatchingArgMaxLayerFloat, 
                    (lp, shape[0], shape[1], step, nghrad) )
    
    # append activation blobs of previous layers
    blobs = self.layers[-1][1].blobs()
    for layer_name,activation_blob in self.activation_blobs:
      if layer_name ==  tag%len(blobs):
        blobs.push_back(activation_blob)
    assert blobs.size() == nlevels
















