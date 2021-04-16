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
import os, sys, pdb
from PIL import Image
from numpy import *
from scipy import sparse
try:
  from matplotlib.pyplot import *
  ion()
except:
  pass

import gpudm, helper
from net import set_GPU, Net, blob_shape


def pipeline_pixel_desc( net, 
                         presmooth_sigma = 1.0,
                         hog_sigmoid = 0.2,
                         mid_smoothing = 1.5,
                         post_smoothing = 1.0,
                         ninth_dim = 0.3,
                         extend = None, 
                         **kwargs ):
    
    def kernel_grid(ks):
      return mgrid[:ks,:ks] - ks/2
    gaussian_kernel = lambda dist, smooth_sigma: exp( -0.5* (dist/smooth_sigma)**2 )
    def smooth_mask(ks, smooth_sigma):
      assert (ks%2==1) and gaussian_kernel(1+ks/2,smooth_sigma) < 0.1, 'kernel is too small'
      mask = gaussian_kernel(sqrt(sum(kernel_grid(ks)**2, axis=0)), smooth_sigma)
      mask /= mask.sum()
      return mask
    
    def convol( mask, pad=0 ):
      """ compute the convolution mask normalization
      """
      from scipy.signal import correlate
      ks = mask.shape[-1]
      pad = pad or ks/2
      res = correlate(ones(mask.shape),mask, mode='full')
      if res.ndim==3: # don't care about correlation in z direction
        res = res[res.shape[0]/2] 
      assert abs(res[res.shape[1]/2,res.shape[0]/2]-1)<1e-6, pdb.set_trace()
      res[res.shape[1]/2,res.shape[0]/2] = 1  # force exactly 1
      h = ks-1-pad
      return res[h:-h, h:-h]
    
    # convert rgb to gray
    net.add_convolution('pxl_rgb2gray', num_output=1, kernelsize=1, biasterm=False,
                        weights=(ones((1,3,1,1))/3.0,))
    
    # pre-smooth image
    assert presmooth_sigma>=0
    if presmooth_sigma:
      ks = 5
      mask = smooth_mask(ks, presmooth_sigma / 0.996)
      net.add_convolution('pxl_smooth_pre', num_output=1, kernelsize=ks, pad=1+ks/2, biasterm=False,
                          weights=(mask.reshape(1,1,ks,ks),) )
      # we purposedly increase the image size, so that the next step 'pxl_grad' poses no problem
      net.add_border_rectification('pxl_smooth_pre_border', kernelsize=ks+2, 
                          weights=(1/convol(mask, pad=1+ks/2).reshape(1,1,ks+2,ks+2),) )
    
    # extract HOG
    net.add_convolution('pxl_grad', num_output=2, kernelsize=3, biasterm=False, pad=0, #pad_mode='nearest',
                        weights = (array([(((0,0,0),(-1,0,1),(0,0,0)),),(((0,-1,0),(0,0,0),(0,1,0)),)]),) )
    # prepare ninth_dim such that after going through sigmoid it gets the correct value
    inv_ninth = -log(2/(1+ninth_dim)-1)/hog_sigmoid if hog_sigmoid else ninth_dim
    net.add_convolution('pxl_hog', num_output=9, kernelsize=1, biasterm=True, # add ninth dim here
                        weights = (array([(cos(a*pi/4),sin(a*pi/4)) for a in range(8)]+[(0,0)]).reshape(9,2,1,1),
                                   array([0]*8+[inv_ninth]).reshape(9,1,1,1)) )
    net.add_relu('pxl_hog_relu')
    
    net.add_reshape_layer('pxl_flatten', (-1, 1, 0, 0) )
    
    # intermediate smoothing
    assert mid_smoothing>0
    if mid_smoothing:
      ks = 5
      mask = smooth_mask(ks, mid_smoothing / 0.996)
      net.add_convolution('pxl_smooth_mid', num_output=1, kernelsize=ks, pad=ks/2, biasterm=False,
                                weights=(mask.reshape(1,1,ks,ks),) )
      # we multiply by sigmoid coefficient at the same time
      net.add_border_rectification('pxl_smooth_mid_border', kernelsize=ks, 
                                 weights=((hog_sigmoid or 1)/convol(mask).reshape(1,1,ks,ks),) )
    
    # apply non-linearity
    assert hog_sigmoid>=0
    if hog_sigmoid:
      net.add_rectified_sigmoid('pxl_sigmoid')
    
    # final smoothing
    assert post_smoothing>=0
    if post_smoothing:
      ks = 5
      mask = smooth_mask(ks, post_smoothing / 1.004)
      net.add_convolution('pxl_smooth_post', num_output=1, kernelsize=ks, pad=ks/2, biasterm=False,
                                  weights=(mask.reshape(1,1,ks,ks),) )
      # we multiply by sigmoid coefficient at the same time
      net.add_border_rectification('pxl_smooth_post_border', kernelsize=ks, 
                                 weights=(1/convol(mask).reshape(1,1,ks,ks),) )
    
    net.add_reshape_layer('pxl_unflatten', (-1, 9, 0, 0) )


def compute_local_matchings( net, img_shape,
                             base_psize = 4,
                             downsize2 = False,
                             downsize_method = 'maxpool2',
                             overlap = 999,
                             ngh_rad = None,
                             nlpow = None,
                             verbose = 1,
                             **kwargs ):
    assert base_psize == 4
    assert overlap == 999
    assert nlpow > 0
    
    # compute the initial patches in source image
    L = 0 # initial pyramid level
    psize = helper.get_patch_size(base_psize,downsize2,0) # initial patch size
    f = psize/base_psize # intial downscaling factor
    
    # subsample if needed
    if downsize2:
      size = 2 ** downsize2
      shape = blob_shape(net.activation_blobs[-1][1])[2:]
      assert (shape[0]%size==0) and (shape[1]%size==0)
      pooling_types = {'avgpool2':net.AVE, 'maxpool2':net.MAX}
      net.add_pooling('pool0', kernelsize=size, stride=size, pool=pooling_types[downsize_method])
    target_shape = blob_shape(net.activation_blobs[-1][1])
    
    if True:
      # normalize each pixel to fixed norm
      net.add_pixel_norm('pxl_norm', norm=1./base_psize**2, inplace=True)  # so that dot(patch, patch)=1
      border_norm = "static"
    else:
      # we accomodate the situation of unnormalized patches
      border_norm = "dynamic"
    
    assert overlap==999
    step, grid, children = gpudm.prepare_big_cells( img_shape, psize, overlap<L+1, False, None, None, 0 )
    norms = ones(grid.shape[:2], dtype=float32)
    
    # perform convolution of patches from images[0::2] on images[1::2]
    trueshape = (target_shape[2]+1, target_shape[3]+1)
    net.add_reshape_layer('reshape', (target_shape[0]/2, 2*target_shape[1], target_shape[2], target_shape[3]), inplace=True)
    
    if ngh_rad<0 or ngh_rad is None:
      offsets = None
      net.add_patch_correlation('conv0', kernelsize=base_psize, pad=base_psize/2, 
                                         normalize_borders=border_norm)
    else:
      assert ngh_rad%f == 0, 'ngh_rad must be a multiple of f'
      offsets = (grid.reshape(-1,2) - ngh_rad)/f
      net.add_patch_correlation('conv0', kernelsize=base_psize, pad=ngh_rad/f, nghrad=ngh_rad/f, 
                                         normalize_borders=border_norm)
    
    # non-linear correction
    net.add_power_law('pow0', nlpow, inplace=True)
    
    if verbose:
      print "layer %d, patch_size = %dx%d," % (L, psize, psize),
      sh = blob_shape(net.activation_blobs[-1][1])[1:]
      print "%d channels --> res shape = %dx%dx%d " % (grid.size/2, sh[0], sh[1], sh[2])
    return helper.PyrLevel(f, psize, grid, norms, None, None, trueshape, offsets, None)



def build_response_pyramid( net, first_level, img_shape, 
                            overlap = 999, 
                            subsample_ref = False, 
                            subsample_target = True, 
                            ngh_rad = None,
                            nlpow = None, 
                            max_psize = 999,
                            use_sparse = False,
                            verbose = 1,
                            **kwargs ):
    assert first_level.children is None
    assert overlap == 999
    assert subsample_ref == False
    assert subsample_target == True
    assert nlpow > 0
    
    res_maps = [helper.PyrLevel(
         first_level.f,         # subsampling factor with respect to original image size
         first_level.psize,     # patch size in original image coordinates
         first_level.grid,      # position (center) of each patch
         first_level.norms,     # norms of each patch
         first_level.assign,    # assignement of patches to res_map layers
         first_level.res_map,   # response map in img1, 
         first_level.trueshape, # true shape of res_map if ngh_rad
         first_level.offsets,   # offsets of response maps if ngh_rad
         first_level.children)] # index of children patches in the lower scale
    
    psize = first_level.psize
    dense_step = 0 if subsample_ref else psize/(1+(overlap<1))
    L=0
    while (2*psize <= min(max_psize,max(img_shape)) and 
           min(blob_shape(net.activation_blobs[-1][1])[-2:]) > 3):  # res_map mustn't be smaller than this
        child = res_maps[-1]
        f, L, psize = child.f, L+1, psize*2
        
        sh = blob_shape(net.activation_blobs[-1][1])[1:]
        if ngh_rad>0:
          if not all(array(sh[1:]) % 4 == 1): break
          #assert all(array(sh[1:]) % 4 == 1), """error: blob_shape must be odd and 
          #      must have a center pixel, (i.e. ngh_rad should a power of 2)"""
        
        # build the set of patches at this scale
        _, grid, children, norms = gpudm.prepare_big_cells( img_shape, psize, overlap<L+1, overlap<L, 
                                                         child.grid, child.norms, dense_step )
        
        # max pooling + subsampling 
        net.add_pooling('pool%d'%L, kernelsize=3, stride=2, pad=1)
        # new_size = (old_size - 1)/2 + 1
        f *= 2
        offsets = (grid.reshape(-1,2)-ngh_rad)/f if ngh_rad>0 else None
        trueshape = None
        
        # perform sparse convolutions
        n_patch_prev = child.grid.size/2
        n_patch_cur = grid.size/2
        nc2 = children.shape[-1]  # number of children per parent patch
        # parent-child connection matrix
        pc = vstack((repeat(arange(n_patch_cur),nc2), children.ravel(),
                     repeat((children>=0).sum(axis=-1).ravel(), nc2),
                     tile(arange(nc2),n_patch_cur) ))
        pc = pc[:,pc[1]>=0]  # eliminate null children
        if ngh_rad>0:
          ks = 1
          pad = 0
          row, col = pc[:2]
        else:
          ks = 3
          pad = 1
          row, col = pc[0], (pc[1]*ks+ 2*(pc[3]/2))*ks + 2*(pc[3]%2)
        sparse_weights = sparse.coo_matrix( (float32(1.0/pc[2]), (row, col)), 
                                            shape=(n_patch_cur, n_patch_prev*ks*ks)).tocsr()
        if use_sparse:
          net.add_sparse_convolution('conv%d'%L, sparse_weights, kernelsize=ks, pad=pad, biasterm=False )
        else:
          net.add_convolution('conv%d'%L, num_output=n_patch_cur, kernelsize=ks, pad=pad, biasterm=False, 
                              weights = (sparse_weights.toarray().reshape(n_patch_cur, n_patch_prev, ks, ks),) )
        
        # non-linear rectification
        net.add_power_law('pow%d'%L, nlpow, inplace=True)
        
        if verbose: 
          print "layer %d, patch_size = %dx%d," % (L, psize, psize),
          sh = blob_shape(net.activation_blobs[-1][1])[1:]
          print "%d channels --> res shape = %dx%dx%d " % (grid.size/2, sh[0], sh[1], sh[2])
        res_maps.append(helper.PyrLevel(f,psize,grid,norms,None,None,trueshape,offsets,children))
    
    return res_maps


def gather_correspondences( net, levels, 
                            img_shape,
                            ngh_rad = None,
                            viz=(), **kwargs ):
    step = levels[0].psize
    ngh_rad = -1 if ngh_rad is None else ngh_rad
    net.add_dm_argmax('argmax', img_shape, tag='pow%d', 
                       nghrad=ngh_rad, step=step, nlevels=len(levels))


def filter_correspondences_cpu((corres0, corres1)):
    # very fast post-processing and simpler to run on CPU
    return gpudm.intersect_corres( ascontiguousarray(corres0[:,:,:6]), 
                                   ascontiguousarray(corres1[:,:,:6]) )




def create_matching_net( input_blob, params, viz=()):
  # build CNN that perform DeepMatching
  assert type(input_blob)==tuple and len(input_blob)==4
  assert input_blob[1]==3, 'error: images must have RGB channels'
  net = Net(input_blob)
  
  imgsize = input_blob[2:]
  patch_size = helper.get_patch_size(**params)
  assert (imgsize[0]%patch_size)==(imgsize[1]%patch_size)==0
  
  # extract pixel descriptors
  pipeline_pixel_desc( net, **params['desc_params'])
  if 'pxl_desc' in viz: return net, []
  
  # compute local matchings: first base level
  levels = [compute_local_matchings( net, imgsize, **params )]
  if 'patch_corr' in viz: return net, levels
  # then rest of the pyramid
  levels = build_response_pyramid( net, levels[0], imgsize, **params )
  
  # select the best displacements (maxpool merge)
  if 'rmap' in viz: return net, levels
  gather_correspondences( net, levels, imgsize, viz=viz, **params )
  
  return net, levels


def match_images( image_pairs, params, net=None, GPU=0, viz=()):
  # prepare images
  if type(image_pairs[1])!=tuple: image_pairs=[image_pairs]
  num_pairs = len(image_pairs)
  image_pairs = [i for (p1,p2) in image_pairs for i in (p1,p2)]
  images = array(image_pairs, dtype=float32).transpose(0,3,1,2)
  
  # check size
  psize = helper.get_patch_size(4,params['downsize2'], 0)
  ity, itx = images.shape[2:] # image shape
  if itx%psize or ity%psize:
    print >>sys.stderr, 'Warning: cropping images so that size is multiple of psize=%d' % psize
    mult = lambda s: (s/psize)*psize
    images = ascontiguousarray(images[:,:,:mult(ity),:mult(itx)])
    ity, itx = images.shape[2:] # image shape
  
  # build cnn if necessary
  if net is None:
    net, levels = create_matching_net( images.shape, params, viz=viz )
  
  if 'net' in viz: net.describe(); pdb.set_trace()
  if 'mem' in viz: helper.viz_mem(net)
  set_GPU(GPU)
  
  res = net.test( images )
  
  if 'pxl_desc' in viz:
    figure()
    subplot(221)
    imshow(img0, interpolation='nearest')
    subplot(222)
    imshow(res[0].mean(axis=0), cmap=cm.gray, interpolation='nearest', vmin=0, vmax=1,)
    subplot(223)
    imshow(img1, interpolation='nearest')
    subplot(224)
    imshow(res[1].mean(axis=0), cmap=cm.gray, vmin=0, vmax=1, interpolation='nearest')
    pdb.set_trace()
    sys.exit()
  
  elif 'patch_corr' in viz:
    lev = levels[0]._replace(res_map=res[0])
    helper.show_conv([lev], img0=img0, img1=img1 )
    sys.exit()
  
  elif 'rmap' in viz:
    layers = net.get_activation('pow*', list=True, ret_index=True)
    for i,l in enumerate(layers):
      print "showing response maps for layer %s" % net.activation_blobs[l][0]
      lev = levels[i]
      lev = lev._replace(res_map=net.activation_blobs[l][1].to_numpy_ref()[0])
      helper.show_conv([lev], img0=img0, img1=img1 )
    sys.exit()
  
  # post-process correspondences with
  # reciprocal verification
  
  assert res.shape[1]==12
  grid_w, grid_h = res.shape[-2:]
  
  all_corres = []
  for n in range(len(res)):
    corres0 = res[n,:6].reshape(grid_h, grid_w, 6)
    corres1 = res[n,6:].reshape(grid_h, grid_w, 6)
    corres = filter_correspondences_cpu((corres0, corres1))
    all_corres.append(corres)
    print 'found %d correspondences for img pair %d' % (len(corres), n)
    
    if 'corres' in viz:
      helper.show_flow(levels, None, corres, mode='rainbow', **viz)
    if 'flow' in viz:
      helper.show_flow(levels, None, corres, mode='flow', **viz)
  
  return all_corres



def parse_args():
  import argparse
  parser = argparse.ArgumentParser()
  
  parser.add_argument("img1", help="Path to the first image")
  parser.add_argument("img2", help="Path to the second image")
  
  parser.add_argument("-GPU", type=int, default=-1, const=0, nargs='?', choices=range(-1,5), 
      help="GPU device number (default=0), or -1 for CPU (default)")
  
  parser.add_argument('-ds','--downscale', type=int, default=1, choices=range(4), 
      help="Prior downscale of input images by 2^D", metavar="D")
  
  parser.add_argument('-sp','--use_sparse', action='store_true', 
      help="Use CUSPARSE for ligther convolutions (GPU only)")
  
  parser.add_argument('-ngh','--ngh_rad', type=int, default=-1, 
      help="Restrict matching to local neighborhood of RAD pixels", metavar="RAD")
  
  parser.add_argument('-pow','--powerlaw', default=1.4, 
      help="Non-linear power-law rectification (default = 1.4)", metavar="G")
  
  #parser.add_argument("--resize", type=int, nargs=2, 
  #    help="[Pre-processing] resize the images to a given shape", metavar=('W','H')) 
  parser.add_argument("--crop", type=int, nargs=2, 
      help="[Pre-processing] crop the images to a given shape", metavar=('W','H')) 
  
  parser.add_argument("-out","--output", type=argparse.FileType('w'), default=sys.stdout, 
      help="Output the matching to a text file") 
  
  parser.add_argument("-v","--verbose", action='count', help="Increase verbosity") 
  parser.add_argument("-viz", type=str, default=[], action='append', 
      choices='net mem pxl_desc patch_corr rmap corres flow'.split(), help="Vizualisation options") 
  
  return parser.parse_args()



if __name__=='__main__':
  # read arguments
  args = parse_args()
  img0 = array(Image.open(args.img1).convert('RGB'))
  img1 = array(Image.open(args.img2).convert('RGB'))
  
  # pre-process images
  img0, img1 = helper.preprocess_images(img0, img1, args)
  
  params = dict(args._get_kwargs())
  params["downsize2"] = args.downscale
  params["nlpow"] = args.powerlaw
  hog_params = dict(presmooth_sigma = 1.0,
                    mid_smoothing = 1.0,
                    hog_sigmoid = 0.2,
                    post_smoothing = 1.0,
                    ninth_dim = 0.3 )
  params["desc_params"] = hog_params
  del params['viz']
  viz = dict( img0=img0, img1=img1, params=params )
  for v in args.viz:  viz[v] = True
  
  # launch matching
  corres = match_images((img0, img1), params, GPU=args.GPU, viz=viz)[0]
  
  helper.output_file(corres, args.output)


'''
example usage:

# no restriction for matching neighborhood
python deep_matching_gpu.py liberty1.png liberty2.png -v -viz corres

# restricting matching to a neighborhood of 16 pixels radius
python deep_matching_gpu.py liberty1.png liberty2.png -v -ds 0 -ngh 16 -viz corres

'''

































