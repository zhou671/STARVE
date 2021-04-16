"""
Contains utilities function for flows: read/write, convert to color, displaying the color coding,
The functions are similar to the Middelbury function (which were written in matlab by Deqing Sun.
Author: Philippe Weinzaepfel
Version: 1.0
Date: 19 November 2012
"""

import numpy as np
"""import matplotlib.pylab as plt"""
import struct
from PIL import Image

# for read/write 
TAG_FLOAT = 202021.25 # tag to check the sanity of the file
TAG_STRING = 'PIEH'   # string containing the tag
MIN_WIDTH = 1
MAX_WIDTH = 99999
MIN_HEIGHT = 1
MAX_HEIGHT = 99999

# for colors
RY = 15
YG = 6
GC = 4
CB = 11
BM = 13
MR = 6

# for flow
UNKNOWN_THRESH = 1e9

def flowToColor(flow, maxflow=None, maxmaxflow=None, saturate=False):
    """
    flow_utils.flowToColor(flow): return a color code flow field, normalized based on the maximum l2-norm of the flow
    flow_utils.flowToColor(flow,maxflow): return a color code flow field, normalized by maxflow

    ---- PARAMETERS ----
        flow: flow to display of shape (height x width x 2)
        maxflow (default:None): if given, normalize the flow by its value, otherwise by the flow norm
        maxmaxflow (default:None): if given, normalize the flow by the max of its value and the flow norm

    ---- OUTPUT ----
        an np.array of shape (height x width x 3) of type uint8 containing a color code of the flow
    """
    h,w,n = flow.shape
    # check size of flow
    if not n == 2:
        raise Exception("flow_utils.flowToColor(flow): flow must have 2 bands")
    # compute max flow if needed
    if maxflow is None:
        maxflow = flowMaxNorm(flow)
    if maxmaxflow is not None:
        maxflow = min(maxmaxflow, maxflow)
    # fix unknown flow
    unknown_idx = np.max(np.abs(flow),2)>UNKNOWN_THRESH
    flow[unknown_idx] = 0.0
    # normalize flow
    eps = np.spacing(1) # minimum positive float value to avoid division by 0
    # compute the flow
    img = _computeColor(flow/(maxflow+eps), saturate=saturate)
    # put black pixels in unknown location
    img[ np.tile( unknown_idx[:,:,np.newaxis],[1,1,3]) ] = 0.0 
    return img

def flowMaxNorm(flow):
    """
    flow_utils.flowMaxNorm(flow): return the maximum of the l2-norm of the given flow

    ---- PARAMETERS ----
        flow: the flow
        
    ---- OUTPUT ----
        a float containing the maximum of the l2-norm of the flow
    """
    return np.max( np.sqrt( np.sum( np.square( flow ) , 2) ) )


def _computeColor(flow, saturate=True):
    """
    flow_utils._computeColor(flow): compute color codes for the flow field flow
    
    ---- PARAMETERS ----
        flow: np.array of dimension (height x width x 2) containing the flow to display

    ---- OUTPUTS ----
        an np.array of dimension (height x width x 3) containing the color conversion of the flow
    """
    # set nan to 0
    nanidx = np.isnan(flow[:,:,0])
    flow[nanidx] = 0.0
    
    # colorwheel
    ncols = RY + YG + GC + CB + BM + MR
    nchans = 3
    colorwheel = np.zeros((ncols,nchans),'uint8')
    col = 0;
    #RY
    colorwheel[:RY,0] = 255
    colorwheel[:RY,1] = [(255*i) // RY for i in range(RY)]
    col += RY
    # YG    
    colorwheel[col:col+YG,0] = [255 - (255*i) // YG for i in range(YG)]
    colorwheel[col:col+YG,1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC,1] = 255
    colorwheel[col:col+GC,2] = [(255*i) // GC for i in range(GC)]
    col += GC
    # CB
    colorwheel[col:col+CB,1] = [255 - (255*i) // CB for i in range(CB)]
    colorwheel[col:col+CB,2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM,0] = [(255*i) // BM for i in range(BM)]
    colorwheel[col:col+BM,2] = 255
    col += BM
    # MR
    colorwheel[col:col+MR,0] = 255
    colorwheel[col:col+MR,2] = [255 - (255*i) // MR for i in range(MR)]

    # compute utility variables
    rad = np.sqrt( np.sum( np.square(flow) , 2) ) # magnitude
    a = np.arctan2( -flow[:,:,1] , -flow[:,:,0]) / np.pi # angle
    fk = (a+1)/2 * (ncols-1) # map [-1,1] to [0,ncols-1]
    k0 = np.floor(fk).astype('int')
    k1 = k0+1
    k1[k1==ncols] = 0
    f = fk-k0

    if not saturate:
        rad = np.minimum(rad,1)

    # compute the image
    img = np.zeros( (flow.shape[0],flow.shape[1],nchans), 'uint8' )
    for i in range(nchans):
        tmp = colorwheel[:,i].astype('float')
        col0 = tmp[k0]/255
        col1 = tmp[k1]/255
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx] = 1-rad[idx]*(1-col[idx]) # increase saturation with radius
        col[~idx] *= 0.75 # out of range
        img[:,:,i] = (255*col*(1-nanidx.astype('float'))).astype('uint8')

    return img





