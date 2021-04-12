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
import os, sys, pdb, cPickle
from collections import namedtuple
from PIL import Image
from numpy import *
try:
  from matplotlib.pyplot import *
  ion()
except:
  pass

########################################################
# visualization

def noticks():
  xticks([])
  yticks([])

def plot_rect(l,t,r,b,ls='-',rescale=True,**kwargs):
    (L,R), (B,T) = xlim(), ylim()
    plot([l,r,r,l,l],[t,t,b,b,t],ls,scalex=0,scaley=0,**kwargs)
    if rescale:
      xlim((min(L,l),max(R,r)))
      ylim((max(B,b),min(T,t)))

def plot_square(cx,cy,rad,ls='-',**kwargs):
    plot_rect(cx-rad,cy-rad,cx+rad,cy+rad,ls,**kwargs)


def get_assign(assign, n_maps):
  return arange(n_maps) if assign==None else assign 


def show_conv( levels, rot45=None, nshow=0, img0=None, img1=None, **kwargs ):
    level = levels[-1]
    grid = level.grid.reshape(-1,2)
    if rot45: grid = apply_rot45(rot45,grid)
    rad = level.psize/2
    assign = get_assign(level.assign,len(level.res_map))
    
    ax1 = subplot(311)
    ax1.numplot = 1
    imshow(img0, interpolation='nearest')
    ax2 = subplot(312)
    ax2.numplot = 2
    imshow(img1, interpolation='nearest')
    ax3 = subplot(313)
    ax3.numplot = 3
    fig = get_current_fig_manager().canvas.figure
    
    def redraw():
      # we redraw only the concerned axes
      renderer = fig.canvas.get_renderer()
      ax1.draw(renderer)  
      ax2.draw(renderer)
      ax3.draw(renderer)
      fig.canvas.blit(ax1.bbox)
      fig.canvas.blit(ax2.bbox)
      fig.canvas.blit(ax3.bbox)
    
    global cur  # ugly but wo cares
    cur = None
    def motion_notify_callback(event):
      global cur
      if not event.inaxes:  return
      x,y = event.xdata, event.ydata
      if x and y: # we are somewhere on a plot
        if cur is not None and event.inaxes.numplot in (2,3):
          ax2.lines = ax2.lines[:1]
          ax3.lines = []
          offx,offy = (0,0) if level.offsets is None else level.offsets.reshape(-1,2)[cur]
          if event.inaxes.numplot==2:
            f = level.f
            x,y = int(0.5 + x/f), int(0.5 + y/f)
          else:
            f = 1
            x,y = int(0.5 + (offx+x)/f), int(0.5 + (offy+y)/f)
          fig.add_subplot(312)
          xl,yl=xlim(),ylim()
          ax2.plot(x*level.f,y*level.f,'+',c=(0,1,0),ms=10,scalex=0,scaley=0)
          plot_square(x*level.f-0.5,y*level.f-0.5,rad,color='b')
          xlim(xl);ylim(yl)
          ax3.plot(x-offx,y-offy,'+k',ms=20,scalex=0,scaley=0)
          redraw()
    
    def mouse_click_callback(event):
      global cur
      if not event.inaxes:  return
      x,y = event.xdata, event.ydata
      if x and y: # we are somewhere on a plot
        if event.inaxes.numplot==1:
          cur = sum((grid - [x,y])**2,1).argmin()   # find nearest point
          x,y = grid[cur]
          ax1.lines = []
          ax2.lines = []
          ax3.lines = []
          fig.add_subplot(311)
          xl,yl=xlim(),ylim()
          plot(x,y,'+',color=(0,1,0),ms=10,mew=1)
          plot_square(x-0.5,y-0.5,rad,color='b')
          xlim(xl);ylim(yl)
          ax3.images = []
          ax3.imshow(level.res_map[assign[cur]], vmin=0, vmax=1.1, interpolation='nearest')
          if level.offsets is not None:
            ox, oy = level.offsets.reshape(-1,2)[cur]
            sx, sy = level.res_map.shape[1:]
            subplot(312)
            plot_rect(level.f*ox,level.f*oy,level.f*(ox+sx),level.f*(oy+sy),'-',c=(0,1,0),rescale=False)
          redraw()
        elif cur is not None and event.inaxes.numplot>1:
          offx,offy = (0,0) if level.offsets is None else level.offsets.reshape(-1,2)[cur]
          offx,offy = (0,0) if level.offsets is None else level.offsets.reshape(-1,2)[cur]
          if event.inaxes.numplot==2:
            f = level.f
            x,y = int(0.5 + x/f), int(0.5 + y/f)
            score = level.res_map[cur,y-offy,x-offx]
          else:
            x,y = int(0.5 + x), int(0.5 + y)
            score = level.res_map[cur,y,x]
          print 'res_map[%d, %d, %d] = %g' % (cur, y, x, score)
    
    class FakeEvent:
      def __init__(self, x,y,ax):
        self.xdata=x; self.ydata=y; self.inaxes=ax
    mouse_click_callback(FakeEvent(1,1,ax1))
    
    subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.02, hspace=0.02)
    cid_move = fig.canvas.mpl_connect('motion_notify_event',motion_notify_callback)
    cid_clic = fig.canvas.mpl_connect('button_press_event',mouse_click_callback)
    print "Click on the top image to select a patch..."
    pdb.set_trace()
    fig.canvas.mpl_disconnect(cid_move)
    fig.canvas.mpl_disconnect(cid_clic)


def get_imatches(matches, shape, psize=8):
    half = psize/2
    imatches = -ones(shape, dtype=int32)
    nums = arange(len(matches))
    for j in range(-half,half+(psize%2)):
      for i in range(-half,half+(psize%2)):
        imatches[ clip(matches[:,1]+i,0,shape[0]-1), 
                  clip(matches[:,0]+j,0,shape[1]-1)] = nums
    return imatches


def show_flow( lm, maxima, corr, img0=None, img1=None, mode='flow', full_corres=None, psize=None, **viz ):
    assert img0 is not None
    assert img1 is not None
    if type(corr)==tuple: corr = corr[0]
    assert corr.size, 'error: empty correspondences'
    if corr.ndim==3:
      corr = corr[corr[:,:,4]>0]
    set_max = set(corr[:,5])
    colors = {m:i for i,m in enumerate(set_max)}
    colors = {m:cm.jet(i/float(len(colors))) for m,i in colors.items()}
    for key in viz:
      if key.startswith('mode_') and viz[key] is True:
        mode = key[5:]
    
    def motion_notify_callback(event):
      if not event.inaxes:  return
      x,y = event.xdata, event.ydata
      if x and y: # we are somewhere on a plot
        ax1.lines = []
        ax2.lines = []
        if event.inaxes.numplot==0:
          if mode=='score_path':
            ax3.lines = []
            col = fc0[int(y/step),int(x/step)]
            x0, y0, x1, y1 = col[:4]
            ax3.plot( col[6:], '+-', color='k' )
          elif mode in ('comatches','argmax'):
            n = sum((corr[:,0:2] - [x,y])**2,1).argmin()   # find nearest point
            x0,y0,x1,y1,_,m = corr[n,0:6]
            # print leading correspondences
            ax1.plot(x0,y0,'o',ms=10,mew=2,color='blue',scalex=False,scaley=False)
            ax2.plot(x1,y1,'o',ms=10,mew=2,color='red',scalex=False,scaley=False)
            # find co-matches
            corres0 = retrieve_one_maxima_corres( lm, maxima[m], **viz['params'] )
            corres0 = set(map(tuple,corres0[:,0:4]))
            if mode == 'comatches':
              real0 = set(map(tuple,corr[:,0:4]))
              intersect = corres0 & real0
            else:
              intersect = corres0 # no filtering
            x0,y0,x1,y1 = zip(*list(intersect))
            
          else:
            n = sum((corr[:,0:2] - [x,y])**2,1).argmin()   # find nearest point
            x0,y0,x1,y1,score,m = corr[n,0:6]
            print "\rmatch #%d (%d,%d) --> (%d,%d) (len=%.1f), score=%.3f from maxima %d" % (n,
              x0,y0,x1,y1,sqrt((x0-x1)**2+(y0-y1)**2),score,m),;sys.stdout.flush()
          
          ax1.plot(x0,y0,'+',ms=10,mew=2,color='blue',scalex=False,scaley=False)
          ax2.plot(x1,y1,'+',ms=10,mew=2,color='red',scalex=False,scaley=False)
        
        elif event.inaxes.numplot==1:
          if mode=='score_path':
            ax3.lines = []
            col = fc1[int(y/step),int(x/step)]
            x0, y0, x1, y1 = col[:4]
            ax3.plot( col[6:], '+-', color='k' )
          else:
            n = sum((corr[:,2:4] - [x,y])**2,1).argmin()  # find nearest point
            x0,y0,x1,y1,score,m = corr[n,0:6]
            print "\rmatch #%d (%d,%d) --> (%d,%d) (len=%.1f), score=%.3f from maxima %d" % (n,
              x0,y0,x1,y1,sqrt((x0-x1)**2+(y0-y1)**2),score,m),;sys.stdout.flush()
          ax1.plot(x0,y0,'+',ms=10,mew=2,color='red',scalex=False,scaley=False)
          ax2.plot(x1,y1,'+',ms=10,mew=2,color='blue',scalex=False,scaley=False)
        # we redraw only the concerned axes
        renderer = fig.canvas.get_renderer()
        ax1.draw(renderer)  
        ax2.draw(renderer)
        fig.canvas.blit(ax1.bbox)
        fig.canvas.blit(ax2.bbox)
        if mode=='score_path':
          ax3.set_ylim((0,1))
          ax3.draw(renderer)
          fig.canvas.blit(ax3.bbox)
    
    wider_than_high = (img0.shape[0]+img1.shape[0]<img0.shape[1]+img1.shape[1])
    if mode in ('corres', 'rainbow'):
      if wider_than_high:
        layouts = (311, 312, 325, 326)
      else:
        layouts = (221, 222, 223, 224)
    else:
      if wider_than_high:
        layouts = (311, 312, 313)
      else:
        layouts = (221, 222, 212)
    
    clf()
    ax1 = subplot(layouts[0])
    ax1.numplot = 0
    imshow(img0,interpolation='nearest')
    noticks()
    ax2 = subplot(layouts[1])
    ax2.numplot = 1
    imshow(img1,interpolation='nearest')
    noticks()
    
    if mode in 'flow score score_path comatches argmax':
      ax3 = subplot(layouts[2])
      if mode == 'score_path':
        assert full_corres is not None
        ax3.numplot = -1
        # retrieve score's path
        step = full_corres.step
        fc0 = full_corres.corres0
        fc1 = full_corres.corres1
        m = fc0[:,:,6:].mean(axis=0).mean(axis=0)
        # find periodicity
        fc_period = min([p for p in (1,4,6) if all(m[p-1::p]<=1)])
        fc0 = fc0[:,:,range(6)+range(6+fc_period-1,fc0.shape[-1],fc_period)]
        fc1 = fc1[:,:,range(6)+range(6+fc_period-1,fc1.shape[-1],fc_period)]
        plot( m[fc_period-1::fc_period], '+-', color='k' )
        ylim((0,1))
        
      else:
        ax3.numplot = 0
        from flow_utils import flowToColor
        matches = int32(corr)
        imatch = get_imatches(matches,img0.shape[:2],lm and lm[0].psize or psize or 1)
        if mode in 'flow comatches argmax':
          corr_flow = (matches[:,2:4]-matches[:,0:2])[imatch]
          corr_color = flowToColor(corr_flow, maxflow=50)
        if mode == 'score':
          corr_color = corr[:,4][imatch]
        corr_color[imatch<0] = 0
        imshow(corr_color,interpolation='nearest')
    
    elif mode == 'rainbow':
      # make beautiful colors
      center = corr[:,[1,0]].mean(axis=0) # array(img0.shape[:2])/2 #
      corr[:,5] = arctan2(*(corr[:,[1,0]] - center).T)
      corr[:,5] = int32(64*corr[:,5]/pi) % 128
      
      set_max = set(corr[:,5])
      colors = {m:i for i,m in enumerate(set_max)}
      colors = {m:cm.hsv(i/float(len(colors))) for m,i in colors.items()}
      
      ax3 = subplot(layouts[2])
      ax3.numplot = 0
      imshow(img0/2+64,interpolation='nearest')
      for m in set_max:
        plot(corr[corr[:,5]==m,0],corr[corr[:,5]==m,1],'+',ms=10,mew=2,color=colors[m],scalex=0,scaley=0)
      noticks()
      
      ax4 = subplot(layouts[3])
      ax4.numplot = 1
      imshow(img1/2+64,interpolation='nearest')
      for m in set_max:
        plot(corr[corr[:,5]==m,2],corr[corr[:,5]==m,3],'+',ms=10,mew=2,color=colors[m],scalex=0,scaley=0)
      noticks()
    
    else:
      ax3 = subplot(layouts[2])
      ax3.numplot = None
      imshow(img0/4+192,interpolation='nearest')
      #plot(corr[:,0],corr[:,1],'+',ms=10,mew=2)
      for m in set_max:
        plot(corr[corr[:,5]==m,0],corr[corr[:,5]==m,1],'+',ms=10,mew=2,color=colors[m],scalex=0,scaley=0)
      noticks()
      
      ax4 = subplot(layouts[3])
      ax4.numplot = None
      imshow(img1/4+192,interpolation='nearest')
      for m in set_max:
        plot(corr[corr[:,5]==m,2],corr[corr[:,5]==m,3],'+',ms=10,mew=2,color=colors[m],scalex=0,scaley=0)
      noticks()
    
    subplots_adjust(left=0.03, bottom=0.03, right=1, top=1, wspace=0.02, hspace=0.02)
    
    fig = get_current_fig_manager().canvas.figure
    cid_move = fig.canvas.mpl_connect('motion_notify_event',motion_notify_callback)
    print "Move your mouse on the top images..."
    pdb.set_trace()
    fig.canvas.draw()
    fig.canvas.mpl_disconnect(cid_move)


def viz_mem(net):
    pos = arange(len(net.activation_blobs))
    width = 0.9/2
    from collections import OrderedDict
    blob_sizes = OrderedDict()
    hash_blob = lambda b: b.cpu_data().__long__()
    blob_size = lambda b: int(b.count())*4 # sizeof(float) (diff is never used)
    
    ab_sizes, ab_fakes = [], []
    for n,b in net.activation_blobs:
      if b is None: 
        ab_sizes.append(0)
        ab_fakes.append(0)
        continue
      h = hash_blob(b)
      size = blob_size(b)
      if h not in blob_sizes:
        blob_sizes[h] = size
        ab_sizes.append(size)
        ab_fakes.append(0)
      else:
        ab_sizes.append(0)
        ab_fakes.append(size)
    
    w_sizes, w_fakes = [0],[0]
    for n,l in net.layers:
      total = fake = 0
      for i in range(len(l.blobs())):
        b = l.blobs()[i]
        h = hash_blob(b)
        size = blob_size(b)
        total += size
        if h in blob_sizes:
          fake += size
          blob_sizes[h] = size
      w_sizes.append(total)
      w_fakes.append(fake)
    
    try:
      ab_rects = barh(pos, ab_sizes, width, color='r', label="Activation blobs")
      barh(pos, ab_fakes, width, color='pink', label="Duplicated (inplace) activation blobs")
      w_rects = barh(pos-0.5, w_sizes, width, color='b', label="Layer's blobs")
      barh(pos-0.5, w_fakes, width, color=(0.5,0.5,1), label="Duplicated (inplace) layer's blobs")
      
      yticks(pos+0.5+width/2, ['[%d] %s'%(i,n) for i,(n,l) in enumerate(net.layers)])
      legend(loc='upper right')
    except:
      print "error with matplotlib display"
    
    total = sum(ab_sizes) + sum(w_sizes)
    print "/!\\ WARNING: this estimation is optimistic"
    print "              (it doesn't count layer's hidden blobs)"
    print 'total size = %dB (%.3f GB)' % (total, total/1.e9)
    pdb.set_trace()


########################################################
# Main 

def get_patch_size( base_psize=4, downsize2=False, truedownsize2=False, **kwargs):
    upsize = 2**(downsize2 + truedownsize2)
    return base_psize*upsize

# robust definition of a pyramid level
PyrLevel = namedtuple('PyrLevel', 'f psize grid norms assign res_map trueshape offsets children')


########################################################
# argument parsing / parameters 


def preprocess_images(img0, img1, args):
  if args.crop:
    W,H = args.crop
    img0 = img0[:H,:W]
    img1 = img1[:H,:W]
  
  return img0, img1


def output_file( corres, outfile ):
    for x1, y1, x2, y2, score, index in corres:
      outfile.write("%d %d %d %d %g %d\n" % (x1, y1, x2, y2, score, index) )


















