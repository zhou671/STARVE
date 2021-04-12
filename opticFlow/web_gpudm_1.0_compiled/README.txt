Implementation of the Deep Matching algorithm on GPU. 
See paper "DeepMatching: Hierarchical Deformable Dense Matching", 
at http://lear.inrialpes.fr/src/deepmatching/ by Jerome Revaud, 
Philippe Weinzaepfel, Zaid Harchaoui and Cordelia Schmid.
Main code by Jerome Revaud, INRIA. The code is only for scientific 
or personnal use. Please contact me/INRIA for commercial use.
Email: jerome.revaud@inria.fr

Copyright (C) 2015 Jerome Revaud

Version 1.0

License:

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


Installation:

1) Install 'caffe'
   somewhere at /path/to/caffe

2) Add a softlink named 'caffe' in the current directory that points 
   toward the caffe directory (just to make things simpler):
   ln -s /path/to/caffe caffe

3) Edit Makefile
   Set the different paths for caffe and other dependencies

4) Compile
   make all
   
   (IMPORTANT NOTE: 
    if you compile the SWIG file, it is normal that you see lots of errors. 
    However, as long as it generates 'gpudm_warp.cxx', you can just relaunch 
    'make' and the rest of the compilation should go smoothly. )

5) Test
  Set your environment variables:
    LD_LIBRARY_PATH should point to your libcaffe directory, to mkl, etc. 
    (basically to all the paths you edited in the Makefile)
  
  Try executing the following command:
  >> python deep_matching_gpu.py liberty1.png liberty2.png -v -viz corres
  
  You should see a rainbow visualization of correspondences. 
  When you continue (type 'c'+enter), you should get the following output: 
    36 36 36 26 3.77558 13
    36 44 36 34 3.65536 24
    [...]
    28 4 22 6 3.59854 94
    28 20 28 10 3.75238 90
    28 28 28 18 3.77126 83


Example usages and explanations:
  
  To get detailed information on parameters:
    python deep_matching_gpu.py -h
    python deep_matching_gpu.py --help
  
  Typical command to match Sintel images:
    # Requires 4.8 Go of memory on your GPU
    python deep_matching_gpu.py .../SINTEL/training/final/temple_3/frame_0041.png \
                                .../SINTEL/training/final/temple_3/frame_0042.png \
                   -GPU -v --downscale 1 --ngh_rad 256 --use_sparse -viz mem -viz flow 
    
    # Note: if you don't use CUsparse, (no --use_sparse), it requires 5.3 Go.
  
  Nice vizualizations options: 
    #   e.g. use "-viz rmap" to examine response_maps
    #   (Then click on the top image to select a patch)
    python deep_matching_gpu.py liberty1.png liberty2.png -v --downscale 0 -viz net -viz mem -viz rmap 


For details about the options, please refer to the help, the papers or the code.


Important tip:
  If the program stops with an error in memset/memcpy, then it means that your GPU 
  doesn't have enough memory. In this case, you should consider:
    * decreasing the neighborhood radius with "--ngh_rad"  
    * incrementing the "--downscale" parameter (i.e. to downscale images before matching). 
    * You can also investigate the memory usage with '-viz mem', but keep in mind that
      it under-estimates the actual memory usage.


Versions of the different modules:
  * python 2.7
  * fedora 21
  * gcc/g++ 4.9.2
  * swig 3.0.7 
  * cuda 6.5
  * caffe version is from November 12th (included in zip)
  * protobuf 2.5.0
  * glog 0.3.3




























