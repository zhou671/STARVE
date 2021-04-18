These files are to produce forward and backward optic flows (`.flo` files) 
and consistency weights (`.pgm` files).

See this notebook tutorial for how to compile 
[Caffe](https://caffe.berkeleyvision.org) and 
[DeepMatching-GPU](https://thoth.inrialpes.fr/src/deepmatching/) (optional). 
<a href="https://colab.research.google.com/github/zhou671/STRAVE/blob/master/compile-caffe-and-deepmatching-gpu-tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

`./consistencyChecker/`, `makeOptFlow.sh`, `run-deepflow.sh`
are copied from [manuelruder/artistic-videos](https://github.com/manuelruder/artistic-videos).
But bash commands are rewritten as Python codes with the subprocess package.

`deepflow2-static` is copied from [DeepFLow2](https://thoth.inrialpes.fr/src/deepflow/).

`deepmatching-static`, `viz.py` are copied form [DeepMatching](https://thoth.inrialpes.fr/src/deepmatching/).
The `print` functions in `viz.py` is modified to accommodate for Python3.x.

`web_gpudm_1.0_source` is downloaded from [DeepMatching-GPU V1.0](https://thoth.inrialpes.fr/src/deepmatching/code/deepmatching_gpu_1.0.zip).
`web_gpudm_1.0_compiled` contains the files compiled on Colab.
`web_gpudm_1.0_compiled/deep_matching_gpu_folder.py` is based on `deep_matching_gpu.py`.
However, we use `deep_matching_gpu_folder.py` to load the model only once and process multiple images in a folder.

To compile the files in `./consistencyChecker`, run
```bash
cd consistencyChecker
make
cd ..
```