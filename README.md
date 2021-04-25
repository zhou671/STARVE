# STARVE: Style TrAnsfeR for VidEos

This is the final project for class 
[CSCI 1430: Introduction to Computer Vision](https://browncsci1430.github.io/webpage/).

**Team members** (alphabetical by first name): 
[Yicheng Shi](https://github.com/yshi77), 
[Yuchen Zhou](https://github.com/zhou671), 
[Yue Wang](https://github.com/yuewangpl),  
and [Zichuan Wang](https://github.com/GuardianWang).

In this project, we explored **video style transfer** with TensorFlow2.
Our work is heavily based on paper 
[Artistic style transfer for videos](http://arxiv.org/abs/1604.08610)
and [this repo](https://github.com/manuelruder/artistic-videos) 
written in Lua and C++.
We also refer to in this 
[tutorial](https://www.tensorflow.org/tutorials/generative/style_transfer)
for basic functions.

See this notebook tutorial for how to run the model.
<a href="https://colab.research.google.com/github/zhou671/STARVE/blob/master/run-style-transfer-for-videos-tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

See this notebook tutorial for how to compile 
[Caffe](https://caffe.berkeleyvision.org) and 
[DeepMatching-GPU](https://thoth.inrialpes.fr/src/deepmatching/) (optional). 
<a href="https://colab.research.google.com/github/zhou671/STARVE/blob/master/compile-caffe-and-deepmatching-gpu-tutorial.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Deliverables**:  
[Report]()  
[Slides](https://docs.google.com/presentation/d/1dJDt8xB92ljd9HKefz_WP4k5leUGgdYZpiRoiVlTr3Y/edit?usp=sharing)  
[Video Demo](https://youtu.be/i5yk5Y3pp4g)

## Optic Flow

If calculate optic flow with 2 CPU cores, it takes 35s per frame on average.
 
`Optic flow: 100% 120/120 [1:09:26<00:00, 34.72s/it]`

When using the GPU version of DeepMatching, it takes 6s per frame on average.
There's a 6x speed up!

```
DeepMatching: 100% 120/120 [00:18<00:00, 6.68it/s]
Optic flow: 100% 120/120 [12:14<00:00, 6.12s/it]
```
