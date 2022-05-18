# FusionAwareConv

This project is based on our CVPR 2020 paper,[Fusion-Aware Point Convolution for Online Semantic 3D Scene Segmentation
](https://arxiv.org/abs/2003.06233)
<img src="asset/scene0708_00.gif" height=250px align="right"/>



## Introduction

We propose a novel fusionaware 3D point convolution which operates directly on the geometric surface being reconstructed and exploits effectively the inter-frame correlation for high quality 3D feature
learning.
## Installation
This code is based on [PyTorch](https://pytorch.org/)  and needs [open3D](http://www.open3d.org/) for convenient visualization

Our code has been tested with Python 3.7.6, PyTorch 1.1.0, open3d 0.9.0, CUDA 10.0 on Ubuntu 16.04. 


## Dataset and Pre-trained weights
We use the ScanNetv2 as our test dataset. If you want to test all the data, you can download the ScanNetV2 dataset from [here](http://www.scan-net.org/). For a quick visulazation test, we provide several pre-proessing scenes of the test set [sequence](https://1drv.ms/u/s!AvuKnc9E9hmqhXJWps9cdc-hDPgA?e=kQ8Bw5). Put the ***scene.h5*** in `path/data`.

We also provide the pre-trained weights for [ScanNet benchmark](http://kaldir.vc.in.tum.de/scannet_benchmark/), you can download from [here](https://1drv.ms/u/s!AvuKnc9E9hmqhXMuH6MUHIitw4iw?e=e2d3vb). After finishing the download, put the weights in `path/weight`.


## Test

### Online Segmentation Visulization
We have already intergrate the open3d for visulizaiton, you can run the command below:

```
python vis_sequence.py --weight2d_path=weight_path/weight2d_name --weight3d_path=weight_path/weight3d_name --gpu=0 --use_vis=1 --scene_path=scene_path/scene_name
```
The complete segmentation result will be generated in `result.ply`.


### Global-local Tree Visualization
We achieve the a test demo for global-local tree visulizaiton only. Run the command below to see the processing of the tree built.
```
python vis_sequence.py  --use_vis=1 --scene_path=scene_path/scene_name
```
The complete result will be generated in  `result_GLtree.ply`.





## Citation
If you find our work useful in your research, please consider citing:
```
@article{zhang2020fusion,
  title={Fusion-Aware Point Convolution for Online Semantic 3D Scene Segmentation},
  author={Zhang, Jiazhao and Zhu, Chenyang and Zheng, Lintao and Xu, Kai},
  journal={arXiv preprint arXiv:2003.06233},
  year={2020}
}
```

## Acknowledgments
Code is inspired by [Red-Black-Tree](https://github.com/stanislavkozlovski/Red-Black-Tree) and [FuseNet_PyTorch](https://github.com/zanilzanzan/FuseNet_PyTorch).

## Contact
If you have any questions, please email Jiazhao Zhang at zhngjizh@gmail.com.




