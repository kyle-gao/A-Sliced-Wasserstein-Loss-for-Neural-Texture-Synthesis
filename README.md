# Reproducibility experiment for A Sliced Wasserstein Loss for Neural Texture Synthesis

For the coding portion of the project we added 3 jupyter notebooks that tackle 3 different problems proposed in the paper; 1) texture synthesis, 2) texture synthesis with user defined tags, and 3) style transfer

Texture synthesis: (texture_synthesis.ipynb)

For texture synthesis we reimplemented the authors Lsw code and optimization using the L-BFGS optimizer in scipy. We also implemented and tested various padding methods for the input images. After testing the Lsw implementation, the gram matrix loss is implemented to compare both loss functions. After this we also adapted both loss functions in a way that they can be used with tensorflows Adam optimizer, we then tested optimizing with the Adam optimizer.

Texture synthesis with user defined tags: (texture_synthesis_with_tags.ipynb)

For texture synthesis with user defined tags we modified the VGG feature extractor to downsample and concatenate the tag to the end of the feature space for each of the feature layer outputs. We also modified the loss and fitting functions to take the input and desired output tag. We then tested this method on the images the authors used in their paper after extracting them manually and after some preprocessing steps.

Style transfer: (style_transfer.ipynb)

The implementation in the style transfer notebook was mostly the same as texture synthesis. The only differences were slight variations in the fitting and loss functions to allow for the input of a content image. This notebook compared the loss functions on style transfer, the original authors' custom VGG and the Keras default, and the effect of using more vs less VGG layers in the loss function.

Our helper functions which are required to run the notebooks are found in helper.py and decoders.py.
Additional textures used in our experiments are found in /AdditionalTextures. Spatial Tags are found in /SpatialTags.

The layers from the custom pretrained VGG19 we used for feature extraction are:
         ['block1_conv1',
          'block1_conv2',
          'block2_conv1',
          'block2_conv2',
          'block3_conv1', 
          'block3_conv2',
          'block3_conv3',
          'block3_conv4',
          'block4_conv1', 
          'block4_conv2',
          'block4_conv3',
          'block4_conv4',
          'block5_conv1',
          'block5_conv2'
          ]
unless otherwise stated in the code.

-texturegen.py is the code from the authors of the paper. The code does texture synthsis with output image of the same size as input image (The original authors' code was hardcoded to 256; passing other sizes does not work as argument due to how vgg_customized.h5 was saved.)
____________________________________________________________________________________________________________________________________________________________________
# A Sliced Wasserstein Loss for Neural Texture Synthesis

This is the official implementation of  ["A Sliced Wasserstein Loss for Neural Texture Synthesis" paper](https://arxiv.org/abs/2006.07229) (to appear in CVPR 2021).

![caption paper](https://unity-grenoble.github.io/website/images/thumbnails/publication_sliced_wasserstein_loss.png)

If you use this work, please cite our paper
```Bibtex
@InProceedings{Heitz_2021_CVPR,
author = {Heitz, Eric and Vanhoey, Kenneth and Chambon, Thomas and Belcour, Laurent},
title = {A Sliced Wasserstein Loss for Neural Texture Synthesis},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2021}
}
```

This implementation focuses on the key part of the paper: the sliced wasserstein loss for texture synthesis.

## Requirements

### Librairies

The following libraries are required:

- Tensorflow 2
- SciPy
- Matplotlib
- Numpy

To install requirements:


```setup
pip install -r requirements.txt
```

This code has been tested with Python 3.7.5 on Ubuntu 18.04.
We recommend setting up a dedicated Conda environment using Python 3.7.

### Pretrained vgg-19 network

A custom vgg network is used, as explained in the supplementals.
It has been modified compared to the keras standard model:

- inputs are preprocessed (including normalization with imagenet stats).
- activations are scaled.
- max pooling layers are replaced with average pooling layers.
- zero padding is remplaced with reflect padding.

## Texture generation

To generate a texture use the following command:

```eval
python texturegen.py [-h] [--size SIZE] [--output OUTPUT] [--iters ITERS] filename
```

The parameters are:

- iters: number of calls to l-bfgs (by default: 20). Each step is one call to scipy's l-bfgs implementation with maxfun=64.
- size: the input texture will be resized to this size (by default: 256, which resizes to 256x256). If the image is not square, it will be center-cropped. The generated texture will have the same resolution.
- output: name of the output file (by default: output.jpg).
- filename: name of the input texture (only mandatory parameter).

Outputs files are:

- resized-input.jpg: input image is resized following the --size parameter. It will be exported as "resized-input.jpg" so it can be compared with the generated output.
- output file: final output file after all iterations. The name is specified by the --output tag (output.jpg by default).
- output-iterN.jpg: the intermediate result after N iterations. If there are 20 iterations, there will be 20 output images.


For instance :

```
python texturegen.py input.jpg
```

## Timing

Timing reference for 20 iterations (which is overkill as good results appear earlier) on 256x256 resolution:

- On GPU (NVIDIA GTX 1080 Ti): 3min58.
- On CPU (intel i5-8600k CPU): 37min33.
