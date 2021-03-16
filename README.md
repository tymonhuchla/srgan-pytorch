# srgan-pytorch
SRGAN implementation in PyTorch based on a paper <a href='https://arxiv.org/pdf/1609.04802.pdf'>Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network
</a>

# Abstract
**GANs** have been fascinating every machine learning enthusiast with their accomplishments, starting from generating not that realistic human faces to **StyleGANs** 
generating fake faces that are not recognizable from real ones. **GANs** applications don't stop on merely generating data, **pix2pix** and **CycleGAN** are performing
incredibly at image translation tasks. 

Recently I've came across <a href='https://arxiv.org/pdf/1609.04802.pdf'>this</a> paper. It describes a architecture called **SRGAN** which is able to perform super-resolution, meaning it can increase resolution of a given low-resolution image.

# Implementation

The original paper provides a well defined architecture for both the generator so luckly for me it wasn't to difficult to implement :)

**Generator's** mainly consists of residual blocks, which are constructed of 
* conv3x3 
* batchnorm 
* PReLU 
* conv3x3 
* batchnorm 
* element-wise sum

**Discriminator** is very similar

* conv3x3
* batchnorm
* LeakyReLU

<img src='https://paperswithcode.com/media/methods/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png'>

<i>Source : https://paperswithcode.com/media/methods/Screen_Shot_2020-07-19_at_11.13.45_AM_zsF2pa7.png</i>

**Loss**

Generator loss composes **reconstruction loss** and **adversarial loss**. SRGAN uses a adversarial loss a function called **perceptual loss** which measures the 
MSE between features extracted by VGG19 net.

For discriminator training SRGAN uses the regular discriminator loss.

**Training**

I have used CelebA for my dataset. Generator uses a low-resolution image and tries to recreate it to super-resolution version. For my *high resolution* data 
I have used the original CelebA images and for *low resolution* I simply resized images to 64x64 and applied gaussian blur. 

I had memory issues when I was trying to train the model on a bigger batch size, so I have chosen batch size of 16 for my training which in turn made the training
very slow, in fact I was only able to train the model for 2 epochs.

# Results

After 2 epochs (~5h on GCP's V100 VM instance ) I have decided to finish the training and save the generator and discriminator weights. Below you may see some of
my results. Image on the left is 64x64 with gaussian noise, middle one is my models prediction and on the right - ground truth.

<img src='https://github.com/ty-on-h12/srgan-pytorch/blob/master/generated__data/1.png'>
<img src='https://github.com/ty-on-h12/srgan-pytorch/blob/master/generated__data/2.png'>
<img src='https://github.com/ty-on-h12/srgan-pytorch/blob/master/generated__data/3.png'>
<img src='https://github.com/ty-on-h12/srgan-pytorch/blob/master/generated__data/4.png'>

While not perfect I belive that results are quite impressive and in comparison to non nerual network based super-resolution methods this seem to perform much better.
