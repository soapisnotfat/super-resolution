# SRGAN: Super-Resolution using GANs
This is a complete Pytorch implementation of [Christian Ledig et al: "Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network"](https://arxiv.org/abs/1609.04802), 
reproducing their results. 
This paper's main result is that through using an adversarial and a content loss, a convolutional neural network is able to produce sharp, almost photo-realistic upsamplings of images.

The implementation tries to be as faithful as possible to the original paper.
See [implementation details](#method-and-implementation-details) for a closer look.  


## Method and Implementation Details
Architecture diagram of the super-resolution and discriminator networks by Ledig et al:

<p align='center'>
<img src='https://github.com/mseitzer/srgan/blob/master/images/architecture.png' width=580>  
</p>

The implementation tries to stay as close as possible to the details given in the paper. 
As such, the pretrained SRGAN is also trained with 1e6 and 1e5 update steps. 
The high amount of update steps proved to be essential for performance, which pretty much monotonically increases with training time.

Some further implementation choices where the paper does not give any details:
- Initialization: orthogonal for the super-resolution network, randomly from a normal distribution with std=0.02 for the discriminator network
- Padding: reflection padding (instead of the more commonly used zero padding)

## Batch-size
batch size of 2 is recommended if GPU has only 8G RAM.