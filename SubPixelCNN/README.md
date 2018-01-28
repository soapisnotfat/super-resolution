## sub-pixel ConvNN
A practice implementation of one simple super-resolution model.This repo is modified from [pytorch sample code](https://github.com/pytorch/examples/tree/master/super_resolution) and more features are added.
This repo keeps same [BSD300 dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) and data settings used in sample code. This [model](https://github.com/IvoryCandy/super-resolution/tree/master/sub_pixel_CNN) illustrates how to use the efficient sub-pixel convolution layer described in ["Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network" - Shi et al.](https://arxiv.org/abs/1609.05158) for increasing spatial resolution within your network for tasks such as superresolution.



## Usage
#### Train
```bash
python main.py
```
optional arguments:

    --upscale_factor        super resolution upscale factor
    --batchSize             training batch size
    --testBatchSize         testing batch size
    --nEpochs               number of epochs to train for
    --lr                    Learning Rate. Default=0.01
    --threads               number of threads for data loader to use
    --seed                  random seed to use. Default = 123

#### Super Resolve
```bash
python super_resolve.py --input xxx.jpg --output out.png
```
optional arguments:

    --input                 image will be siper-resolved
    --model                 model path
    --output                super-resolved image


## Performance
- with weight_initialization: 22.75 PSNR
- without Weight_initialization: 20.13 PSNR


## Credit
This code is modified from [pytorch sample code](https://github.com/pytorch/examples/tree/master/super_resolution), which is  under [BSD-3-License](https://github.com/pytorch/examples/blob/master/LICENSE) protection. 

Any modification by me is under [MIT](https://github.com/IvoryCandy/super-resolution/blob/master/LICENSE.md) protection.
