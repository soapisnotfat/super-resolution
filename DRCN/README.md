# DRCN

## overview
This project is a test implementation of ["Deeply-Recursive Convolutional Network for Image Super-Resolution", CVPR2016](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Kim_Deeply-Recursive_Convolutional_Network_CVPR_2016_paper.pdf) using tensorflow


Paper: ["Deeply-Recursive Convolutional Network for Image Super-Resolution"](https://arxiv.org/abs/1511.04491) by Jiwon Kim, Jung Kwon Lee and Kyoung Mu Lee Department of ECE, ASRI, Seoul National University, Korea

## model structure

Those figures are from the paper. There are 3 different networks which cooperates to make images fine.

![alt tag](https://raw.githubusercontent.com/jiny2001/deeply-recursive-cnn-tf/master/documents/figure1.png)

![alt tag](https://raw.githubusercontent.com/jiny2001/deeply-recursive-cnn-tf/master/documents/figure3.png)

This model below is made by my code and drawn by tensorboard.

![alt tag](https://raw.githubusercontent.com/jiny2001/deeply-recursive-cnn-tf/master/documents/model.png)
![alt tag](https://raw.githubusercontent.com/jiny2001/deeply-recursive-cnn-tf/master/documents/network_graph2.png)