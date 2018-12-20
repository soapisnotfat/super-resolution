## Super Resolution CNN
The authors of the SRCNN describe their network, pointing out the equivalence of their method to the sparse-coding method, which is a widely used learning method for image SR. This is an important and educational aspect of their work, because it shows how example-based learning methods can be adapted and generalized to CNN models.

The SRCNN consists of the following operations:
1. **Preprocessing**: Up-scales LR image to desired HR size.
2. **Feature extraction**: Extracts a set of feature maps from the up-scaled LR image.
3. **Non-linear mapping**: Maps the feature maps representing LR to HR patches.
4. **Reconstruction**: Produces the HR image from HR patches.

Operations 2–4 above can be cast as a convolutional layer in a CNN that accepts as input the preprocessed images from step 1 above, and outputs the HR image
