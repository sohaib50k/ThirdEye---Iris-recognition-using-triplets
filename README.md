# ThirdEye---Iris-recognition-using-triplets
Iris recognition using variants of the triplet loss.

For starters the implementation uses Keras and Tensorflow, other needed packages include numpy, opencv, matplotlib among others.

The losses borrow heavily from the [omoindrot triplet loss implementations](https://github.com/omoindrot/tensorflow-triplet-loss).



# Training process 
The recognition pipeline works on the outputs of the [segmentation pipeline](https://github.com/sohaib50k/Unconstrained-iris-segmentation-using-Mask-R-CNN). It needs square images with irises in the center. The background is set to black. This implementation uses a custom Keras generator, the data format is in this format:

* Training data
  * class1
    * iris1_class1
    * iris2_class1
  * class2
    * iris1_class1
    * iris2_class1

Left and right irises are different classes due to them being statistically different.


# Testing process
In progress


If you use this repository consider citing our [ paper ](https://arxiv.org/pdf/1907.06147.pdf)

@article{ahmad2019thirdeye,
  title={ThirdEye: Triplet Based Iris Recognition without Normalization},
  author={Ahmad, Sohaib and Fuller, Benjamin},
  journal={arXiv preprint arXiv:1907.06147},
  year={2019}
}
