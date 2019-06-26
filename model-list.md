#Machine Learning Models

Cortex has 23 models trained with 4 datasets that serve 7 different purposes. All models have been quantized using MRT that is compatible to make inference in CVM, the runtime environment for smart contracts equipped with machine learning models.

##Model List

![model-list](/Users/oscarwei/Dropbox/markdown/tech-doc/model-list.jpg)

##Methods

Converting the original floating-point model to our CVM representation results in approximately 4x model size reduction while the model accuracy does not decrease significantly. Besides, we only introduce minor additional computation overheads, e.g. requantization, keeping the number of operations in the same order of magnitude. All operators in the model can be further optimized using vectorization techniques, which will reduce the computation time dramatically, e.g., avx512-vnni instruction set.

We apply the proposed converter on pre-trained models with ImageNet dataset from MXNet Model Zoo. The results of top-1 accuracy are shown below,

![img](mxnetvscvm.png)*Figure. 3*

