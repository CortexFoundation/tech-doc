# Synapse: Cortex Deterministic Deep Learning Inference Framework

## Abstract
Simply put, the sole purpose of The Framework is to guarantee the execution of any compatible deep learning model is deterministic on any full node, while not downgrading performance or accuracy by too much.

## Summary
The deterministic condition formally translates into the following criteria:

1. `Criteria-of-Closure`: No external randomness should be introduced as input or in the runtime, software introduced or hardware introduced.
2. `Criteria-of-Invariance`: Output of parallel procedures should be invariant of scheduling strategies.

As far as we know, `Criteria-of-Closure` can easily be met,  because pseudo-randomness suffice in deep learning, which is deterministic in nature. `Criteria-of-Invariance` is non-trivial for most models and inference frameworks, because low-level parallelism can behave differently if the order of execution is different.

Cortex replaces floating point numbers to integer in The Framework to eliminate randomness of floating point calculation errors in different hardware architectures. Furthermore, Cortex limits the width and depth for different type of neural network layer to eliminate potential overflow for intermediate results, therefore implementing `Criteria-of-Invariance`.

Under the new set of limitation, precision could downgrade for most models. Currently fine tuning is needed to train the new integer models to reduce precision loss. With 8-bit integer models we can have a compression rate of 25% (75% size reduction).

## Specification
Parameter type:
```
8-bit integer(int8)
```
For basic operations:
1. Convolutional layers
```
Conv1D, Conv2D, Conv3D(Not supported in testnet)
Kernel size * channel number < 2^16
int8 weights with int8 input
int8 output features
int32 intermediate results
```
2. Normalization Layers
```
BatchNorm(scale factors must be int8)
```
3. Non-Linear Activation Layers
```
ReLU, PReLU(p must be int8)
```
4. Fully Connected Layers
```
int8 weights with int8 input
int8 output features
int32 intermediate results
```
5. Residual Network Configuration
```
In x + cf(x), c=1
```
6. Model constraints
```
Input image dimension less than 3 * 224 * 224 * 8-bit color
Model depth <= 200 layers
Channel size <= 4096
```

## Notes
- Please use Conv1D and quantization instead of RNN-like structures, e.g. GRU/LSTM.  Total Accuracy will not be harmed in NLP/OCR practices.
- Please use wider or deeper residual structure and grouping instead of Concat Layers.

## Roadmap
- int4/int2 models for higher model compression rate
- int8 models with less harm to model accuracy
- int32/int16 activation for compatibility
- More models beyond images

## References
[Cor(2018)] Coreml document, integrate machine learning models into your app., 2018. URL https://developer.apple.com/documentation/coreml.

[tvm(2018)] Open deep learning compiler stack for cpu, gpu and specialized accelerators: dmlc/tvm, 2018. URL https://github.com/dmlc/tvm.

[Ba et al.(2016)Ba, Kiros, and Hinton] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
1

[Chen et al.(2018)Chen, Moreau, Jiang, Zheng, Yan, Cowan, Shen, Wang, Hu, Ceze, Guestrin, and Krishnamurthy]  Tianqi Chen, Thierry Moreau, Ziheng Jiang, Lianmin Zheng, Eddie Yan,
Meghan Cowan, Haichen Shen, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, and Arvind Krishnamurthy. TVM: An Automated End-to-End
Optimizing Compiler for Deep Learning. arXiv preprint arXiv:1802.04799, February 2018.

[He et al.(2015)He, Zhang, Ren, and Sun] K He, X Zhang, S Ren, and J Sun. Deep residual learning for image recognition. corr, vol. abs/1512.03385, 2015.

[Ioffe & Szegedy(2015)Ioffe and Szegedy] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167, 2015.

[Krizhevsky et al.(2012)Krizhevsky, Sutskever, and Hinton] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In Advances in neural information process- ing systems, pp. 1097–1105, 2012.

[Li & Liu(2016)Li and Liu] F Li and B Liu. Ternary weight networks.(2016). arXiv preprint arXiv:1605.04711, 2016.

[NVIDIA(2018)] NVIDIA. Tensorrt document, 2018. URL https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/index.html.

[Sabour et al.(2017)Sabour, Frosst, and Hinton] Sara Sabour, Nicholas Frosst, and Geoffrey E Hinton. Dynamic routing between capsules. In Advances in Neural Information Processing Systems, pp. 3856–3866, 2017.

[Srivastava et al.(2015)Srivastava, Greff, and Schmidhuber] Rupesh Kumar Srivastava, Klaus Greff, and Ju ̈rgen Schmidhuber. Highway networks. arXiv preprint arXiv:1505.00387, 2015.

[Szegedy et al.(2016)Szegedy, Ioffe, and Vanhoucke] C Szegedy, S Ioffe, and V Vanhoucke. Inception-v4, inception-resnet and the impact of residual connections on learning. corr abs/1602.07261. URL http://arxiv.org/abs/1602.07261, 2016.

[Wu & He(2018)Wu and He] Yuxin Wu and Kaiming He. Group normalization. arXiv preprint arXiv:1803.08494, 2018.