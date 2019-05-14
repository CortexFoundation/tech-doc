---
typora-copy-images-to: ../mxnet
---

### Quantizing NN models for deployment on blockchain

##  

Towards A Novel Deterministic Inference Infrastructure on Blockchain

## Introduction

There are emerging interests on delpoying deep learning models on various platforsm and devices. Especially, deep networks are increasingly used for applications at the edge. Devices at the edge typically have lower compute capabilities and are constrained in memory and power consumption.  This situation is more critial on deploying DNN models on blockchain, which **much more poor**. In addition to limited computation resource, determinstic is another issue, e.g. each running of single model on different device must produce bit-level identical result. One of the nondeterministic comes from the float-point number arithmetic, e.g. summation over a series of float-point number. 

In the post, we propose a metholody to both accelate DNN models' inference and eliminate nondeterministic behavior in model inference for blockchain adoption. Before we go into the detail of  implementation, we first go through the oberservation and intuition befind this metholody.

In term of edge computing, unlike GPU, float point unit is less effective on edge device in usual. Thus, researchers propsed serveral approaches to tackle this problem:

1. **Fake Quantization**: quantizing float-point number into 8-bit interger and transfer data to accelator, which takes linear time to apply this operation. The most costly part of calculation, e.g. conv,  only happens in accelator that dedicated in 8-bit arithmetic. Afterward, results is transformed back to float-point.
2. **Integer-Only Inference**: quantization scheme that allows inference to be carried out using integer-only arithmetic, which can be implemented more efficiently than floating point inference on commonly available integer-only hardware. Fine-tune proceduce is usally utilized to preserve model accuracy after post quantization

## Implementation

for proof of concept, we leverage the following techniques:

### Fusion
batchnorm and dropout, rewriting average pooling

![graph_trans](/Users/ml_pm/Code/tech-doc/cvm/mxnet/graph_trans.png)

### Simulated quantization

using float to simulate quantization. Given all weights and inputs, $$w_i = w’_i * s_w$$, $$s_w$$ is a float number defined as $$s_w=a * 2^b$$, where $a$ and $b$ are both integers. 



### Calibrating Requantization Parameter

estimating requantization bits for activation layer

1. Trivial approach: projecting $[a_{min}, a_{max}]$ to $[-127, 127]$
2. MXNet approach: entropy based requantization

### OPs rewriting for numerical-bound

* rewrite computation graph to avoid nondeterministic behaviors in parallel computations, which is common in ML accelerators.

## Experiment

#### 1. Reduce OPs

a **table** showing comparasion of OPs between cvm(int8) and mxnet(float), ususally int8 can run 4x faster than float.

#### 2. model size reduction

commonly, 4x model reize reduction can be achieved


| MODEL       | GluonModelZoo | CVM  |
| ----------- | ------------- | ---- |
| ResNetV1_50 |               |      |
| InceptionV2 |               |      |
| LeNet       |               |      |


## Conclusion

Using MXNet’s quantization technology, model inference can be enabled on the limited-resource and strict environment of blockchain, unlocking a novel domain of smart contracts with ml models. The use case could be DeFi, Entertainment, Information service, BaaS, etc.

## Future work

enhancing privacy, accuracy, and efficency. Mobile/edge computing realization is also one of our goals.