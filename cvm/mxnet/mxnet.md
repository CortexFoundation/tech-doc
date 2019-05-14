---
typora-copy-images-to: ../mxnet
---

### Quantizing NN models for deployment on blockchain

##  

Towards A Novel Deterministic Inference Infrastructure on Blockchain

## Introduction

There are emerging interests on delpoying deep learning models on various platforsm and devices. Especially, deep networks are increasingly used for applications at the edge. Devices at the edge typically have lower compute capabilities and are constrained in memory and power consumption.  This situation is more critial on deploying DNN models on blockchain, which **much more poor**. In addition to limited computation resource, determinstic is another issue, e.g. each running of single model on different device must produce bit-level identical result. One of the nondeterministic comes from the float-point number arithmetic, e.g. summation over a series of float-point number. 

In the post, we propose a metholody to both accelate DNN models' inference and eliminate nondeterministic behavior in model inference for blockchain adoption. Before we go into the detail of  implementation, we first go through the oberservation and intuition befind this metholody.

In term of edge computing, unlike GPU, float point unit is less effective on edge device in usual. Thus, researchers have propsed serveral approaches to tackle this problem:

1. **Fake Quantization**: quantizing float-point number into 8-bit interger and transfer data to accelator, which takes linear time to apply this operation. The most costly part of calculation, e.g. conv,  only happens in accelator that dedicated in 8-bit arithmetic. Afterward, results is transformed back to float-point.
2. **Integer-Only Inference**: quantization scheme that allows inference to be carried out using integer-only arithmetic, which can be implemented more efficiently than floating point inference on commonly available integer-only hardware. Fine-tune proceduce is usally utilized to preserve model accuracy after post quantization

The current implmentation in MXNet's contrib libaray follows fake quantization routine and redirect the compuation to MKLDNN math libaray. However, in blockchain's determinstic sensitive scenario, float-point number is not unacceptable. Therefore, we adopt integer-only inference as our methodology. In addition, numerical bound is checked to avoid integer overflow by ultizing operation level rewriting. 

## Implementation

According to above disscusion, we implemented a converter using MXNet's nnvm module, named MRT,  that transform a plain MXNet model into our Cortex Virtual Machine(CVM) represenation.

### Fusion
batchnorm and dropout, rewriting average pooling

### Simulated quantization

using float to simulate quantization. Given all weights and inputs, $$w_i = w’_i * s_w$$, $$s_w$$ is a float number defined as $$s_w=a * 2^b$$, where $a$ and $b$ are both integers. 

We adpot symmetric quantization approach to quantize float-point vector $x$ to signed int8 type $x^Q$, specifally 

​                                                                                      $$\begin{align}x=sx^{Q} \end{align}$$                                                  

 where $x\in \mathbf{R}^{n}, s \in \mathbf{R}, x^Q \in \text{int8}^{n}$

As `matmul` is the core of NN's workflows, we take it as a example to illustrate how to transform float-point operator to integer operator. 



let's define float-point `matmul` as $y = Wx$, where $y\in \mathbf{R}^m, x\in \mathbf{R}^n, W\in \mathbf{R}^{m\times n}$. First we rewrite $x$, $y$  and $W$ into quantized representation $s_y * y^Q   = s_w s_x * W^QX $ , and rewrite it into

​                                                                    $$ \begin{align}\\ y^Q &=(\frac{s_w s_x}  {s_y}) W^QX^Q = s_q W^QX^Q \end{align}$$

where $s_q =\frac{s_w s_x}  {s_y} $ is the requantization scalar, which can be calibrated offline. We will discuss more about the calibration in following section.  

In ususal, $s_q$ is determined in adavance. With calibrated requantization scalar $s_q$ for output $y$ of each operator and weight scalar $ s_w$, we can further determine $s_y$ by definiton. Thus, we can rewrite the original graph to a annotated graph as figure showing befow:

![img]() 

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

commonly, 4x model reize reduction can be achieved. 


| MODEL       | Gluon Model Zoo | CVM  |
| ----------- | :-------------: | ---- |
| ResNetV1_50 |                 |      |
| InceptionV2 |                 |      |
| LeNet       |                 |      |


## Conclusion

Using MXNet’s quantization technology, model inference can be enabled on the limited-resource and strict environment of blockchain, unlocking a novel domain of smart contracts with ml models. The use case could be DeFi, Entertainment, Information service, BaaS, etc.

## Future work

enhancing privacy, accuracy, and efficency. Mobile/edge computing realization is also one of our goals.