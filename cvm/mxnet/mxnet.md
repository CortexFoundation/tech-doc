---
typora-copy-images-to: ../mxnet
---

### Quantizing NN models for deployment on blockchain

##  

Towards A Novel Deterministic Inference Infrastructure on Blockchain

## Introduction

There are emerging interests in deploying deep learning models on various platforms and devices. Especially, deep networks are seeing increasingly used for applications at the edge devices which typically have lower compute capabilities and are constrained in memory and power consumption. Due to limited-resource and strict environment, the situation is more critical to deploy DNN models on the blockchain. In addition to limited computation resource, being deterministic is another issue, e.g. each running of a single model on the different device must produce a bit-level identical result. Nondeterministic occurs from the float-point number arithmetic, e.g. summation over a series of float-point number. 

In this post, we propose a methodology to both accelerate DNN models' inference and eliminate nondeterministic behavior in model inference for blockchain adoption. Before we go into the detail of implementation, we first go through the observation and intuition behind this methodology.

In term of edge computing, unlike GPU, float point unit is less effective and desirable on edge device. Thus, researchers have proposed several approaches to tackle this problem:

1. **Fake Quantization**: quantizing float-point number into 8-bit integer and transfer data to the accelerator, which takes linear time to apply this operation. The most costly part of the calculation, e.g. conv,  only happens in the accelerator that dedicated in 8-bit arithmetic. Afterward, results are transformed back to float-point.
2. **Integer-Only Inference**: quantization scheme that allows inference to be carried out using integer-only arithmetic, which can be implemented more efficiently than floating point inference on commonly available integer-only hardware. Fine-tune procedure is usually utilized to preserve model accuracy post-quantization

The current implementation in MXNet's Contrib library follows fake quantization routine and redirect the computation to MKLDNN math library. However, in blockchain's deterministic sensitive scenario, the float-point number is unacceptable. Therefore, we propose integer-only inference as our methodology. In addition, the numerical bound is checked to avoid integer overflow by utilizing operation level rewriting. 

## Implementation

According to the above discussion, we implemented a converter using MXNet's nnvm module, named MRT,  that transform a plain MXNet model into our Cortex Virtual Machine(CVM) representation.

### Fusion
batchnorm and dropout, rewriting average pooling

##### Fuse Constant

After all the fusion process as below, we do constant-fuse process for reducing graph complexity and better quantization performance.

##### Matrix Decomposition

Big matrix process may exceed the max-length : `range(INT32) / range(8 + 8)` in computation, which result may be out of range INT32. Matrix decomposition is to do for matrix computation scale reduction.
$$
A * W = \sum_{start, stop} A[:,start:stop] * W[:, start:stop]
$$

#### Pooling

*pool_type*: indicates pooling type, only supported `avg` and `max`.
*is_global*: indicates is global average pooling.
*pooling_convention*: must be `valid` for equivalent transformation with depthwise *Convolution*.
*count_include_pad*: must be `true` for equivalent transformation with depthwise *Convolution*.
*kernel, stride, pad*: pooling kernel, stride and pad attributes.

##### GlobalAvgPooling

```python
GlobalAvgPooling(data) =
```

$$
\sum_{k_i, k_j}^{kernel}data[:,:,k_i,k_j] / (size_{kernel})
$$

```python
= broadcast_mul(sum(data, axis=(2, 3)), scale), which scale equals 1 / K / K
```

##### AvgPooling

```python
AvgPooling(data) = Convolution(data,
            kernel=kernel, stride=stride, pad=pad, # depthwise conv2d
            no_bias=True, dilate=(1, 1), layout='NCHW',
            num_filter=in_channel, num_group=in_channel)
```

#### LeakyReLU

*act_type*: action type, only supported `leaky`.
*slope*: attribute.

```python
LeakyReLU(data) = relu(data) - slope * relu(-data)
```

#### BatchNorm

*gamma, beta, data_mean, data_var*: attributes.

```python
BatchNorm(data) =
```

$$
out[:,i,:...] 
= {data[:,i,:...] - data\_mean[i] \over data\_var[i]} * gamma[i] + beta[i]
= data[:,i:...] * \alpha + \beta
$$

, where $\alpha$ is gamma / data_var and $\beta$ is beta - data_mean * gamma / data_var.

while data is *Convolution*(x), we can get equation as belows:
$$
out[:,i,:...] 
= (X * W + B) * \alpha + \beta 
= X * (W * \alpha) + (B * \alpha + \beta) \\
= Convolution(X, weight=W_{new}, bias=B_{new})
$$

#### Dropout

do nothing in inference.

#### _div_scalar, _mul_scalar

To avoid division in INT8 graph, we use the operator `broadcast_mul` to fuse scalar operator above.

#### slice_like

We can infer shapes within internal symbols in a graph, and the output shape of *slice_like* operator can be inferred. To reduce dependence between symbols, operator *slice* is enough for specific output shape.

### Simulated quantization

We adopt a symmetric quantization approach to quantize float-point vector $x$ to signed int8 type $x^Q$, specifically

​                                                                                      $$\begin{align}x=sx^{Q} \end{align}$$                                                  

 where $x\in \mathbf{R}^{n}, s \in \mathbf{R}, x^Q \in \text{int8}^{n}$

As `matmul` is the core of NN's workflows, we take it as an example to illustrate how to transform float-point operator to an integer operator. 



let's define float-point `matmul` as $y = Wx$, where $y\in \mathbf{R}^m, x\in \mathbf{R}^n, W\in \mathbf{R}^{m\times n}$. First we rewrite $x$, $y$  and $W$ into quantized representation $s_y * y^Q   = (s_wW^Q)  s_x  X^Q $ , and rewrite it into

​                                                                    $$ \begin{align}\\ y^Q &=(\frac{s_w s_x}  {s_y}) W^QX^Q = s_q W^QX^Q \end{align}$$

where $s_q =\frac{s_w s_x}  {s_y} $ is the requantization scalar, which can be calibrated offline. We will discuss more about the calibration in following section.  

In usual, $s_y $ is determined in advance. With calibrated requantization scalar $s_y$ for output $y$ of each operator and weight scalar $ s_w$, we can further determine $s_y$ by definition. Thus, we can rewrite the original graph to an annotated graph as the figure showing below:

![img](/Users/oscarwei/Dropbox/markdown/tech-doc/cvm/mxnet/graph_trans.png) 

### Calibrating Requantization Parameter

estimating requantization bits for the activation layer

1. Trivial approach: projecting $[a_{min}, a_{max}]$ to $[-127, 127]$
2. MXNet approach: entropy based requantization

### OPs rewriting for numerical-bound

* rewrite computation graph to avoid nondeterministic behaviors in parallel computations, which is common in ML accelerators.

## Experiment

#### 1. Reduce OPs

a **table** showing the comparison of OPs between cvm(int8) and mxnet(float), usually, int8 can run 4x faster than float.

#### 2. model size reduction

commonly, 4x model size reduction can be achieved. 


| MODEL       | Gluon Model Zoo | CVM  |
| ----------- | :-------------: | ---- |
| ResNetV1_50 |                 |      |
| InceptionV2 |                 |      |
| LeNet       |                 |      |


## Conclusion

Using MXNet’s quantization technology, model inference can be enabled on the limited-resource and strict environment of blockchain, unlocking a novel domain of smart contracts with ml models. The use case could be DeFi, Entertainment, Information service, BaaS, etc.

## Future work

Enhancing privacy, accuracy, and efficiency. Mobile/edge computing realization is also one of our goals.