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

##### Fuse Constant

After all the fusion process as belows, we do constant-fuse process for reducing graph complexity and better quantization performance.

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

do nothing in inference and strip it.

#### _div_scalar, _mul_scalar

To aviod division in INT8 graph, we use operator `broadcast_mul` to fuse scalar operator above.

#### slice_like

We can infer shapes within internal symbols in graph, and the output shape of *slice_like* operator can be inferred. To reduce dependence between symbols, operator *slice* is enough for specific output shape.

### Simulated quantization

We adpot symmetric quantization approach to quantize float-point vector $x$ to signed int8 type $x^Q$, specifically

​                                                                                      $$\begin{align}x=sx^{Q} \end{align}$$                                                  

 where $x\in \mathbf{R}^{n}, s \in \mathbf{R}, x^Q \in \text{int8}^{n}$

As `matmul` is the core of NN's workflows, we take it as a example to illustrate how to transform float-point operator to integer operator. 



let's define float-point `matmul` as $y = Wx$, where $y\in \mathbf{R}^m, x\in \mathbf{R}^n, W\in \mathbf{R}^{m\times n}$. First we rewrite $x$, $y$  and $W$ into quantized representation $s_y * y^Q   = (s_wW^Q)  s_x  X^Q $ , and rewrite it into

​                                                                    $$ \begin{align}\\ y^Q &=(\frac{s_w s_x}  {s_y}) W^QX^Q = s_q W^QX^Q \end{align}$$

where $s_q =\frac{s_w s_x}  {s_y} $ is the requantization scalar, which can be calibrated offline. We will discuss more about the calibration in following section.  

In ususal, $s_y $ is determined in adavance. With calibrated requantization scalar $s_y$ for output $y$ of each operator and weight scalar $ s_w$, we can further determine $s_y$ by definiton. Thus, we can rewrite the original graph to a annotated graph as figure showing befow:

![img]() 

### Calibrating Requantization Parameter

Estimating requantization scale for activation layer.

Supposed that our purpose is quantizing layer into `bit` of INT, then the range of output for each layer is between $[-clip, clip], clip=2^{bit-1}-1$. Note that we strip the `MIN` value of INT which is $-2^{bit-1}$ for processing symmetric range in simplify quantization, and it shouldn't lead to a lot of acccuracy loss. The process steps is as belows:

1. Calibrate the output for each internal layer with calibration data. 

   The result is represented as $[a_{min}, a_{max}]$. And there are two calibration approaches in mainly:

   - Trivial approach: naive calibration, projecting $[a_{min}, a_{max}]$ to $[min(out), max(out)]$.
   - MXNet approach: entropy based calibration.

2. Calculate the scale of activation layer to target INT bit, equation is as belows:

   ```python
   alpha = absmax(amin, amax)
   scale = alpha / clip
   ```

   Note: we have tried a simple method in real-environment quantization, which use shift bit instead of floating scale for requantization that will reduce work in symbol realizing. In this case, scale is int times of power 2 specifically. And the relative inference is:

   ```python
   scale = alpha / clip ie. target_range = alpha / scale = clip
   we calculate shift_bit with sb = ceil(log2(scale)) 
   ie. scale <= 2 ** (sb) < scale * 2
   so that we can get target range: 
   	target_range = alpha / (2 ** sb) <= alpha / scale = clip
     target_range = alpha / (2 ** sb) > alpha / (scale * 2) = clip / 2 
   ```

   So the target range is not full of INT `bit`, it's max value is between $(clip/2, clip]$. Usually, the target range is decreased a lot, and accuracy after quantization is a bit lower due to this. But it do reduce the work of realizing requantization operator for the scale in the next steps.

   More details please infer to section [Realize](#calibrating-requantization-parameter).

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