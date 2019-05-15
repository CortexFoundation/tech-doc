---
typora-copy-images-to: ../mxnet
---

### Quantizing NN models for deployment on blockchain

Towards A Novel Deterministic Inference Infrastructure on Blockchain

## Introduction

There are emerging interests in deploying deep learning models on various platforms and devices. Especially, deep networks are seeing increasingly used for applications at the edge devices which typically have lower compute capabilities and are constrained in memory and power consumption. Due to limited-resource and strict environment, the situation is more critical to deploy DNN models on the blockchain. In addition to limited computation resource, being deterministic is another issue, e.g. each running of a single model on the different device must produce a bit-level identical result. Nondeterministic occurs from the float-point number arithmetic, e.g. summation over a series of float-point number. 

In this post, we propose a methodology to both accelerate DNN models' inference and eliminate nondeterministic behavior in model inference for blockchain adoption. Before we go into the detail of implementation, we first go through the observation and intuition behind this methodology.

In term of edge computing, unlike GPU, float point unit is less effective and desirable on edge device. Thus, researchers have proposed several approaches to tackle this problem:

1. **Fake Quantization**: quantizing float-point number into 8-bit integer and transfer data to the accelerator, which takes linear time to apply this operation. The most costly part of the calculation, e.g. conv,  only happens in the accelerator that dedicated in 8-bit arithmetic. Afterward, results are transformed back to float-point.
2. **Integer-Only Inference**: quantization scheme that allows inference to be carried out using integer-only arithmetic, which can be implemented more efficiently than floating point inference on commonly available integer-only hardware. Fine-tune procedure is usually utilized to preserve model accuracy post-quantization

The current implementation in MXNet's Contrib library follows fake quantization routine and redirect the computation to MKLDNN math library. However, in blockchain's deterministic sensitive scenario, the float-point number is unacceptable. Therefore, we propose integer-only inference as our methodology. In addition, the numerical bound is checked to avoid integer overflow by utilizing graph level rewriting. 

## Implementation

According to the above discussion, we implemented a converter using MXNet's nnvm module, named MRT,  that transform a plain MXNet model into our Cortex Virtual Machine(CVM) representation.

### Fusion and Opeartor Rewriting
##### Fuse Constant

After all the fusion process as below, we do constant-fuse process for reducing graph complexity and better quantization performance.

##### MAC Decomposition

suppose we are calculating the inner dot of two vector $x \in Z_{\text{int8}}^{n}$ and $y \in Z_{\text{int8}}^{n}$, which may results in a 32-bit integer, sepecifically $s=<x, y> = \sum_i^n x_i y_i \in  Z_{\text{int32}}^{n}$ . However, this condition of numerical bound is only hold when $n$ is less $2^{16}$. In another word, we cannot assume abense of overflow when $n$ is large, which may introduce nondeterministic behavior during parallel computing. To resolve this problem, we can decomposite the computation into small peices in graph level and aggreate the results, mathmatically $s=<x^{(1)}, y^{(1)}>+<x^{(2)}, y^{(2)}> + … + <x^{(K)}, y^{(K)}>$, $x^{(k)}$ is the $k$-th part of vector $x$. Each parts of vector has length small than $2^{16}$.

Matrix multiplication operator `matmul` can also rewrite in this fashion, which results in a series of `elemwise_add` operator that sum over several intermediate matries. Although, this rewriting will introduce additional opeators in computation graph, semantic will remain unchanged.

#### Pooling

*pool_type*: indicates pooling type, only supported `avg` and `max`.
*is_global*: indicates is global average pooling.
*pooling_convention*: must be `valid` for equivalent transformation with depthwise *Convolution*.
*count_include_pad*: must be `true` for equivalent transformation with depthwise *Convolution*.
*kernel, stride, pad*: pooling kernel, stride and pad attributes.

##### Rewrite GlobalAvgPooling 

```python
GlobalAvgPooling(data) =
```

$$
\sum_{k_i, k_j}^{kernel}data[:,:,k_i,k_j] / (size_{kernel})
$$

```python
= broadcast_mul(sum(data, axis=(2, 3)), scale), which scale equals 1 / K / K
```

##### Rewrite AvgPooling

```python
AvgPooling(data) = Convolution(data,
            kernel=kernel, stride=stride, pad=pad, # depthwise conv2d
            no_bias=True, dilate=(1, 1), layout='NCHW',
            num_filter=in_channel, num_group=in_channel)
```

#### Rewrite LeakyReLU

*act_type*: action type, only supported `leaky`.
*slope*: attribute.

```python
LeakyReLU(data) = relu(data) - slope * relu(-data)
```

#### Fuse BatchNorm

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
= \text{Convolution}(X, \text{weight}=W_{new}, \text{bias}=B_{new})
$$

#### Rewrite Dropout

do nothing in inference and strip it.

#### Rewrite _div_scalar, _mul_scalar

To avoid division in INT8 graph, we use the operator `broadcast_mul` to rewrite scalar operator above.

### Simulated quantization

Before we can make whole computational graph integer-only, we should firstly rewrite float-point number into simulated quantization representation. In current implementation, we adopt a symmetric quantization approach to quantize float-point vector $x$ to signed int8 type $x^Q$, specifically

​                                                                                      $$\begin{align}x=sx^{Q} \end{align}$$                                                  

 where $x\in \mathbf{R}^{n}, s \in \mathbf{R}, x^Q \in Z_{\text{int8}}^n$

After quantization pass applied, we can reorder the operators in graph in order to further processing.  As `matmul` is the core of NN's workflows, we take it as an example to illustrate how to transform float-point operator to an integer operator. 

let's define float-point `matmul` as $y = Wx$, where $y\in \mathbf{R}^m, x\in \mathbf{R}^n, W\in \mathbf{R}^{m\times n}$. First we rewrite $x$, $y$  and $W$ into quantized representation $s_y * y^Q   = (s_wW^Q)  s_x  X^Q $ , and rewrite it into

​                                                                    $$ \begin{align}\\ y^Q &=(\frac{s_w s_x}  {s_y}) W^QX^Q = s_q W^QX^Q \end{align}$$

where $s_q =\frac{s_w s_x}  {s_y} $ is the requantization scalar.

In our approach, scalar $s_y $ is determined in advance by calibration. With calibrated scalar $s_y$ for output $y$ of each operator and weight scalar $ s_w$, we can further determine requantization scalar $s_q$ by definition. Thus, we can rewrite the original graph to an annotated graph as the figure showing below:

![img]()

### Calibrating Requantization Parameter

Estimating requantization scale for activation layer.

Supposed that our purpose is quantizing layer into `bit` of INT, then the range of output for each layer is between $[-clip, clip], clip=2^{bit-1}-1$. Note that we strip the `MIN` value of INT which is $-2^{bit-1}$ for processing symmetric range in simplify quantization, and it shouldn't lead to a lot of acccuracy loss. The process steps is as belows:

1. Calibrate the output for each internal layer with calibration data. 

   The result is represented as $[a_{min}, a_{max}]$. And there are two calibration approaches in mainly:

   - Trivial approach: naive calibration, projecting $[a_{min}, a_{max}]$ to $[\min(out), \max(out)]$.
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
