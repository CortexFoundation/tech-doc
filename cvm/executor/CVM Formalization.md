# CVM Formalization

### Predefined Structure

#### TShape

struct `TShape`, a array of `uint32_t`, which is used to standing for data shape.

#### Optional\<T\>

the value can be `None`, or type of T.

#### DLTensor

Data format，a wrapper of data content pointer.

*Attributes*

- dtype must be INT8 or INT32.

- shape is `TShape` type, storing data shape, for example (1, 2, 3).

- precision is the max bit of each element take up, -1 stands for non-defined; normal value betweens range (0, 32]; others is non-available, which means a error occurs, either logic error or runtime error.

  *Notes*: we define a `p` precision data is between range $[-\alpha, \alpha], \alpha=2^{p-1}-1$ for all elements, for example: data that precision is 8 means all value is larger than -128 and less than 128, this is strong constraint.

*Public Interface*

- ndim is the dimensions of data, for example 3.

#### Attribute Constant

*max_attr* = 4096

*min_attr* = 0

## Executor

## Ops

本节主要介绍 *cvm executor* 的算子形式化描述，包含但不限于

1. 输入，输出和参数限制 (restriction of inputs and outputs)
2. 源代码引用 (source code reference)
3. 数学形式化描述 (math formalization)

##### Inputs & Outputs

算子输入输出的数据格式为`DLTensor`，precision属性在(0, 32]范围内。

### Reduce Operator

Reduce operator perform the reduction function to input data based on the parameters, and the process logic over all the type-based operators is consistent. We abstract the formalization here and introduce the details as belows:

*Attributes Constraints*

1. axes is `TShape` type, by default (). The axis or axes along which to perform the reduction.
   + no duplicate
   + range [-ndim, ndim). If axis < 0, axis += ndim
2. keepdims is `false` (`boolean` type) by default. If this is set to `True`, the reduced axes are left with 1.
3. exclude is `false` by default. Whether to perform reduction on axis that are NOT in axis instead.

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axes`, `exclude` and reduce function `REDUCE_OP`.

```python
real_axes = calculate the real axes depending on (axes, exclude)
```

Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/reduce.cc#L6

1. len(real_axes) == 0 and exclude == true 

   Math:	$Y = X$

   Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/reduce.cc#L42

2. len(real_axes) == 0 and exclude == false

   Math: 

   ```python
   Y[0] = X[0] # do reduce op over all axis
   for x in X[1:]:
     Y[0] = REDUCE_OP(x, Y[0])
   ```

   Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/reduce.cc#L45

3. other case

   Math:

   ```python
   init Y with 0
   for xidx in range(X.shape):
   	yidx = [x if ix not in real_axes for ix in xidx]
   	Y[yidx] = REDUCE_OP(Y[yidx], X[xidx])
   ```

   Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/reduce.cc#L52


#### max

set `REDUCE_OP` to be `max`.

#### sum

set `REDUCE_OP` to be `sum`.

Example:

```python
data = [[[1, 2], [2, 3], [1, 3]],
        [[1, 4], [4, 3], [5, 2]],
        [[7, 1], [7, 2], [7, 3]]]

sum(data, axis=(1))
[[  4.   8.]
 [ 10.   9.]
 [ 21.   6.]]

sum(data, axis=[1,2])
[ 12.  19.  27.]
```

### Broadcast Operator

Broadcast operator perform the broadcast function to input datas, and the process logic over all the type-based operators is consistent. We abstract the formalization here and introduce the details as belows:

*Math Formalization*

Suppose Input `A`, `B`, Output `Y` and broadcast function `BROADCAST_OP`. Where `A`'s shape is M dimension, exactly $(m_0, m_1, \cdots, m_{M-1})$, `B`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$.
$$
K = max(M, N) \\
SA[p] = \begin{cases}
m_{p-K+M}, & p \geqslant K - M \\
1, & p < K - M
\end{cases}, p \in [0, K) \\
SB[q] = \begin{cases}
n_{q-K+N}, & q \geqslant K-N \\
1, & q < K - N
\end{cases}, q \in [0, K) \\
\forall \{i \mid i \in [0, K) \}: SA[i]=SB[i] \or SA[i]=1 \or SB[i]=1 \\
\forall (d_0, d_1, \cdots, d_{K-1}), \text{where } d_j \in [0, max(SA[j], SB[j])) \and j \in [0, K): \\
Y[d_0, d_1, \cdots, d_{K-1}] = 
\text{BROADCAST_OP}(A[a_0, a_1, \cdots, a_{M-1}], B[b_0, b_1, \cdots, b_{N-1}]), \\
\text{where } a_i = min(d_{K-M+i}, m_i-1) \and i \in [0, M) \and \\
b_j = min(d_{K-N+j}, n_j-1) \and j \in [0, N)
$$
Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/ops.cc#L575

#### broadcast_add

set `BROADCAST_OP` to `add`.

Example:

```python
x = [[ 1.,  1.,  1.],
     [ 1.,  1.,  1.]]

y = [[ 0.],
     [ 1.]]

broadcast_add(x, y) = [[ 1.,  1.,  1.],
                       [ 2.,  2.,  2.]]
```

#### broadcast_sub

set `BROADCAST_OP` to `sub`.

#### broadcast_mul

set `BROADCAST_OP` to `multiply`.

#### broadcast_max

set `BROADCAST_OP` to `max`.

### NN Operator

#### Convolution

We only supported 2-D convolution operator.

*Math Formalization*

Suppose Input `X`, `W`, `B`, and output `Y`, attributes `padding`, `stride`, `dilation`, `groups`, where `X`'s shape is $(N, C, H, W)$, `W`' shape is $(OC, IC, KH, KW)$, and $IC = C \div \text{groups}$, `B` is `Optional<DLTensor>`, if `B` is not None, it's shape is $(\text{OC},)$. `padding` is  2-D `TShape`, exactly $(PH, PW), PH,PW \in [min\_attr, max\_attr)$, `stride` is 2-D `TShape`, exactly $(SH, SW) \in [1, max\_attr)$, `dilation` is 2-D `TShape`, exactly $(DH, DW) \in [1, max\_attr)$, `grous` is `int`, the value is in $\{1, C\}$.

1. Case `groups` = 1

$$
Y[n,i,p,q]=\sum_{j=0}^{C} kernel(X[n,j, p:p+\text{KH}, q:q+\text{KW}], W[i,j,:,:])
+ \begin{cases}
0, & \text{if B is None}\\
B[i], & \text{otherwise}
\end{cases}, \\
\forall \quad n \in [0, N) \and i \in [0, OC) \and \\
p \in \left[0, \left\lfloor{H+2 \cdot \text{PH}-\text{DH} \cdot (\text{KH}-1)-1\over\text{SH}}\right\rfloor+1 \right) \and \\
q \in \left[0, \left\lfloor{W+2 \cdot \text{PW}-\text{DW} \cdot (\text{KW}-1)-1 \over \text{SW}}\right\rfloor+1 \right)
$$
where $kernel$ function is 
$$
kernel(A, B) = \sum_{k_i=0}^{\text{KH}} \sum_{k_j = 0}^{\text{KW}} A[k_i, k_j] \cdot B[k_i, k_j]
$$
Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/ops.cc#L475

2. Case `groups ` = $C$

This case is named *Depth-Wise Convolution*
$$
IC = 1 \and OC = C \\
Y[n,i,p,q]= kernel(X[n,i, p:p+\text{KH}, q:q+\text{KW}], W[i,0,:,:]) + \begin{cases}
0, & \text{if B is None}\\
B[i], & \text{otherwise}
\end{cases}\\
\forall n \in [0, N) \and i \in [0, C) \and\\
p \in \left[0, \left\lfloor{H+2 \cdot \text{PH}-\text{DH} \cdot (\text{KH}-1)-1\over\text{SH}}\right\rfloor+1 \right) \and \\
q \in \left[0, \left\lfloor{W+2 \cdot \text{PW}-\text{DW} \cdot (\text{KW}-1)-1 \over \text{SW}}\right\rfloor+1 \right)
$$
Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/ops.cc#L390

#### Dense

*Math Formalization*

Suppose Input `X`, `W`, `B`, where `X` shape is $(M * K)$, `W` shape is $(N * K)$, `B` is `Optional<DLTensor>`, if `B` is not `NONE`, it's shape is $(N,)$.

Math:
$$
Y=X W^T + \begin{cases}
0, & \text{if B is None} \\
B, & \text{otherwise}
\end{cases}
$$
Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/ops.cc#L86

#### MaxPooling

*Math Formalization*

Suppose Input `X`, Output `Y` and attributes `pool_size`,  `	padding`, `strides`, `ceil_mode`, where `X`'s shape is $(N, C, H, W)$, `pool_size` is 2-D `TShape`, exactly $(PSH, PSW)$, `padding` is 2-D `TShape`, exactly $(PH, PW) \in [min\_attr, max\_attr)$, if `padding` is  1-D, which means $PH = PW$, `strides` is 2-D `TShape`, exactly $(SH, SW)$, `ceil_mode` is `boolean`.

Math:
$$
PSH \in [0, H + 2PH + 1) \and PSW \in [0, W + 2PW + 1)\\
Y[n,i,p,q] = max\{x \mid x \in X[n,i, p:p+\text{PSH}, q:q+\text{PSW}]\}, \\
\forall n \in [0, N) \and i \in [0, C) \and \\
p \in \left[0, \text{ceil_func}\left({H+2 \cdot \text{PH}-  \text{PSH}\over\text{SH}}\right)+1 \right) \and \\
q \in \left[0, \text{ceil_func}\left({W+2 \cdot \text{PW}- \text{PSW} \over \text{SW}}\right)+1 \right) \and \\
\text{ceil_func(val)} = \begin{cases}
\lceil \text{val} \rceil, & \text{if ceil_mode is true} \\
\lfloor \text{val} \rfloor, & \text{otherwise}
\end{cases}
$$
Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/ops.cc#L692

#### Upsampling

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `scale`, where `X`'s shape is $(N, C, H, W)$, `scale` is in range $[1, max\_attr)$.
$$
Y[n, i, h, w] = X[n, i, \left\lfloor {h \over \text{scale}}\right\rfloor, \left\lfloor {w \over \text{scale}}\right\rfloor], \\
\forall n \in [0, N) \and i \in [0, C) \and 
h \in [0, H \cdot \text{scale}) \and w \in [0, W \cdot \text{scale})
$$


### Elemwise Operator

#### abs

*Math Formalization*

Suppose Input `X`, Output `Y`.

Math:
$$
y = \begin{cases}
x, &  x \geqslant 0  \\
-x, & x < 0 
\end{cases},
\forall x \in X
$$

#### cvm_precision

*Math Formalization*

Suppose Input `X`, Output `Y`.

Math:
$$
y = \begin{cases}
\lceil log_2(abs(x+1)) \rceil, & x \neq 0\\
1, & x = 0 
\end{cases}, 
\forall x \in X
$$

#### elemwise_add

*Math Formalization*

Suppose Input `A`, `B`, Output `Y`. Where `A` and `B` have the same shape, $(n_0, n_1, \cdots, n_{N-1})$.
$$
Y = A + B
$$

#### elemwise_sub

*Math Formalization*

Suppose Input `A`, `B`, Output `Y`. Where `A` and `B` have the same shape, $(n_0, n_1, \cdots, n_{N-1})$.
$$
Y = A - B
$$

#### negative

*Math Formalization*

Suppose Input `X`, Output `Y`.
$$
Y = -X
$$

#### clip

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `a_min`, `a_max`.
$$
y = \begin{cases}
\text{a_max}, & x \geqslant \text{a_max} \\
x, & x \in (\text{a_min}, \text{a_max}) \\
\text{a_min}, & x \leqslant \text{a_min}
\end{cases},
\forall x \in X
$$


#### cvm_clip

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `precision`, where `precision` is in range $[1, 33)$.
$$
Y = clip(X, \text{a_min}=-\alpha, \text{a_max}=\alpha), \\
\text{where } \alpha = 2^\text{precision-1}-1
$$


#### cvm_right_shift

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `precision`, `shift_bit`, where `precision` is in range $[1, 33)$, and `shift_bit` is in range $[1, 33)$.
$$
Y = clip(T, \text{a_min} = -\alpha, \text{a_max}=\alpha), \\
\text{where } T = {\left\lfloor 
\left(\left\lfloor \frac{X}{2^{\text{shift_bit}} - 1} \right\rfloor + 1 \right) 
\div 2 \right\rfloor} \and \alpha = 2 ^ {\text{precision} - 1} - 1
$$

#### cvm_left_shift

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `precision`, `shift_bit`, where `precision` is in range $[1, 33)$, and  `shift_bit` is in range $[1, 33)$.
$$
Y = clip(T, \text{a_min} = -\alpha, \text{a_max}=\alpha), \\
\text{where } T = X * 2^\text{shift_bit} \and \alpha = 2 ^ {\text{precision} - 1} - 1
$$

### Transform Operator

#### repeat

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `axis`, `repeats`. Where `X`'s shape is N dimension, exactly  $(n_0, n_1, \cdots, n_{\text{axis}}, \cdots, n_{N-1})$, `Y`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{\text{axis}} \cdot repeats, \cdots, n_{N-1})$. Obviously, axis is in range $[0, N)$, repeats is in range $[1, +\infty)$.
$$
\forall \{(d_0, d_1, \cdots, d_{N-1}) \mid d_j \in 
\begin{cases} 
[0, n_j), & j \ne \text{axis} \\
[0, n_j \cdot \text{repeats}), & j = \text{axis}
\end{cases} \and j \in [0, N) \}: \\
Y[d_0, d_1, \cdots, d_\text{axis}, \cdots, d_{N-1}] = 
X[d_0, d_1, \cdots, \left\lfloor{d_\text{axis} \over \text{repeats}}\right\rfloor, \cdots, d_{N-1}]
$$


#### tile

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `reps`. Where `X`'s shape is N dimension, exactly  $(n_0, n_1, \cdots, n_{N-1})$, `reps` is M dimension, exactly $(m_0, m_1, \cdots, m_{M-1})$.
$$
\forall r \in \text{reps}: r \in [1, max\_attr) \\
\forall (k_0, k_1, \cdots, k_{K-1}), 
\text{where } k_j \in [0, SX[j] \cdot SR[j]) \and j \in [0, K) \and K=max(M, N): \\
Y[k_0, \cdots, k_{K-N-1}, k_{K-N}, k_{K-N+1}, \cdots, k_{K-1}] = \\
X[k_{K-N+0} \text{ mod } n_0, k_{K-N+1} \text{ mod } n_1, \cdots, k_{K-N+N-1} \text{ mod } n_{N-1}], \\
\text{where } SX[p] = \begin{cases}
n_{p-K+N}, & p \in [K-N, K-1) \\
1, & p \in [0, K-N)
\end{cases} \and p \in [0, K) \and \\
SR[q] = \begin{cases}
m_{q-K+M}, & q \in [K-M, K-1) \\
1, & q \in [0, K-M)
\end{cases} \and q \in [0, K)
$$


#### flatten

*Math Formalization*

Suppose Input `X`, Output `Y`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$.
$$
\forall (d_0, d_1, \cdots, d_{N-1}), \text{where } d_j \in [0, n_j) \and j \in [0, N): \\
Y[flatten\_index(d_0, d_1, \cdots, d_{N-1}, n_0, n_1, \cdots, n_{N-1})]  =  \\
X[d_0, d_1, \cdots, d_{N-1}]
$$
where $flatten\_index$ is 
$$
flatten\_index(d_0, d_1, \cdots, d_{N-1}, n_0, n_1, \cdots, n_{N-1}) = \\
d_0 * n_1 * n_2 * \cdots * n_{N-1} + d_1 * n_2 * n_3 * \cdots * n_{N-1} + 
\cdots + d_{N-2} * n_{N-1} + d_{N-1}
$$


#### concatenate

*Math Formalization*

Suppose `M` Inputs $I^0, I^1, \cdots, I^{M-1}$, Output `Y`, attribute `axis`. Where all inputs' shape is N dimension, exactly $I^i$'s shape is $(n^i_0, n^i_1, \cdots, n^i_{N-1})$, and `axis` is in range $[0, N)$.
$$
\forall i \in [1, M) \; \forall j \in [0, N) \and j \ne \text{axis}: \;
n^i_j = n^{0}_j \\
\forall i \in [0, M) \; \forall (d_0, d_1, \cdots, d_{N-1}), \text{where } d_j \in [0, n^i_j) \and j \in [0, N): \\
Y[d_0, d_1, \cdots, d_\text{axis-1}, \text{new_idx}, d_\text{axis+1}, \cdots, d_{N-1}] = I^i[d_0, d_1, \cdots, d_{N-1}], \\
\text{where new_idx} = n^0_\text{axis} + n^1_\text{axis} + \cdots + n^{i-1}_\text{axis} + d_\text{axis}
$$


#### expand_dims

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axis`, `num_newaxis`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$,  `axis` is in range $[-N-1, N+1)$,  and `num_newaxis` is in range $[min\_attr, max\_attr)$.
$$
\forall (d_0, d_1,\cdots, d_{N-1}), \text{where } d_j \in [0, n_j) \and j \in [0, N): \\
Y[d_0,d_1, \cdots, d_{axis-1}, \underbrace{1, 1, \cdots, 1}_{\text{num_newaxis}}, d_\text{real_axis}, \cdots, d_{N-1}] = X[d_0, d_1, \cdots, d_{N-1}], \\ 
\text{where } \text{real_axis} = 
\begin{cases}
\text{axis},& \text{axis} \geqslant 0 \\
\text{axis} + N,& \text{axis} < 0
\end{cases}
$$


#### reshape

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `target_shape`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, and `target_shape` is M dimension, exactly $(m_0, m_1, \cdots,  m_{M-1})$ , and satisfy constraint : $m_0 * m_1 * \cdots * m_{M-1} = n_0 * n_1 * \cdots * n_{N-1}$.
$$
T = flatten(X) \\
\forall (d_0, d_1, \cdots, d_{N-1}), \text{where } d_j \in [0, m_j) \and j \in [0, M): \\
Y[d_0, d_1, \cdots, d_{N-1}] = T[flatten\_index(d_0, d_1, \cdots, d_{N-1}, n_0, n_1, \cdots, n_{N-1})]
$$


#### squeeze

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axes`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, and `axes` is TShape and dimension is M.
$$
\forall \text{axis} \in \text{axes}: axis \in [-N, N) \\
\text{real_axes} = 
\begin{cases}
\{\text{axis} \mid \text{axis} \geqslant 0 \and \text{axis} \in \text{axes} \} \bigcup
\{\text{axis} + N \mid \text{axis} < 0 \and \text{axis} \in \text{axis}\},
& M > 0 \\
\{\text{axis} \mid n_\text{axis} = 1 \and \text{axis} \in [0, N) \}, & M = 0
\end{cases} \\
\forall \text{axis} \in \text{real_axes}: n_{axis} = 1 \\
\forall (d_0, d_1, \cdots, d_{N-1}), \text{where } d_j \in [0, n_j) \and j \in [0, N): \\
Y[s_0, s_1, \cdots, s_{N-K-1}] = X[d_0, d_1, \cdots, d_{N-1}], \\
\text{where } K = \mathbf|\text{real_axes}\mathbf| \and s_{i-count(i)} = d_i 
\and i \in [0, N) \and i \notin \text{real_axes} \and \\
count(i) = \mathbf|\{\text{axis} \mid \text{axis} < i \and \text{axis} \in \text{real_axes} \}\mathbf|
$$


#### transpose

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axes`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, and `axes` is TShape and dimension is M, where M is in $\{0, N\}$.
$$
\forall \text{axis} \in \text{axes}: \text{axis} \in [-N, N) \\
\text{real_axes} = \begin{cases}
(r_0, r_1, \cdots, r_{N-1}), \text{where } r_j = 
\begin{cases} 
\text{axes}_j, & \text{axes}_j \geqslant 0 \\
\text{axes}_j + N, & \text{axes}_j < 0 
\end{cases} \and
j \in [0, N), & M=N\\
(n_{N-1}, n_{N-2}, \cdots, n_0), & M = 0
\end{cases} \\
\{i \mid i \in \text{real_axes}\} = \{j \mid j \in [0, N) \} \\
\forall (d_0, d_1, \cdots, d_{N-1}), 
\text{where } d_j \in [0, n_{\text{real_axes}_{j}}) \and j \in [0, N): \\
Y[d_0, d_1, \cdots, d_{N-1}] = X[s_0, s_1, \cdots, s_{N-1}], \\
\text{where } s_{\text{real_axes}[j]} = d_j \and j \in [0, N)
$$


#### slice | strided_slice

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `begin`, `end`, `strides`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, and `begin`'s shape is B dimension, `end`'s shape is E dimension, `stride`'s shape is S dimension.
$$
\text{b_arr}[b] = \begin{cases}
\text{begin}[b] + n_i, & b \in [0, N) \and b < B \and begin[b] < 0 \\
\text{begin}[b], & b \in [0, N) \and b < B \and begin[b] \geqslant 0 \\
0, & b \in [0, N) \and b \geqslant B
\end{cases}, b \in [0, N) \\
\text{e_arr}[e] = \begin{cases}
\text{end}[e] + n_i, & e \in [0, N) \and e < E \and end[e] < 0\\
\text{end}[e], & e \in [0, N) \and e < E \and end[e] \geqslant 0\\
n_{e}, & e \in [0, N) \and e \geqslant E
\end{cases}, e \in [0, N) \\
\text{s_arr}[s] = \begin{cases}
\text{stride}[s], & s \in [0, N) \and s < S \\
1, & s \in [0, N) \and s \geqslant S
\end{cases}, s \in [0, N) \\
\forall \{i \mid i \in [0, N)\}: \text{s_arr}[i] \ne 0 \\
\text{b_range}(i) = \begin{cases}
-1, & \text{s_arr}[i] < 0 \\
0, & \text{s_arr}[i] \geqslant 0
\end{cases} \\
\text{e_range}(i) = \begin{cases}
n_i - 1, & \text{s_arr}[i] < 0 \\
n_i, & \text{s_arr}[i] \geqslant 0
\end{cases} \\
\text{b_vec}[b] = 
clip(\text{b_arr}[b], \text{a_min}=\text{b_range}(b), \text{a_max}=\text{e_range}(b)-1), b \in [0, N) \\
\text{e_vec}[e] = 
clip(\text{e_arr}[e], \text{a_min}=\text{b_range}(e), \text{a_max}=\text{e_range}(e)-1), e \in [0, N) \\
\forall \{i \mid i \in [0, N) \}: 
\begin{cases}
\text{b_vec}[i] < \text{e_vec}[i], & \text{s_arr}[i] > 0 \\
\text{e_vec}[i] < \text{b_vec}[i], & \text{s_arr}[i] < 0
\end{cases} \\
\forall (d_0, d_1, \cdots, d_{N-1}), 
\text{where } d_j \in [0, 
\left\lceil{\text{e_vec}[j] - \text{b_vec}[j] \over \text{s_arr}[j]}\right\rceil) 
\and j \in [0, N): \\
Y[d_0, d_1, \cdots, d_{N-1}] = \\
X[\text{b_vec}[0] + \text{s_arr}[0] * d_0,
\text{b_vec}[1] + \text{s_arr}[1] * d_1,
\cdots, \text{b_vec}[N-1] + \text{s_arr}[N-1] * d_{N-1}]]
$$


#### take

*Math Formalization*

Suppose Input `X`, `indices`, Output `Y`, attributes `axis`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, `indices`'s shape is M dimension, exactly $(m_0, m_1, \cdots, m_{M- 1})$, and `axis` is `Optional<int>` .

1. Case axis is `None` :

$$
T = flatten(X) \\
\forall (d_0, d_1, \cdots, d_{M-1}), \text{where } d_j \in [0, m_j) \and j \in [0, M) :\\
Y[d_0, d_1, \cdots, d_{M-1}] = T[clip(\text{xidx}, \text{a_min}=0, \text{a_max}=|T|-1],\\
\text{where } \text{xidx} = \text{indices}[d_0, d_1, \cdots, d_{M-1}]
$$

2. Case axis is `int`:

$$
\text{axis} \in [-N, N) \\
\text{real_axis} = \begin{cases}
\text{axis}, & \text{axis} \geqslant 0 \\
\text{axis} + N, & \text{axis} < 0
\end{cases} \\
\forall (d_0, d_1, \cdots, d_{M+N-1}), \text{where } d_j \in \begin{cases} 
[0, n_j), & j < \text{axis} \\
[0, m_{j-\text{axis}}), & j \in [\text{axis}, \text{axis}+M) \\
[0, n_{j-M+1}), & j \in [\text{axis} + M, M+N)
\end{cases} \and j \in [0, M+N) : \\
Y[d_0, d_1, \cdots, d_{M+N-1}] = X[d_0, \cdots, d_{\text{axis}-1}, \text{xdix}, d_{\text{axis}+M}, \cdots, d_{M+N-1}], \\
\text{where } \text{xidx}{} = clip(\text{indices}[d_{\text{axis}}, d_{\text{axis}+1}, \cdots, d_{\text{axis}+M-1}], \text{a_min}=0, \text{a_max}=n_{\text{axis}}-1)
$$



#### cvm_lut

*Math Formalization*

Suppose Input `indices`,`X`, Output `Y`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, `indices`'s shape is M dimension, exactly $(m_0, m_1, \cdots, m_{M- 1})$ .
$$
Y=take(X, \text{indices}, \text{axis}=\text{None})
$$


#### slice_like

*Math Formalization*

Suppose Input `X`, `shape_like`, Output `Y`, attributes `axes`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, `shape_like`'s shape is M dimension, exactly $(m_0, m_1, \cdots, m_{M- 1 })$, and `axes` is `TShape`, `axes`'s shape is K dimension.
$$
\forall axis \in \text{axes}: axis \in [-N, M)\\
\text{real_axes} = \begin{cases} 
\{j \mid j \in axes \and j \geqslant 0\} \bigcup
\{j + N \mid j \in axes \and j < 0\}, & K > 0\\
\{0, 1, \cdots, M-1\}, & K = 0
\end{cases} \\
\forall j \in \text{real_axes}: m_j \leqslant n_j\\
Y[d_0, d_1, \cdots, d_{N-1}] = X[d_0, d_1, \cdots, d_{N-1}], \\
\text{where } j \in [0, N) \and d_j \in \begin{cases}
[0, m_j), & j \in \text{real_axes} \\
[0, n_j), & j \notin \text{real_axes}
\end{cases}
$$


### NMS Operator

#### get_valid_count

*Math Formalization*

Suppose Input `X`, Output `valid_count`, `Y`, attributes `score_threshold`, where `X`'s shape is $(B, N, K), K \geqslant 2$,  `score_threshold` is `int`.
$$
\text{valid_count}[b] = card\{ q \mid q \in [0, N) \and
X[b, q, 1] > \text{score_threshold} \}, \\
\quad \forall b \in [0, B)
$$

$$
Y[b, \text{idx}, k] = X[b, n, k], \\
\quad \forall b \in [0, B) \and n \in [0, N) \and 
k \in [0, K) \and X[b, n, 1] > \text{score_threshold}, \\
\quad \text{where idx = }
card \{q \mid q \in [0, n) \and 
X[b, q, 1] > \text{score_threshold} \}
$$

$$
Y[b,n, k] = -1, \forall b \in [0, B) \and 
n \in [\text{valid_count}[b], N) \and k \in [0, K)
$$



#### non_max_suppression

*Math Formalization*

Suppose Input `X`, `valid_count`, Output `Y`, attributes `iou_threshold`, `max_output_size`, `force_suppress`, `top_k`, where `X`'s shape is $(B, N, K), K = 6$,  `iou_threshold` is `int`, the value is in range $[0, +\infty)$, `max_output_size` is `int`, `force_suppress` is `boolean`, `top_k` is `int`.
$$
Y[b, idx, k] = T[b, n, k], \\
\forall b \in [0, B) \and n \in [0, min(N, \text{top_k}) \and 
k \in [0, K) \and idx \in [0, \text{max_output_size}) \and \\
iou(n, p) <= \text{out_threshold}, \forall p \in [0, n), \\
\text{where } T = \text{reversed sort X over axis 1 by index at axis 2 foreach axis 0, } \\
\text{ which index priority is [1, 0, 2, 3, 4, 5] and } \\
iou(p, q) = \text{intersaction over union [T[b, p, 2:3], T[b, p, 4:5]], [T[b, q, 2:3], T[b, q, 4:5]] and } \\
\text{idx} = card\{ p \mid p \in [0, n) \and
iou(p, q) \leqslant \text{iou_threshold}, \forall q \in [0, p) \}
$$

$$
Y[b, n, k] = -1, \\
\forall b \in [0, B) \and n \in [idx, N) \and k \in [0, K), \\
\text{where idx} = card\{ p \mid p \in [0, N) \and
iou(p, q) \leqslant \text{iou_threshold}, \forall q \in [0, p) \}
$$