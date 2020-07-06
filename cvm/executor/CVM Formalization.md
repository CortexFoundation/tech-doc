# CVM Formalization

### Predefined Structure

#### TShape

struct `TShape`, a array of `uint32_t`, which stands for data shape.

#### Optional\<T\>

the value can be `None`, or type of T.

#### DLTensor

Data format，a wrapper of data content pointer.

*Attributes*

- `dtype` must be INT8 or INT32.

- `shape` is `TShape` type, storing data shape, for example (1, 2, 3).

- `precision` is the max bit of each element take up, -1 stands for non-defined; a normal values' is between range (0, 32]; others is non-available, which means an error occurs, either logic error or runtime error.

  *Notes*: we define a `p` precision data is between range $[-\alpha, \alpha], \alpha=2^{p-1}-1$ for all elements, for example: data that precision is 8 means all value is larger than -128 and less than 128, this is strong constraint.

*Public Interface*

- `ndim` is the dimensions of data, for example 3.

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

算子输入输出的数据格式为`DLTensor`，`precision`属性在(0, 32]范围内。

##### Description Format

所有的数学描述采用规范：
$$
Y[y_\text{indices}] = X[x_\text{indices}], \\
\forall \text{given range}, \\
\text{where } \text{condition}_1 \text{ and } \text{condition}_2 \text{ and } 
\cdots \text{ condition}_n
$$
描述：对于所有给定取值范围，在指定坐标下，都有$Y=X$，其中定义的变量或条件用$\text{and}$相连接。

### Reduce Operator

A reduce operator performs the reduction function to input data based on the parameters, and the process logic over all the type-based operators is consistent. We abstract the formalization here and introduce the details as below:

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axes`, `keepdims`, `exclude`, where `X` has is N dimensions, exactly $(n_0, n_1, \cdots, n_{N-1})$, elements in `axes` should be different from each other and within range $[-N, N)$, indicating on which axes the reduction is done. It is of `TShape` type and will be treated as a set with M values, formally $M=card\{i \mid i \in \text{axes}\}, M \in [0,N]$, `keepdims` is a `boolean` indicating if dimension kept, `exclude` is a `boolean` giving users ability to inverse select. $R$, with size $r$, is the set of real axes where we do the reduction.
$$
T = \left\{x \mid i \in \text{axes} \and 
x = \begin{cases}
i, & \text{if } i\geqslant 0 \\
i + N, & \text{otherwise}
\end{cases} \right\}, \\
\text{where } card\{T\} = M \text{ and }
j \in [0, N), \forall j \in \text{T}
$$

$$
\text{let }U =\{0, 1,..., N-1\}\\
R = \begin{cases}
U -T , & \text{if exclude is true} \\
T, & \text{otherwise}
\end{cases}, \\
r = card\{R\}
$$



1. Case `exclude` = `true` and $M=N$: nothing to be reduced.

$$
Y = X
$$

2. Case `exclude` = `false` and $M = 0$: `axes` is not assigned so the the whole input tensor is reduced by default.

$$
Y[\underbrace{0, 0, \cdots, 0}_{K}] = \begin{cases} 
\sum_{x \in X} x, & \text{if op is sum} \\[1ex]
max\{x \mid x \in X \}, & \text{if op is max}
\end{cases}, \\
\text{where } K = \begin{cases}
1, & \text{if keepdims is false} \\
N, & \text{otherwise}
\end{cases}
$$

3. Case `keepdims` is false, $R$ dimensions of input $X$ will be reduced and the result $Y$ will have only 

$$
Y[d_{I(0)}, d_{I(1)}, \cdots, d_{I(N-r-1)}] = \\
\begin{cases}
\sum_{d_{J(0)}=0}^{n_{J(0)}} \cdots \sum_{d_{J(R-1)}=0}^{n_{J(R-1)}}
X[d_0, d_1, \cdots, d_{N-1}], & \text{if op is sum} \\[1ex]
\max \{ X[d_0, d_1, \cdots, d_{N-1}] \mid d_{J(0)} \in [0, n_{J(0)}) \and \cdots \and
d_{J(R-1)} \in [0, n_{J(R-1)}) \}, & \text{if op is max}
\end{cases}, \\
\text{where } 0 \leq d_i < n_i , \forall i\in[0, N),\text{ and }\\
I: \{ 0, 1,...,N-r-1 \} \to U-R,
\text{ s.t. } I(i) < I(j), \forall 0 \leqslant i < j < N-r \text{ and } \\
J : \{0, 1,...,r-1 \} \to R,
\text{ s.t. } J(i) < J(j), \forall 0 \leqslant i < j < R
$$

4. Otherwise

$$
Y[d_0, d_1, \cdots, d_{N-1}] = M[d_{I(0)}, d_{I(1)}, \cdots, d_{I(N-r-1)}], \\

\text{where }  0 \leq d_i < n_i , \forall i \in U-R, \text{ and } d_i=0, \forall i\in R\\
I: \{0, 1,..., N-r-1\} \to U-R,
\text{ s.t. } I(i) < I(j), \forall 0 \leqslant i < j < N-r \text{ and } \\
J : \{0, 1,..., r-1\} \to R,
\text{ s.t. } J(i) < J(j), \forall 0 \leqslant i < j < R \text{ and } \\
M = \text{reduce_op}(X, \text{axes=axes, keepdims=false, exclude=exclude})
$$

*Example*

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

#### sum

Test Parameter (sum):

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 34, 67\}, l=\{1, 58\}, r=\{1, 64\}$

​	axis: $(1, )$

#### max

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 34, 67\}, l=\{1, 58\}, r=\{1, 64\}$



### Broadcast Operator

Broadcast operator perform the broadcast function to input data, and the process logic over all the type-based operators is consistent. We abstract the formalization here and introduce the details as below:

*Math Formalization*

Suppose Input `A`, `B`, Output `Y` and broadcast function `BROADCAST_OP`. `A`'s shape is $M$ dimension, exactly $(m_0, m_1, \cdots, m_{M-1})$, `B`'s shape is $N$ dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$. 

1. Extends `A`'s shape and  `B`'s shape into $K = max(M, N)$ dimension by prefixing their shapes with $1$, denoted by $SA$ and  $SB$, respectively, where $$SA_i = \begin{cases}
   m_{i-K+M}, & i \geqslant K - M \\
   1, & i < K - M
   \end{cases} \text{ and } 
   SB_i = \begin{cases}
   n_{i-K+N}, & i \geqslant K-N \\
   1, & i < K - N
   \end{cases}$$ 

2. For $\forall i \in [0, K)$, assert $SA_i=SB_i$ or $SA_i=1$ or $SB_i=1$.  `Y`'s shape is $K$ dimension, exactly $(k_0, k_1, \cdots k_{K-1}), k_i = \max( SA_i, SB_i )$. 
3. For $\forall i \in [0, K)$, $Y[d_0, d_1, \cdots, d_{K-1}] = 
   \text{BROADCAST_OP}(A[a_0, a_1, \cdots, a_{K-1}], B[b_0, b_1, \cdots, b_{K-1}])$, where $d_{i} \in [0, k_{i}), a_i = \min(d_{i}, SA_i-1)$ and $b_i = \min(d_{i}, SB_i-1)$.

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

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	Y.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



#### broadcast_sub

set `BROADCAST_OP` to `sub`.

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$
	
	Y.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



#### broadcast_mul

set `BROADCAST_OP` to `multiply`.

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$
	
	Y.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



#### broadcast_div

set `BROADCAST_OP` to `divide`.

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	Y.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



#### broadcast_max

set `BROADCAST_OP` to `max`.

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	Y.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



#### 

### NN Operator

#### Convolution

We only supported 2-D convolution operator.

*Math Formalization*

Suppose Input `X`, `W`, `B`, and output `Y`, attributes `padding`, `stride`, `dilation`, `groups`, where `X`'s shape is $(N, C, H, W)$, `W`' shape is $(OC, IC, KH, KW)$, and $IC = C \div \text{groups}$, `B` is `Optional<DLTensor>`, if `B` is not None, it's shape is $(\text{OC},)$. `padding` is  2-D `TShape`, exactly $(PH, PW), PH,PW \in [min\_attr, max\_attr)$, `stride` is 2-D `TShape`, exactly $(SH, SW) \in [1, max\_attr)$, `dilation` is 2-D `TShape`, exactly $(DH, DW) \in [1, max\_attr)$, `grous` is `int`, the value is in $\{1, C\}$.

1. Case `groups` = 1

$$
Y[n,i,p,q]=\sum_{j=0}^{C} 
\text{kernel}(n, j, p, q, n, i)
+ \begin{cases}
0, & \text{if B is None}\\
B[i], & \text{otherwise}
\end{cases}, \\
\forall n \in [0, N) \and i \in [0, OC) \and \\
p \in \left[0, \left\lfloor{H+2 \cdot \text{PH}-\text{DH} \cdot (\text{KH}-1)-1\over\text{SH}}\right\rfloor+1 \right) \and \\
q \in \left[0, \left\lfloor{W+2 \cdot \text{PW}-\text{DW} \cdot (\text{KW}-1)-1 \over \text{SW}}\right\rfloor+1 \right),\\
$$
where $\text{kernel}$ function is 
$$
\text{kernel}(n, j, p, q, o, i) = \sum_{k_i=0}^{\text{KH}} \sum_{k_j = 0}^{\text{KW}} \text{pad}(p'+k_i*\text{DH},q'+k_j*\text{DW}) \cdot W[o, i, k_i, k_j], \\
\text{where } p' = p \cdot \text{SH} -\text{PH} \text{ and }
q' = q \cdot \text{SW}-\text{PW} \text{ and } \\
\text{pad}(p, q) = \begin{cases} 
X[n, j, p, q], & \text{ if } p \in [0, H) \and q \in [0, W) \\
0, & \text{otherwise}
\end{cases}
$$
Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/ops.cc#L475

2. Case `groups ` = $C$

This case is named *Depth-Wise Convolution*.
$$
IC = 1 \text{ and }OC = C
$$

$$
Y[n,i,p,q]= \text{kernel}(n,i, p, q, i,0) + \begin{cases}
0, & \text{if B is None}\\
B[i], & \text{otherwise}
\end{cases}, \\
\forall n \in [0, N) \and i \in [0, OC) \and\\
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

Test Parameter:

​	X.shape: $(i, j), i = \{1, 14, 27\}, j = \{1, 12, 23\}$

​	W.shape: $(k, j),  k = \{1, 18\}, j = \{1, 12, 23\}$

​	units: $k$

​	use_bias:false

#### Relu

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	

#### MaxPooling

*Math Formalization*

Suppose Input `X`, Output `Y` and attributes `pool_size`,  `	padding`, `strides`, `ceil_mode`, where `X`'s shape is $(N, C, H, W)$, `pool_size` is 2-D `TShape`, exactly $(PSH, PSW)$, `padding` is 2-D `TShape`, exactly $(PH, PW) \in [min\_attr, max\_attr)$, if `padding` is  1-D, which means $PH = PW$, `strides` is 2-D `TShape`, exactly $(SH, SW)$, `ceil_mode` is `boolean`.
$$
PSH \in [0, H + 2PH + 1), \\
PSW \in [0, W + 2PW + 1), \\
PSH > PH \and PSW > PW
$$

$$
Y[n,i,p,q] = \max\{\text{pad}(n, i, p', q') \\
\mid p' \in [p \cdot \text{SH} -\text{PH}, p \cdot \text{SH} -\text{PH}+\text{PSH}), 
q' \in [q \cdot \text{SW}-\text{PW}, q \cdot \text{SW}-\text{PW}+\text{PSW})\}, \\
\forall n \in [0, N) \and i \in [0, C) \and \\
p \in \left[0, \text{ceil_func}\left({H+2 \cdot \text{PH}-  \text{PSH}\over\text{SH}}\right)+1 \right) \and \\
q \in \left[0, \text{ceil_func}\left({W+2 \cdot \text{PW}- \text{PSW} \over \text{SW}}\right)+1 \right), \\
\text{where } \text{ceil_func(val)} = \begin{cases}
\lceil \text{val} \rceil, & \text{if ceil_mode is true} \\
\lfloor \text{val} \rfloor, & \text{otherwise}
\end{cases} \text{ and } \\
\text{pad}(n, i, p, q) = \begin{cases} 
X[n, i, p, q], & \text{ if } p \in [0, H) \and q \in [0, W) \\
0, & \text{otherwise}
\end{cases}
$$
Reference: https://github.com/CortexFoundation/CortexTheseus/blob/76320455f0769dbf22115d82181b7ba876c5f942/infernet/src/cvm/ops/cpu/ops.cc#L692

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	pool_size: $(1, 2)$



#### Upsampling

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `scale`, where `X`'s shape is $(N, C, H, W)$, `scale` is in range $[1, max\_attr)$.
$$
Y[n, i, h, w] = X[n, i, \left\lfloor {h \over \text{scale}}\right\rfloor, \left\lfloor {w \over \text{scale}}\right\rfloor], \\
\forall n \in [0, N) \and i \in [0, C) \and 
h \in [0, H \cdot \text{scale}) \and w \in [0, W \cdot \text{scale})
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	scale: 2

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

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



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

Test Parameter:

	A.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$
	
	B.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



#### elemwise_sub

*Math Formalization*

Suppose Input `A`, `B`, Output `Y`. Where `A` and `B` have the same shape, $(n_0, n_1, \cdots, n_{N-1})$.
$$
Y = A - B
$$

Test Parameter:

	A.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$
	
	B.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



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

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	a_max: $10$

​	a_min: $-19$



#### cvm_clip

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `precision`, where `precision` is in range $[1, 33)$.
$$
Y = clip(X, \text{a_min}=-\alpha, \text{a_max}=\alpha), \\
\text{where } \alpha = 2^\text{precision-1}-1
$$

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	a_max: 10

​	a_min: -19

​	precision: 2



#### cvm_right_shift

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `precision`, `shift_bit`, where `precision` is in range $[1, 33)$, and `shift_bit` is in range $[1, 33)$.
$$
Y = clip(T, \text{a_min} = -\alpha, \text{a_max}=\alpha), \\
\text{where } T = {\left\lfloor 
\left(\left\lfloor \frac{X}{2^{\text{shift_bit} - 1}} \right\rfloor + 1 \right) 
\div 2 \right\rfloor} \text{ and } \alpha = 2 ^ {\text{precision} - 1} - 1
$$

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	precision: 2

​	shift_bit: 2

#### cvm_left_shift

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `precision`, `shift_bit`, where `precision` is in range $[1, 33)$, and  `shift_bit` is in range $[1, 33)$.
$$
Y = clip(T, \text{a_min} = -\alpha, \text{a_max}=\alpha), \\
\text{where } T = X * 2^\text{shift_bit} \text{ and } \alpha = 2 ^ {\text{precision} - 1} - 1
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$
	
	precision: 2
	
	shift_bit: 2

#### 

### Transform Operator

#### repeat

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `axis`, `repeats`. Where `X`'s shape is N dimension, exactly  $(n_0, n_1, \cdots, n_{\text{axis}}, \cdots, n_{N-1})$, `Y`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{\text{axis}} \cdot repeats, \cdots, n_{N-1})$. Obviously, axis is in range $[0, N)$, repeats is in range $[1, +\infty)$.
$$
Y[d_0, d_1, \cdots, d_\text{axis}, \cdots, d_{N-1}] = 
X[d_0, d_1, \cdots, \left\lfloor{d_\text{axis} \over \text{repeats}}\right\rfloor, \cdots, d_{N-1}], \\
\forall d_0 \in [0, n_0) \and \cdots \and d_{axis-1} \in [0, n_{axis-1}) \and
d_{axis} \in [0, n_{axis} \cdot \text{repeats}) \and \\
d_{axis+1} \in [0, n_{axis+1}) \and \cdots \and d_{N-1} \in [0, n_{N-1})
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	repeats: 2

#### tile

*Math Formalization*

Suppose Input `X`, Output `Y`, attribute `reps`. Where `X`'s shape is N dimension, exactly  $(n_0, n_1, \cdots, n_{N-1})$, `reps` is M dimension, exactly $(m_0, m_1, \cdots, m_{M-1})$.
$$
r \in [1, max\_attr), \forall r \in \text{reps}
$$

$$
Y[k_0, \cdots, k_{K-N-1}, k_{K-N}, k_{K-N+1}, \cdots, k_{K-1}] = \\
X[k_{K-N+0} \text{ mod } n_0, k_{K-N+1} \text{ mod } n_1, \cdots, k_{K-N+N-1} \text{ mod } n_{N-1}], \\
\forall k_0 \in [0, S_0) \and \cdots \and k_{K-1} \in [0, S_{K-1}), \\
\text{where } K = \max\{M, N\} \text{ and } S_i = SX_i \cdot SR_i \text{ and } \\
SX_p = \begin{cases}
n_{p-K+N}, & p \in [K-N, K-1) \\
1, & p \in [0, K-N)
\end{cases} \text{ and } \\
SR_q = \begin{cases}
m_{q-K+M}, & q \in [K-M, K-1) \\
1, & q \in [0, K-M)
\end{cases}
$$

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	reps: $(2, 2, 3)$

#### flatten

*Math Formalization*

Suppose Input `X`, Output `Y`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$.
$$
Y[\text{flatten_index}(d_0, d_1, \cdots, d_{N-1}, n_0, n_1, \cdots, n_{N-1})]  =  \\
X[d_0, d_1, \cdots, d_{N-1}], \\
\forall d_0 \in [0, n_0) \and d_1 \in [0, n_1) \and \cdots \and
d_{N-1} \in [0, n_{N-1})
$$
where $\text{flatten_index}$ is 
$$
\text{flatten_index}(d_0, d_1, \cdots, d_{N-1}, n_0, n_1, \cdots, n_{N-1}) = \\
d_0 \cdot \prod_{i = 1}^{N-1} n_i + 
d_1 \cdot \prod_{i = 2}^{N-1} n_i + 
\cdots + d_{N-2} * n_{N-1} + d_{N-1}
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	

#### concatenate

*Math Formalization*

Suppose `M` Inputs $I^0, I^1, \cdots, I^{M-1}$, Output `Y`, attribute `axis`. Where all inputs' shape is N dimension, exactly $I^i$'s shape is $(n^i_0, n^i_1, \cdots, n^i_{N-1})$, and `axis` is in range $[0, N)$.
$$
n^i_j = n^0_j, \forall i \in [1, M) \and j \in [0, N) \and j \neq \text{axis}
$$

$$
Y[d_0, d_1, \cdots, d_\text{axis-1}, \text{new_idx}, d_\text{axis+1}, \cdots, d_{N-1}] = I^i[d_0, d_1, \cdots, d_{N-1}], \\
\forall d_0 \in [0, n^i_0) \and \cdots \and d_{N-1} \in [0, n^i_{N-1})
\and i \in [0, M), \\
\text{where new_idx} = \sum_{j=0}^{i-1} n^j_\text{axis} + d_\text{axis}
$$

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}

​	Y.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	

#### expand_dims

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axis`, `num_newaxis`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$,  `axis` is in range $[-N-1, N+1)$,  and `num_newaxis` is in range $[min\_attr, max\_attr)$.
$$
Y[d_0,d_1, \cdots, d_{\text{real_axis}-1}, 
\underbrace{0, 0, \cdots, 0}_{\text{num_newaxis}}, 
d_\text{real_axis}, \cdots, d_{N-1}] = X[d_0, d_1, \cdots, d_{N-1}], \\
\forall d_0 \in [0, n_0) \and \cdots \and d_{N-1} \in [0, n_{N-1}), \\
\text{where real_axis} = 
\begin{cases}
\text{axis},& \text{axis} \geqslant 0 \\
\text{axis} + N,& \text{axis} < 0
\end{cases}
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	axis: 2

#### reshape

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `target_shape`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, and `target_shape` is M dimension, exactly $(m_0, m_1, \cdots,  m_{M-1})$ , and satisfy constraint : $m_0 * m_1 * \cdots * m_{M-1} = n_0 * n_1 * \cdots * n_{N-1}$.
$$
Y[d_0, d_1, \cdots, d_{M-1}] = T[\text{flatten_index}(d_0, d_1, \cdots, d_{M-1}, m_0, m_1, \cdots, m_{N-1})], \\
\forall d_0 \in [0, m_0) \and \cdots \and d_{N-1} \in [0, m_{N-1}), \\
\text{where } T = \text{flatten}(X)
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$
	
	shape: $(r, l, j, i), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$



​	

#### squeeze

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axes`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, and `axes` is TShape and dimension is M.
$$
\text{axis} \in [-N, N), \forall \text{axis} \in \text{axes}
$$

$$
\text{real_axes} = 
\begin{cases}
\{\text{axis} \mid \text{axis} \geqslant 0 \and \text{axis} \in \text{axes} \} \bigcup
\{\text{axis} + N \mid \text{axis} < 0 \and \text{axis} \in \text{axis}\},
& M > 0 \\
\{\text{axis} \mid n_\text{axis} = 1 \and \text{axis} \in [0, N) \}, & M = 0
\end{cases} \\
$$

$$
n_\text{axis} = 1, \forall \text{axis} \in \text{real_axis}
$$

$$
Y[d_{I(0)}, d_{I(1)}, \cdots, d_{I(N-K-1)}] = X[d_0, d_1, \cdots, d_{N-1}], \\
\forall d_0 \in [0, n_0) \and d_1 \in [0, n_1) 
\and \cdots \and d_{N-1} \in [0, n_{N-1}), \\
\text{where } K = card \; \text{real_axes} \text{ and } \\
I: \{i \mid i \in [0, N-K) \} \to 
\{i \mid i \in [0, N) \and i \notin \text{real_axes} \}, \\
\text{satisfy } I(i) < I(j), \forall 0 \leqslant i < j < N-K
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	

#### transpose

*Math Formalization*

Suppose Input `X`, Output `Y`, attributes `axes`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, and `axes` is TShape and dimension is M, where M is in $\{0, N\}$.
$$
\text{axis} \in [-N, N), \forall \text{axis} \in \text{axes}
$$

$$
Y[d_{\text{real_axes}_0}, d_{\text{real_axes}_1}, \cdots, d_{\text{real_axes}_{N-1}}] = 
X[d_0, d_1, \cdots, d_{N-1}], \\
\forall d_0 \in [0, n_0) \and \cdots \and d_{N-1} \in [0, n_{N-1}), \\
\text{where real_axes}_i = \begin{cases}
\text{axes}_i, & M = N \and \text{axes}_i \geqslant 0 \\
\text{axes}_i + N, & M = N \and \text{axes}_i < 0 \\
N-1-i, & M = 0
\end{cases} \text{ and } \\
card \; \{\text{real_axes}_i \mid i \in [0, N) \} = N
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	

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

Y[d_0, d_1, \cdots, d_{N-1}] = \\
X[\text{b_vec}[0] + \text{s_arr}[0] * d_0,
\text{b_vec}[1] + \text{s_arr}[1] * d_1,
\cdots, \text{b_vec}[N-1] + \text{s_arr}[N-1] * d_{N-1}]] \\
\forall (d_0, d_1, \cdots, d_{N-1}), 
\text{where } d_j \in [0, 
\left\lceil{\text{e_vec}[j] - \text{b_vec}[j] \over \text{s_arr}[j]}\right\rceil) 
\and j \in [0, N)
$$

Test Parameter:

	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	

#### take

*Math Formalization*

Suppose Input `X`, `indices`, Output `Y`, attributes `axis`. Where `X`'s shape is N dimension, exactly $(n_0, n_1, \cdots, n_{N-1})$, `indices`'s shape is M dimension, exactly $(m_0, m_1, \cdots, m_{M- 1})$, and `axis` is `Optional<int>` .

1. Case axis is `None` :

$$
T = flatten(X) \\
Y[d_0, d_1, \cdots, d_{M-1}] = T[clip(\text{xidx}, \text{a_min}=0, \text{a_max}=|T|-1],\\
\forall (d_0, d_1, \cdots, d_{M-1}), \text{where } d_j \in [0, m_j) \and j \in [0, M) \text{ and }\\
\text{xidx} = \text{indices}[d_0, d_1, \cdots, d_{M-1}] 
$$

2. Case axis is `int`:

$$
\text{axis} \in [-N, N) \\
\text{real_axis} = \begin{cases}
\text{axis}, & \text{axis} \geqslant 0 \\
\text{axis} + N, & \text{axis} < 0
\end{cases} \\
Y[d_0, d_1, \cdots, d_{M+N-1}] = X[d_0, \cdots, d_{\text{real_axis}-1}, \text{xdix}, d_{\text{real_axis}+M}, \cdots, d_{M+N-1}], \\
\forall (d_0, d_1, \cdots, d_{M+N-1}), \text{where } d_j \in \begin{cases} 
[0, n_j), & j < \text{real_axis} \\
[0, m_{j-\text{real_axis}}), & j \in [\text{real_axis}, \text{real_axis}+M) \\
[0, n_{j-M+1}), & j \in [\text{real_axis} + M, M+N)
\end{cases} \and j \in [0, M+N) : \\

\text{where } \text{xidx}{} = clip(\text{indices}[d_{\text{real_axis}}, d_{\text{real_axis}+1}, \cdots, d_{\text{real_axis}+M-1}], \text{a_min}=0, \text{a_max}=n_{\text{real_axis}}-1)
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

Test Parameter:

​	X.shape: $(i, j, l, r), \\ i=\{1\}, j=\{1, 14, 27, 40, 53, 66, 79, 92\}, l=\{1, 18, 35, 52, 69, 86\}, r=\{1, 24, 47, 70, 93\}$

​	axis: $(0, 1)$



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

Suppose Input `X`, `valid_count`, Output `Y`, attributes `iou_threshold`, `max_output_size`, `force_suppress`, `top_k`, where `X`'s shape is $(B, N, K), K = 6$,  `iou_threshold` is `int`, the value is in range $(0, +\infty)$,  101 stands for bounding box full-overlap specifically, and larger integer is equivalent to that. `max_output_size` is `int`, `force_suppress` is `boolean`, `top_k` is `int`. 
$$
R[b, i, k] = X[b, I(i), k], \\
\forall b \in [0, B) \and i \in [0, T) \and k \in [0, K), \\
\text{where } T = \text{max}\{
\text{min}(N, \text{valid_count}[b]), 0\} \text{ and } \\
I: \{ i \mid i \in [0, T) \} \to \{ i \mid i \in [0, T) \}, \\
\text {satisfy } X[b, I(i), 1] > X[b, I(j), 1] \or \\
(X[b, I(i), 1] = X[b, I(j), 1] \and I(i) < I(j)), 
\forall 0 \leqslant i < j < T
$$

$$
Y[b, n, k] = R[b, \text{IDX}(n), k], \\
\forall b \in [0, B) \and n \in [0, \min\{T, \text{MOS}, card\{U\}\}) \and 
k \in [0, K), \\
\text{where } \text{TK} = 
\begin{cases}
+\infty, & \text{if top_k < 0} \\[1ex]
\text{top_k}, & \text{otherwise}
\end{cases} \text{ and } \\
\text{MOS} = 
\begin{cases}
+\infty, & \text{if max_output_size < 0} \\[1ex]
\text{max_output_size}, & \text{otherwise}
\end{cases} \text{ and } \\
\text{iou}(p, q) = \begin{cases}
\text{overlap_ratio}(R[b, p, :], R[b, q, :]), &
\begin{array}{}
\text{force_suppress is true} \or R[b, p, 0] = R[b, q, 0] 
\end{array} \\[1ex]
0, & \text{otherwise}
\end{cases} \text{ and } \\
\text{U} = \{p \mid p \in [0, min\{TK, T\}) \and 
R[b,p,0] >= 0 \and \text{iou}(p, q) < \text{iou_threshold}, 
\forall q \in U \and q < p\}
 \text{ and } \\
\text{IDX}: \{i \mid i \in [0, card\{U\})\} \to U, \text{satisfy }
\text{IDX}(i) < \text{IDX}(j), \forall 0 \leqslant i < j < card\{U\}
$$

$$
Y[b, n, k] = -1, \\
\forall b \in [0, B) \and k \in [0, K) \and 
n \in [min\{T, \text{MOS},card\{U\}\}, N)
$$
