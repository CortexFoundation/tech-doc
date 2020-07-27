# [Bra12]方案 - Fully Homomorphic Encryption without Modulus Switching from Classical GapSVP

## 安全基础

## 思路简介

[BGV12方案](BGV12_cn.md)中提出利用模数切换来压缩噪音的规模，保证每次乘法之后噪音和上限比值的增长是线性的而非最初的双指数级，因此能进行的乘法深度变为线性。这带来一个思路，缩放虽然不能改变模数和噪音的比例，但是能改变我们看待密文的角度，从而管理噪音。

因此Bra12方案直接考虑将噪音缩放到[0,1)的区间，假设噪音的bound是 $B$，则将噪音压缩为 $B/q < 1$。这样一来，后续的所有乘法都不会导致噪音的剧烈增长，因为 $(B/q)^2 < (B/q)$ 。所以乘法后噪音的增长并不是平方级别，而是乘上一个密码参数的多项式 $p(n)$ ，经过 $L$ 层乘法，噪音从 $(B/q)$ 增长到 $(B/q) \cdot p(n)^L$ 。意味着我们的模数上限可以设置为 $q \approx B \cdot p(n)^L$，一方面降低了运算复杂度，另一方面增强了安全性。

在模数 1 的范围内进行运算这一想法在[AD97]，[Reg03]和[Reg05]中都有提及。本方案并没有采用浮点数，而是用整数域 $\mathbb{Z}_q$ 上的元素进行近似计算。

这一设计的好处有：

1. 放缩不变性，只和 $q/B$ 的比值有关，与绝对值无关，因此只要保证 $q$ 的长度符合安全，实际上可以选择 2 的幂次作为模数来加速运算。
2. 不需要模数切换的操作，噪音的增长已经控制住了



## Regev 公钥加密体系

### 私钥生成

**输入**：$1^n$

**输出**：$sk \in \mathbb{Z}_q^n$

**过程**：

采样 $\textbf{s} \stackrel{$}\leftarrow \mathbb{Z}_q^n$，输出 $sk = \textbf{s}$

对消息 $m \in \{0, 1\}$ 加密得到向量 $\textbf{c}$ 满足 $\lang \textbf{c}, \textbf{s} \rang = \lfloor \frac{q}{2} \rfloor \cdot m + e + qI$

### 公钥生成

**输入**：私钥 $\textbf{s} \in \mathbb{Z}_q^n$

**输出**：公钥 $pk = \textbf{P} \in \mathbb{Z}_q^{N \times (n+1)}$

**过程**：

令 $N \triangleq (n+1) \cdot (\log q + O(1))$，采样 $\textbf{A} \stackrel{$}\leftarrow \mathbb{Z}_q^{N \times n}$ 以及 ${\bf e} \stackrel{$}\leftarrow \chi^N$

计算 $\textbf{b} := [\textbf{A} \cdot \textbf{s} + \textbf{e}]_q$ ，并且令 $\textbf{P} := [\textbf{b} \| -\textbf{A}] \in \mathbb{Z}_q^{N \times (n+1)}$

### 加密

**输入**：消息 $m \in \{0, 1\}$，公钥 $pk$

**输出**：密文 $\textbf{c} \in \mathbb{Z}_q^{(n+1)}$

**过程**：

采样 $\textbf{r} \in \{0, 1\}^N$

计算 ${\bf c} := [{\bf P}^T \cdot r + \lfloor \frac{q}{2} \cdot {\bf m} \rfloor]_q \in \mathbb{Z}_q^{n+1}$

其中 ${\bf m} \triangleq (m, 0, ..., 0) \in \{0, 1\}^{n+1}$

### 解密

**输入**：密文 ${\bf c}$，私钥 $sk$

**输出**：明文消息 $m$

**过程**：计算
$$
m := [\lfloor 2 \cdot \frac{[\lang {\bf c}, (1, {\bf s}) \rang]_q}{q}\rceil]_2
$$


### 正确性证明

引理3.1证明加密带来的噪音在固定范围内，引理3.2证明密文包含在给定范围内的噪音能被正确解密

#### 引理3.1

令 $q,n,N,|\chi| \leq B$ 为 ${\sf Regev}$ 的参数，令 ${\bf s} \in \mathbb{Z}^n$ 为任意向量并且 $m \in \{0, 1\}$。设置 ${\bf P} \leftarrow {\sf Regev.PublicKeygen}({\bf s})$ 以及 ${\bf c} \leftarrow {\sf Regev.Enc}_{\bf P}(m)$。则存在 $e$ 满足 $|e| \leq N \cdot B$ ，使得
$$
\lang {\bf c}, (1, {\bf s}) \rang = \lfloor \frac{q}{2} \rfloor \cdot m + e \mod q
$$
证明：
$$
\begin{aligned}
\lang {\bf c}, (1, {\bf s}) \rang &= \lang {\bf P}^T \cdot {\bf r} + \biggl\lfloor \frac{q}{2} \biggl\rfloor \cdot m, (1, {\bf m}) \rang \mod q \\
&= \biggl\lfloor \frac{q}{2} \biggl\rfloor \cdot m + {\bf r}^T {\bf P} \cdot (1, {\bf s}) \mod q \\
&= \biggl\lfloor \frac{q}{2} \biggl\rfloor \cdot m + {\bf r}^T {\bf b} - {\bf r}^T {\bf As} \mod q \\
&= \biggl\lfloor \frac{q}{2} \biggl\rfloor \cdot m + \lang {\bf r, e} \rang \mod q \\
\end{aligned}
$$


因为 $|\lang {\bf r, e} \rang| \leq N \cdot B$，所以引理3.1成立

#### 引理3.2

令 ${\bf s} \in \mathbb{Z}^n$ 为某一向量，${\bf c} \in \mathbb{Z}_q^{n+1}$ 满足
$$
\lang {\bf c}, (1, {\bf s}) \rang = \lfloor \frac{q}{2} \rfloor \cdot m + e \mod q
$$
其中 $m \in \{0,1\}$ 并且 $|e| < \lfloor q / 2 \rfloor / 2$
$$
{\sf Regev.Dec_{\bf s}({\bf c})} = m
$$

## 向量拆解和密钥交换

### 向量拆解

和BV11以及BGV12中使用的技术一样，包含 ${\sf BitDecomp}$ 和 ${\sf PowersOfTwo}$ 函数

### 密钥交换

和BGV12中使用的维数切换技术一样，只实现维数的切换，不改变模数



## 缩放不变性的同态加密方案 （SI-HE, A Scale Invariant Homomorphic Encryption Scheme）

### 密钥生成

**输入**：计算深度和私钥长度 $(1^L, 1^n)$

**输出**：公钥 $pk = {\bf P}_0$，切换密钥 $evk = \{{\bf P}_{(i-1):i}\}_{i \in [L]}$，私钥 $sk = {\bf s}_L$

**过程**：

采样 $L+1$ 个向量 ${\bf s}_0,...,{\bf s}_L \leftarrow {\sf Regev.SecretKeygen}(1^n)$

计算公钥 ${\bf P}_0 \leftarrow {\sf Regev.PublicKeygen}({\bf s}_0)$

对所有 $i \in [L]$，定义
$$
\tilde{\bf s}_{i-1} := {\sf BitDecomp((1,{\bf s}_{i-1}))} \otimes {\sf BitDecomp}((1, {\bf s}_{i-1})) \in \{0,1\}^{((n+1) \lceil \log q \rceil)^2}
$$
计算
$$
{\bf P}_{(i-1):i} \leftarrow {\sf SwitchKeyGen}(\tilde{\bf s}_{i-1}, {\bf s}_i)
$$

### 加密算法

**输入**：密文 ${\bf c}$，私钥 $sk$

**输出**：明文消息 $m$

**过程**：直接调用 ${\bf c} \leftarrow {\sf Regev.Enc}_{pk}(m)$

### 同态加法

**输入**：密文 ${\bf c}_1,{\bf c}_2$，切换密钥 $evk$

**输出**：加法密文 ${\bf c}_{add}$

**过程**：

假设密文 ${\bf c}_1,{\bf c}_2$ 都对应密钥 ${\bf s}_{i-1}$，计算
$$
\tilde{\bf c}_{add} := {\sf PowersOfTwo}({\bf c}_1 + {\bf c}_2) \otimes {\sf PowersOfTwo}((1,0,...,0))
$$
输出
$$
{\bf c}_{add} \leftarrow {\sf SwitchKey}({\bf P}_{(i-1):i}, \tilde{\bf c}_{add}) \in \mathbb{Z}_q^{n+1}
$$

### 同态乘法

**输入**：密文 ${\bf c}_1,{\bf c}_2$，切换密钥 $evk$

**输出**：乘法密文 ${\bf c}_{mult}$

**过程**：

假设密文 ${\bf c}_1,{\bf c}_2$ 都对应密钥 ${\bf s}_{i-1}$，计算
$$
\tilde{\bf c}_{mult} := \biggl\lfloor \frac{2}{q} \cdot \biggl({\sf PowersOfTwo}({\bf c}_1) \otimes {\sf PowersOfTwo}({\bf c}_2) \biggl) \biggl\rceil
$$
输出
$$
{\bf c}_{mult} \leftarrow {\sf SwitchKey}({\bf P}_{(i-1):i}, \tilde{\bf c}_{mult}) \in \mathbb{Z}_q^{n+1}
$$

### 解密算法

**输入**：密文 ${\bf c}$，私钥 $sk$

**输出**：明文消息 $m$

**过程**：不失一般性，假设密文对应 $sk = S_L$，解密等价于
$$
m \leftarrow {\sf Regev.Dec}_{sk}({\bf c})
$$
