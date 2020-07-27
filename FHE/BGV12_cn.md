# [BGV12]方案 - (Leveled) Fully Homomorphic Encryption without Bootstrapping

## 安全基础

BV11方案基于LWE（Learning With Error）困难问题，计算都为整数有限域 $\mathbb{Z}_q$。

BGV12方案引入了RLWE（Ring Learning With Error）困难问题，在同样安全性下运算性能更快。

标量从整数变为多项式环上的元素，$R_q=\mathbb{Z}_q[x]/(x^d+1)$，表示环上的多项式都对 $x^d+1$ 取模（$d$ 是2的幂），并且每个系数都对 $q$ 取模。

BGV12结合LWE和RLWE提出GLWE，其实就是允许 $d = 1$ 成立，此时退化为LWE的情况。

## 思路简介

前面介绍的[BV11方案](BV11_cn.md)中，在完成 $L$ 层运算之后运用了模数-维数压缩技术，将密文的维数和模数从 $(n, q)$ 压缩到 $(k, p)$，使得BV11方案能正确完成解密的计算。

Brakerski等人发现模数压缩和维数压缩可以分开处理，并且模数压缩还同时能降低噪音的绝对值大小（并不能降低噪音的比例，但是降低噪音的绝对值大小可以降低后续计算噪音增长的速率），并且该技术可以多次使用。

简化噪音的原理在于，假设初始噪音为 $E$， 总上限为 $q = E^n$，每次乘法运算噪音变为 $E^2, E^4,...,E^{2^L}$，为了防止噪音的规模达到同一量级而影响运算，需满足 $E^{2^L} \ll q$，即 $L \ll \log n$。而每次乘法运算后采用模数压缩技术，可以将噪音规模降回 $E$，但同时模数上限降到$q/E^L$，因此 $L \ll n$，从对数级别提升到线性级别。

具体的实现方式为设计一系列满足梯度的密钥 $s_0, s_1, ..., s_L$，其中 $s_j$ 的模数长度为 $(j+1) \cdot \mu$，因此满足 $q_j / q_{j-1} = E$

因此，BGV12方案主要在BV11方案的基础上，将重线性技术和维数-模数压缩技术结合起来，每次重线性的同时进行压缩。这一方案可以有效降低噪音增长的速率（从对数级别到线性级别），在不使用Bootstrapping的情况下完成 $L$ 层电路计算，每个门电路的复杂度为 $\tilde{O}(\lambda \cdot L^3)$。当计算层数过深时，复杂度增长剧烈，因此这一方案也不能无限进行，还是需要Bootstrapping来达成FHE。



## 引理

### 引理1

令 $p$ 和 $q$ 均为奇数模数，$\textbf{c}$ 为整数向量，定义 $\textbf{c}^{\prime}$ 为满足 $\textbf{c}^\prime = \textbf{c} \mod 2$ 中最接近 $(p/q) \cdot \textbf{c}$ 的整数向量。对任意满足 $|[\lang \textbf{c}, \textbf{s} \rang]_q| < q/2 - (q/p) \cdot \ell_1(\textbf{s})$ 的向量 $\textbf{s}$，有
$$
[\lang \textbf{c}^\prime, \textbf{s} \rang]_p = [\lang \textbf{c}, \textbf{s} \rang]_q \mod 2 \ \ \textbf{and} \ \ |[\lang \textbf{c}^\prime, \textbf{s} \rang]_p| < (p/q) \cdot |[\lang \textbf{c}, \textbf{s} \rang]_q| + \ell_1(\textbf{s})
$$
其中 $\ell_1(\textbf{s})$ 表示向量 $\textbf{s}$ 的 $\ell_1$ 范式

**证明**：

存在整数 $k$，使得 $[\lang \textbf{c}, \textbf{s} \rang]_q = \lang \textbf{c}, \textbf{s} \rang - kq$。对同样的 $k$，令 $e_p = \lang \textbf{c}^\prime, \textbf{s} \rang - kp  \in \mathbb{Z}$，因为 $\textbf{c}^\prime = \textbf{c} \mod 2$， $p = q = 1\mod 2$

所以有 $e_p = \lang \textbf{c}^\prime, \textbf{s} \rang - kp \equiv \lang \textbf{c}, \textbf{s} \rang - kq = [\lang \textbf{c}, \textbf{s} \rang]_q \mod 2$

还需要证明 $e_p = [\lang \textbf{c}^\prime, \textbf{s} \rang]_p$，并且范式值足够小

有 $e_p = (p/q)[\lang \textbf{c}, \textbf{s} \rang]_q + \lang \textbf{c}^\prime - (p/q)\textbf{c}, \textbf{s} \rang$

因此，$|e_p| \leq (p/q)[\lang \textbf{c}, \textbf{s} \rang]_q + \ell_1(s) < p/2$，不等式的后者证明 $e_p = [\lang \textbf{c}^\prime, \textbf{s} \rang]_p$



### 引理2

对长度相同的向量 $\textbf{c}, \textbf{s}$，有 $\lang {\sf BitDecomp}({\bf c}, q), {\sf Powersof2}({\bf s}, q) \rang = \lang {\bf c}, {\bf s} \rang \mod q$

**证明**：
$$
\lang {\sf BitDecomp}({\bf c}, q), {\sf Powersof2}({\bf s}, q) \rang = \sum_{j=0}^{\lfloor \log q \rfloor} \lang {\bf u}_j,2^j \cdot {\bf s} \rang = \sum_{j=0}^{\lfloor \log q \rfloor} \lang 2^j \cdot {\bf u}_j,{\bf s} \rang = \lang \sum_{j=0}^{\lfloor \log q \rfloor} 2^j \cdot {\bf u}_j,{\bf s} \rang = \lang {\bf c}, {\bf s} \rang
$$

## 基于GLWE的基础加密方案 E

### 参数初始化

**输入**： $(1^\lambda, 1^\mu, b)$

**输出**：初始化参数 $params = (q, d, n, N, \chi)$

**过程**：用 $b \in \{0,1\}$ 决定采用基于LWE的方案（$d=1$）还是基于RLWE的方案（$n=1$），选择一个 $\mu$-bit 的模数 $q$ 以及其他参数。

$d = d(\lambda, \mu, b), n = n(\lambda, \mu, b), N = \lceil (2n+1) \log q \rceil, \chi = \chi(\lambda, \mu, b)$

这些参数保障该方案针对已知攻击的安全性达到 $2^\lambda$

令 $R = \mathbb{Z}[x]/(x^d + 1)$，以及 $params = (q, d, n, N, \chi)$

### 私钥生成

**输入**：$params = (q, d, n, N, \chi)$

**输出**：私钥 $sk \in R_q^{n+1}$

**过程**：抽样 $\textbf{s}^\prime \leftarrow \chi^n$，令 $sk = \textbf{s} \leftarrow (1, \textbf{s}^\prime[1], ..., \textbf{s}^\prime[n]) \in R_q^{n+1}$

### 公钥生成

**输入**：$params = (q, d, n, N, \chi), sk$

**输出**：公钥 $pk \in R_q^{N \times (n+1)}$

**过程**：生成均匀分布的矩阵 $\textbf{A}^\prime \leftarrow R_q^{N \times n}$ 以及向量 $\textbf{e} \leftarrow \chi^N$

令 $\textbf{b} \leftarrow \textbf{A}^\prime \textbf{s}^\prime + 2\textbf{e}$，$\textbf{A} \leftarrow [\textbf{b}\ |\ -\textbf{A}^\prime]$，有 $\textbf{A} \cdot \textbf{s} = \textbf{b} - \textbf{A}^\prime \textbf{s}^\prime = 2\textbf{e}$

令 $pk = \textbf{A}$

### 加密算法

**输入**：$params = (q, d, n, N, \chi), pk, m \in \{0,1\}$

**输出**：密文 $\textbf{c} \in R_q^{n+1}$

**过程**：令 $\textbf{m} \leftarrow (m, 0, ..., 0) \in R_q^{n+1}$，抽样 $\textbf{r} \leftarrow R_2^N$

密文 $\textbf{c} \leftarrow \textbf{m} + \textbf{A}^T \textbf{r} \in R_q^{n+1}$

### 解密算法

**输入**：$params = (q, d, n, N, \chi), sk, \textbf{c}$

**输出**：明文 $m$

**过程**：$m \leftarrow [[\lang \textbf{c}, \textbf{s} \rang]_q]_2$



## 密钥切换（维数压缩）

### 切换密钥生成

**输入**：切换前后的两个维数不同的密钥 $\textbf{s}_1 \in R_q^{n_1}$，$\textbf{s_2} \in R_q^{n_2}$

**输出**：切换矩阵 $\tau_{\textbf{s}_1 \rightarrow \textbf{s}_2}$

**过程**：

1. 计算 $\textbf{A} \leftarrow {\sf E.PublicKeyGen}(\textbf{s}_2, N)$，其中 $N = n_1 \cdot \lceil \log q \rceil$
2. 令 $\textbf{B} \leftarrow \textbf{A} + {\sf Powerof2}(\textbf{s}_1)$，加到矩阵 $\textbf{A}$ 的第一列，输出 $\tau_{\textbf{s}_1 \rightarrow \textbf{s}_2} = \textbf{B}$

### 密钥切换（实际是对密文的切换，保障切换后的密文能用新密钥正常解密）

**输入**：切换矩阵及切换前的密文  $\tau_{\textbf{s}_1 \rightarrow \textbf{s}_2} = \textbf{B}, \textbf{c}_1$

**输出**：切换后的密文 $\textbf{c}_2$

**过程**：计算 $\textbf{c}_2 = {\sf BitDecomp(\textbf{c}_1)}^T \cdot \textbf{B} \in R_q^{n_2}$

### 引理3 【正确性】

令 $\textbf{s}_1, \textbf{s}_2, q, n_1, n_2, \textbf{A}, \textbf{B} = \tau_{\textbf{s}_1 \rightarrow \textbf{s}_2} \leftarrow {\sf SwitchKeyGen}(\textbf{s}_1, \textbf{s}_2)$，$\textbf{A} \cdot \textbf{s}_2 = 2 \textbf{e}_2 \in R_q^N$，$\textbf{c}_1 \in R_q^{n_1}$ 以及 ${\bf c}_2 \leftarrow {\sf SwitchKey}(\tau_{{\bf s}_1 \rightarrow {\bf s}_2}, {\bf c}_1)$

有 $\lang {\bf c}_2, {\bf s}_2 \rang = 2 \lang {\sf BitDecomp}({\bf c}_1), {\bf e}_2 \rang + \lang {\bf c}_1, {\bf s}_1 \rang \mod q$

**证明**：
$$
\begin{aligned}
\lang {\bf c}_2, {\bf s}_2 \rang &= {\sf BitDecomp}({\bf c}_1)^T \cdot \textbf{B} \cdot {\bf s}_2 \\
&= {\sf BitDecomp}({\bf c}_1)^T \cdot (2 {\bf e}_2 + {\sf Powersof2}({\bf s}_1)) \\
&= 2 \lang {\sf BitDecomp}({\bf c}_1), {\bf e}_2 \rang + \lang {\sf BitDecomp}({\bf c}_1), {\sf Powersof2}({\bf s}_1)) \rang \\
&= 2 \lang {\sf BitDecomp}({\bf c}_1), {\bf e}_2 \rang + \lang {\bf c}_1, {\bf s}_1 \rang \\
\end{aligned}
$$
在此基础上，由于 ${\sf BitDecomp}({\bf c}_1)$ 和 ${\bf e}_2$ 都很小，所以新增噪音 $2 \lang {\sf BitDecomp}({\bf c}_1), {\bf e}_2 \rang$ 的数量级别较小，并且保持2的倍数这一性质，不影响解密。可以认为 $c_2$ 是 $m$ 在密钥 $s_2$ 下的密钥。

## 模数切换

### 放缩

对整数向量 ${\bf x}$ 和整数 $q > p > m$，定义 ${\bf x}^\prime \leftarrow {\sf Scale}({\bf x}, q, p, r)$ 为满足 ${\bf x}^\prime \equiv {\bf x} \mod r$ 中最接近 $(p/q) \cdot {\bf x}$ 的向量

### $\ell_1^{(R)}$ 范数

通常在实数域上的范数 $\ell_1({\bf s}) = \sum_i \| s[i] \|$ ，拓展到环上向量 ${\bf s} \in R^n$，定义 $\ell_1^{(R)}({\bf s}) = \sum_i \| s[i] \|$

### 引理4 【噪音变化】

可视为引理1的扩展版，对环上向量进行补充

令 $d$ 为环的度，$q > p > r > 0$ 满足 $q = p = 1 \mod r$。令 ${\bf c} \in R^n$ 并且 ${\bf c}^\prime \leftarrow {\sf Scale}({\bf c}, q, p, r)$。对任意 ${\bf s} \in R^n$，满足 $\| [\lang {\bf c}, {\bf s} \rang]_q \| = q/2 - (q/p) \cdot (r/2) \cdot \sqrt{d} \cdot \gamma(R) \cdot \ell_1^{(R)}({\bf s})$ ，则有
$$
[\lang {\bf c}^\prime, {\bf s} \rang]_p \equiv [\lang \textbf{c}, \textbf{s} \rang]_q \mod 2 \ \ \textbf{and} \ \ |[\lang \textbf{c}^\prime, \textbf{s} \rang]_p| < (p/q) \cdot \|[\lang \textbf{c}, \textbf{s} \rang]_q\| + (r/2) \cdot \sqrt{d} \cdot \gamma(R) \cdot \ell_1(\textbf{s})
$$


## 具体方案实现 FHE

### 参数初始化

**输入**： $(1^\lambda, 1^L, b)$

**输出**：初始化参数集合{$params_j = (q_j, d, n_j, N, \chi)$} 

**过程**：用 $b \in \{0,1\}$ 决定采用基于LWE的方案（$d=1$）还是基于RLWE的方案（$n=1$），令 $\mu = (\lambda, L, b) = \theta(\log \lambda + \log L)$ 。

循环 $j \leftarrow L ...0$，生成参数 $params_j \leftarrow {\sf E.Setup}(1^\lambda, 1^{(j+1) \cdot \mu}, b)$

构成一个梯度序列的参数，其中模数从 $q_L$($(L+1)\cdot \mu$ bits) 到 $q_0$($\mu$ bits)

### 密钥生成算法

**输入**：初始化产生的参数 $\{params_j\}$

**输出**：私钥 $sk \leftarrow {\bf s}_j$，公钥 $pk \leftarrow (\{{\bf A_j}\}, \{\tau_{{\bf s}_{j+1}^{\prime\prime} \rightarrow {\bf s}_j}\})$

**过程**：循环 $j \leftarrow L ...0$

1. 生成私钥和公钥

   计算私钥 $ {\bf s}_j \leftarrow {\sf E.SecretKeyGen}({\bf s}_j)$ 和公钥 $\textbf{A}_j \leftarrow {\sf E.PublicKeyGen}(params_j, {\bf s}_j)$

2. 计算私钥二次项

   计算 ${\bf s}_j^\prime \leftarrow {\bf s}_j \otimes {\bf s}_j \in R_{q_j}^{\tbinom{n_j+1}{2}}$

3. 平展私钥

   计算 ${\bf s}_j^{\prime\prime} \leftarrow {\sf BitDecomp}({\bf s}_j^\prime, q_j)$

4. 生成密钥切换

   计算 $\tau_{{\bf s}_{j+1}^{\prime\prime} \rightarrow {\bf s}_j} \leftarrow {\sf SwitchKeyGen}({\bf s}_{j+1}^{\prime\prime}, {\bf s}_{j})$，当 $j = L$ 时可以忽略该步骤

### 加密算法

**输入**：参数 $params$，公钥 $pk$，消息 $m \in R_2$

**输出**：密文 $\textbf{c} \in R_q^{n+1}$

**过程**：计算 ${\sf E.Enc}({\bf A}_L, m)$

对明文加密得到第一层的密钥

### 解密算法

**输入**：参数 $params$，私钥 $sk$，密文 ${\bf c}$

**输出**：明文 $m$

**过程**：假设密文由密钥 ${\bf s}_j$ 加密，计算 ${\sf E.Dec}({\bf s}_j, {\bf c})$

可以参考BV11增加层级标签

### 同态加法

**输入**：公钥 $pk$，加数密文 ${\bf c}_1, {\bf c}_2$

**输出**：和密文 ${\bf c}_{add}$

**过程**：

1. 直接求和

   ${\bf c}_3 \leftarrow {\bf c}_1 + {\bf c}_2 \mod q_j$

2. 重制密文

   ${\bf c}_{add} \leftarrow {\sf FHE.Refresh}({\bf c}_3, \tau_{{\bf s}_{j}^{\prime\prime} \rightarrow {\bf s}_{j-1}}, q_j, q_{j-1})$

### 同态乘法

**输入**：公钥 $pk$，加数密文 ${\bf c}_1, {\bf c}_2$

**输出**：乘积密文 ${\bf c}_{mult}$

**过程**：

1. 做乘法

   ${\bf c}_3 \leftarrow {\bf c}_1 \otimes {\bf c}_2 \mod q_j$，匹配密钥 ${\bf s}_j^\prime \leftarrow {\bf s}_j \otimes {\bf s}_j$。

2. 重制密文

   ${\bf c}_{add} \leftarrow {\sf FHE.Refresh}({\bf c}_3, \tau_{{\bf s}_{j}^{\prime\prime} \rightarrow {\bf s}_{j-1}}, q_j, q_{j-1})$

### Refresh

**输入**：密文 ${\bf c}$，公钥$\tau_{{\bf s}_{j}^{\prime\prime} \rightarrow {\bf s}_{j-1}}$，两层的模数$q_j$，$q_{j-1}$

**输出**：切换后的密文 ${\bf c}^\prime$

**过程**：

1. 平展：计算 ${\bf c}_1 \leftarrow {\sf Powerof2}({\bf c}, q_j)$，由引理2可知 $\lang {\bf c}_1, {\bf s}_j^{\prime\prime} \rang = \lang {\bf c}, {\bf s}_j^{\prime} \rang \mod q_j$
2. 模数切换：计算 ${\bf c}_2 \leftarrow {\sf Scale}({\bf c}_1, q_j, q_{j-1}, 2)$，该密文匹配私钥 $s_j^{\prime\prime}$，适用模数 $q_{j-1}$
3. 密钥切换：计算 ${\bf c}_3 \leftarrow {\sf SwitchKey}(\tau_{{\bf s}_{j}^{\prime\prime} \rightarrow {\bf s}_{j-1}}, {\bf c}_2, q_{j-1})$，输出密文 ${\bf c}^\prime = {\bf c}_3$

