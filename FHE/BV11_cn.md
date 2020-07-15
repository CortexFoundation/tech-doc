# [BV11]方案 - Efficient Fully Homomorphic Encryption from (Standard) LWE

## 安全基础

和此前基于格困难问题的方案不同，BV11方案基于LWE（Learning With Error）困难问题。

简单来说，该问题中，存在一个密钥向量$\textbf{s}$，和噪音$e$。选择一组随机向量$\{\textbf{a}_i\}$，计算 $\textbf{b}_i=\textbf{a}_i \cdot \textbf{s} + e$。那么在不知道密钥的情况下，难以区分 $\textbf{b}_i$ 和另一组随机值。

## 简单介绍

BV11方案基于LWE问题构造同态加密，传统加密技术的明密文一一对应，即同样的明文在同样密钥的情况下，加密得到的密文是相同的。而这一加密引入了随机采样和随机噪音，使得密文结果并不唯一，并且在解密后得到的信息带有噪音，需要去除噪音后才能得到明文。

同态加法的实现是非常平凡的，而同态乘法的结果会产生二次项，导致多项式系数翻倍的问题。为了保证同态乘法后的系数保持线性个数，BV11采用了重线性技术，即采用新的密钥对原密钥进行重制，假设整个计算中需要 $L$ 层乘法，那么需要预先计算出 $L$ 层的密钥。

此外，为了防止噪音的规模乘上系数后变得过大，BV11采用了分解技术，将系数分解到二进制形式，每位只能为0或1，这样维持了LWE问题的假设

### 加密解密

基于LWE困难问题，BV11方案构造了一种加密方式，对消息的1bit进行加密，即$m \in \{0,1\}$。

加密计算： $b=\langle\textbf{a},\textbf{s}\rangle+2e+m \in \mathbb{Z}_q$，密文 $c =(\textbf{a},b) \in \mathbb{Z}_q^n \times \mathbb{Z}_q$

如果知道密钥 $\textbf{s}$，则可以完成解密计算：$m + 2e = b - \langle \textbf{a}, \textbf{s} \rangle$，对2取模得到 $m$，$m = [b - \langle \textbf{a}, \textbf{s} \rangle]_2$

解密函数可以写为：$f_{\textbf{a},b}(\textbf{x})=b-\langle{\textbf{a},\textbf{x}}\rangle (mod\ q)=b-\sum_{i=1}^n{\textbf{a}[i]\cdot\textbf{x}[i]} \in \mathbb{Z}_q$，当代入密钥 $\textbf{s}$，即可算出 $f_{\textbf{a},b}(\textbf{s}) = m + 2e$

从这里我们可以看出，因为模数 $q$ 为奇数，为了保障 $2e+m$ 的奇偶性不变，那么噪音 $e$ 的规模不能超过 $q$。

### 同态运算

假设现有密文$c = (\textbf{a}, b)$，$c^{'} = (\textbf{a}^{'}, b^{'})$，考虑对这两个密文进行同态运算，得到满足同态性质的新密文。

**同态加法**：$c_{add} = (\textbf{a}_{add}, b_{add}) = (\textbf{a} + \textbf{a}^{'}, b + b^{'})$

那么同样的解密函数得到：$f_{\textbf{a}+\textbf{a}^{'},b+b^{'}}(\textbf{x})=b+b^{'}-\langle{\textbf{a}+\textbf{a}^{'},\textbf{x}}\rangle (mod\ q)=f_{\textbf{a},b}(\textbf{x})+f_{\textbf{a}^{'},b^{'}}(\textbf{x}) = m_1 + m_2 + 2e_1 + 2e_2$

对2取模后与 $m_1 + m_2$ 同余

**同态乘法**：$c_{mult} = (\textbf{a}_{mult}, b_{mult})$

先看解密函数的乘法结果：$f_{\textbf{a},b}(\textbf{x})\cdot f_{\textbf{a}^{'},b^{'}}(\textbf{x}) = \left(b-\sum{\textbf{a}[i]\textbf{x}[i]}\right)\cdot \left(b^{'}-\sum{\textbf{a}^{'}[i]\textbf{x}[i]}\right) = h_0+\sum{h_i\cdot \textbf{x}[i]}+\sum{h_{i,j}\cdot \textbf{x}[i]\textbf{x}[j]}$

其中$h_0, h_i, h_{i,j}$都是合并同类项后得到的参数，这里存在一个问题，由于二次项的存在，参数的个数从$n+1$增加到$n^2$左右，以及不能保持同样的线性解密函数。

为了解决系数爆炸问题，需要采用**重线性（re-linearization）**技术，通过**转换密钥**保持解密函数的系数个数为线性个数

**重线性（re-linearization）**：解决系数爆炸问题，核心在于处理二次项 $\textbf{s}[i]\textbf{s}[j]$ ，采用生成新密钥来实现

密钥 $\textbf{s}$ 的持有者可以生成另一个密钥 $t$，把 $\textbf{s}[i]\textbf{s}[j]$ 当作消息进行加密隐藏，计算 $b_{i,j} = \textbf{a}_{i,j} \cdot \textbf{t} + e_{i,j} + \textbf{s}[i]\textbf{s}[j]$

令 $\textbf{s}[i] = 1$，那么原来的乘法结果可以写为 $\sum_{i,j}{h_{i,j}\cdot (b_{i,j}-\langle{\textbf{a}_{i,j},\textbf{t}}\rangle)}$

这里又面临一个新的问题，$b_{i,j}-\langle{\textbf{a}_{i,j},\textbf{t}}\rangle \approx \textbf{s}[i]\textbf{s}[j]$，但不代表 $h_{i,j}\cdot (b_{i,j}-\langle{\textbf{a}_{i,j},\textbf{t}}\rangle) \approx h_{i,j} \cdot \textbf{s}[i]\textbf{s}[j]$，因为系数 $h_{i,j}$ 可能很大

**分解（BitDecomp）**：解决系数绝对值大的问题，需要对 $h_{i,j}$ 展开到二进制形式

$h_{i,j} = \sum_{\tau=0}^{\lfloor{log\ q}\rfloor}h_{i,j,\tau}2^{\tau}$（因为所有系数都在有限域 $Z_q$ 上，$h_{i,j}$ 最多有 $\lfloor{log\ q}\rfloor$ 位）

乘法结果变为$\sum_{i,j,\tau}{h_{i,j,\tau}2^{\tau} \textbf{s}[i]\textbf{s}[j]}$

因为这一改变，对新密钥 $\textbf{t}$ 的使用也需要一点儿变化，令 $b_{i,j,\tau} = \langle \textbf{a}_{i,j,\tau}, \textbf{t}\rangle + e_{i,j,\tau} + 2^{\tau}\textbf{s}[i]\textbf{s}[j]$，有 $2^{\tau}\textbf{s}[i]\textbf{s}[j] \approx b_{i,j,\tau} - \langle \textbf{a}_{i,j,\tau}, \textbf{t} \rangle$

乘法结果变为$\sum_{i,j,\tau}{h_{i,j,\tau} \cdot (b_{i,j,\tau} - \langle \textbf{a}_{i,j,\tau}, \textbf{t} \rangle)}$

**密钥链（key chain）**：新设计单个密钥 $\textbf{t}$ 能完成一次乘法，为了完成 $L$ 层乘法，可以用同样的方法构造 $L$ 个依次构造的密钥，称之为密钥链。

### Bootstrapping及模数切换

进行了一系列同态运算后，密文中的噪音会逐步累积到临界值，为了保障进一步运算后的结果能够还原到正确的明文，需要采用Bootstrapping技术，简单来说，该技术首先对噪音较大的密文进行解密，然后重新用初始噪音进行加密，保证噪音的规模恢复到较小的状态，这样又可以用该密文进一步计算了。但是LWE的解密的复杂度过大，经过解密计算后的明文已经可能出现错误，为了解决这一问题，BV11提出了模数切换技术，这一技术可以保障Bootstrapping的正确执行，进而保障整个加密方案可以无限步骤地计算下去，为全同态加密方案。

**Bootstrapping**：该技术本身并不复杂，通过对噪音较大的密文进行解密，然后重新用初始噪音进行加密的方法来控制噪音的规模，保障后续计算的正确性。假设噪音规模为 $E$，那么经过一次乘法运算后噪音规模为 $E^2$， $L$ 层乘法之后，噪音规模变为 $E^{2^L}$。而前面我们讲到噪音不能超过模数 $q$，而LWE电路解密的深度是$max(n, \log q)$，不能直接解密。

**维数-模数简化（Dimension-Modulus reduction）**：如上文所说，解密电路深度是由维数 $n$ 和模数 $q$ 决定的，因此如果能减小这两个数值，即可得到可计算的解密电路。我们希望将$(n, \log q)$减小到$(k, \log p)$。其中，对维数的简化非常直接，

