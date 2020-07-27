# Homomorphic Encryption from Learning with Errors: Conceptually-Simpler, Asymptotically-Faster, Attribute-Based

## Basic Introduction

secret key: $\textbf{v} \in \mathbb{Z}_q^N$

plaintext: $\mu_i \in \{0,1\}$

ciphertext: $C_i \in \mathbb{Z}_q^{N \times N}$ with “small” entries (much smaller than q)

error: $e$

$C_i \cdot \textbf{v} \approx \mu_2 \cdot \textbf{v}$

$C_i \cdot \textbf{v} = \mu_2 \cdot \textbf{v} + \textbf{e}$

$(C_1 + C_2) \cdot \textbf{v} = (\mu_1 + \mu_2) \cdot \textbf{v} + \textbf{e}_1 + \textbf{e}_2$

$(C_1 \cdot C_2) \cdot \textbf{v} = C_1 \cdot (\mu_2 \cdot \textbf{v} + \textbf{e}_2) = \mu_1 \cdot \mu_2 \cdot \textbf{v} + \mu_2 \cdot \textbf{e}_1 + C_1 \cdot \textbf{e}_2$

Assumption: $\mu_2, C_1$ are small

Lattice Gadget:

1. Let $\vec{a}, \vec{b}$ be vectors of some dimension $k$ over $\mathbb{Z}_q$, $l=\lfloor log_2q \rfloor + 1$ and $N = k \cdot l$.

   $\textbf{BitDecomp}(\vec{a}) = (a_{1,0},...,a_{1,l-1},...,a_{k,0},...,a_{k,l-1})$, where $a_{i,j}$ is the j-th bit in $a_i$’s binary representation

2. For $\vec{a^{'}} = (a_{1,0},...,a_{1,l-1},...,a_{k,0},...,a_{k,l-1}) \in \mathbb{Z}_q^N$, let $\textbf{BitDecomp}^{-1}(\vec{a^{'}}) = \sum{2^j \cdot a_{1,j} + ... + \sum{2^j \cdot a_{k,j}}}$

3. For N-demensional $\vec{a^{'}}$, let $\textbf{Flatten}(\vec{a^{'}}) = \textbf{BitDecomp}(\textbf{BitDecomp}^{-1}(\vec{a^{'}}))$

4. For N-demensional $\vec{b}$, let $\textbf{Powerof2}(\vec{b}) = (b_1, 2b_1,..., 2^{l-1}b_1,...,b_k,2b_k,...2^{l-1}b_k)$

5. 



## Basic Encryption Scheme

* $\textbf{Setup}(1^\lambda,1^L)$
* $\textbf{SecretKeyGen}(params)$

