# Efficient Fully Homomorphic Encryption from (Standard) LWE

## Security Assumption

LWE problem(Learning With Error)

## Scheme Brief Description

Plaintext:  $ m \in \{0, 1\}$

Secret key: $\textbf{s} \in \mathbb{Z}_q^n$

Random vector: $\textbf{a} \in \mathbb{Z}_q^n$

Ciphertext: $c =(\textbf{a},b) \in \mathbb{Z}_q^n \times \mathbb{Z}_q$

Encryption function: $b=\langle\textbf{a},\textbf{s}\rangle+2e+m \in \mathbb{Z}_q$

Decryption fuction: $f_{\textbf{a},b}(\textbf{x})=b-\langle{\textbf{a},\textbf{x}}\rangle (mod\ q)=b-\sum_{i=1}^n{\textbf{a}[i]\cdot\textbf{x}[i]} \in \mathbb{Z}_q$

## Homomorphic addition

$$
f_{\textbf{a}+\textbf{a}^{'},b+b^{'}}(\textbf{x})=b+b^{'}-\langle{\textbf{a}+\textbf{a}^{'},\textbf{x}}\rangle (mod\ q)=f_{\textbf{a},b}(\textbf{x})+f_{\textbf{a}^{'},b^{'}}(\textbf{x})
$$

The homomorphic addition can be computed directly.

## Homomorphic multiplication

$$
f_{\textbf{a},b}(\textbf{x})\cdot f_{\textbf{a}^{'},b^{'}}(\textbf{x}) = \left(b-\sum{\textbf{a}[i]\textbf{x}[i]}\right)\cdot \left(b^{'}-\sum{\textbf{a}^{'}[i]\textbf{x}[i]}\right)\\ = h_0+\sum{h_i\cdot \textbf{x}[i]}+\sum{h_{i,j}\cdot \textbf{x}[i]\textbf{x}[j]}
$$

The decryption algorithm has to know all the coefficients of this quadratic polynomial, which means that the size of the ciphertext just went up from $n+1$ elements to (roughly) $n^2/2$

### Re-linearization technique

Aim: Reduce the size of the ciphertext back down to $n+1$

Idea: Imagine that we publish “encryptions” of all the linear and quadratic terms in the secret key $s$, namely all the numbers $s[i]$ as well as $s[i]s[j]$, under a new secret key $t$.

New ciphertexts:

$b_{i}=\langle\textbf{a}_{i},\textbf{t}\rangle + 2e_{i} + s[i] \approx \langle\textbf{a}_{i},\textbf{t}\rangle + s[i] \in \mathbb{Z}_q$

$b_{i,j}=\langle\textbf{a}_{i,j},\textbf{t}\rangle + 2e_{i,j} + s[i]s[j] \approx \langle\textbf{a}_{i,j},\textbf{t}\rangle + s[i]s[j] \in \mathbb{Z}_q$

Now, the sum $h_0+\sum{h_i\cdot \textbf{x}[i]}+\sum{h_{i,j}\cdot \textbf{x}[i]\textbf{x}[j]}$ can be written (approximately) as 

$h_0+\sum_i{h_i\cdot (b_i - \langle{\textbf{a}_i},\textbf{t}\rangle)}+\sum_{i,j}{h_{i,j}\cdot (b_{i,j}-\langle{\textbf{a}_{i,j},\textbf{t}}\rangle)}$

which is a linear function of $t$ after simplification.

A “chain” of L secret keys (together with encryptions of quadratic terms of one secret key using the next secret key) allows us to perform up to L levels of multiplications without blowing up the ciphertext size.

#### Flatten(described in the [GSW13](GSW13.md))

Consider the binary representation of $h_{i,j}$, namely $h_{i,j} = \sum_{\tau=0}^{\lfloor{log\ q}\rfloor}h_{i,j,\tau}2^{\tau} \textbf{s}[i]\cdot \textbf{s}[j]$

For each value of $\tau$, we have a pair $(\textbf{a}_{i,j,\tau},b_{i,j,\tau})$ such that

$b_{i,j,\tau} = \langle{\textbf{a}_{i,j,\tau}, \textbf{t}}\rangle + 2e_{i,j,\tau} + 2^{\tau}\textbf{s}[i]\cdot \textbf{s}[j] \approx\langle{\textbf{a}_{i,j,\tau}, \textbf{t}}\rangle + 2^{\tau}\textbf{s}[i]\cdot \textbf{s}[j] $

then $h_{i,j}\cdot \textbf{s}[i]\textbf{s}[j] = \sum_{\tau=0}^{\lfloor{log\ q}\rfloor}2^{\tau}\textbf{s}[i]\cdot \textbf{s}[j] \approx h_{i,j,\tau}(b_{i,j,\tau} - \langle \textbf{a}_{i,j,\tau}, \textbf{t} \rangle)$

### Dimension-Modulus Reduction

Aim: Taking a ciphertext with parameters $(n,log\ q)$ as above, and convert it into a ciphertext of the same message, but with parameters $(k,log\ p)$ which are much smaller than $(n,log\ q)$.

