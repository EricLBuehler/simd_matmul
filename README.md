# SIMD Matmul

This is an optimization of the naive matmul algorithm which uses SIMD. 

## Asymptotic complexity analysis
It is $O(n^2)$. For $A = (m,n)$, $B = (n,p)$ and $A*B = C = (m,p)$, the exact running time for SIMD matmul including addition and multiplication operations is $mp(1+n-1) = mpn$ while the running time for Naive matmul is $mp(2n-1) = 2mpn-mp$. Therefore, the precise ratio is $$\frac{n}{2n-1}$$

However, when calculating the Big-O complexity we ignore addition operations and as such the running time is $mp$ or $O(n^2)$.

## Mathematical Formulation
For $A = (m,n)$, $B = (n,p)$, I calculate $B^T$. This results in the inputs to the algorithm, $A = (m,n)$, $B = (p,n)$. I note that the transpose algorithm is also $O(n^2)$.
The output $C' = (m,p)$ and is equivalent to $C = A*B$.

## Advantages
- Far lower theoretical Big-O
- Header-only library

## Disadvantages
- $n$ is constrained by SIMD lanes
- Requires transpose or store matrices in transposed form
- Requires conversion of matrix rows to SIMD vectors