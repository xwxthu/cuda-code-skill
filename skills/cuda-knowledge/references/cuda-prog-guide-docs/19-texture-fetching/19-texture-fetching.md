# 19. Texture Fetching


Warning

This document has been replaced by a new [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide). The information in this document should be considered legacy, and this document is no longer being updated as of CUDA 13.0. Please refer to the [CUDA Programming Guide](http://docs.nvidia.com/cuda/cuda-programming-guide) for up-to-date information on CUDA.


This section gives the formula used to compute the value returned by the texture functions of [Texture Functions](#texture-functions) depending on the various attributes of the texture object (see [Texture and Surface Memory](#texture-and-surface-memory)).


The texture bound to the texture object is represented as an array _T_ of


  * _N_ texels for a one-dimensional texture,

  * _N x M_ texels for a two-dimensional texture,

  * _N x M x L_ texels for a three-dimensional texture.


It is fetched using non-normalized texture coordinates _x_ , _y_ , and _z_ , or the normalized texture coordinates _x/N_ , _y/M_ , and _z/L_ as described in [Texture Memory](#texture-memory). In this section, the coordinates are assumed to be in the valid range. [Texture Memory](#texture-memory) explained how out-of-range coordinates are remapped to the valid range based on the addressing mode.


##  19.1. Nearest-Point Sampling 

In this filtering mode, the value returned by the texture fetch is

  * _tex(x)=T[i]_ for a one-dimensional texture,

  * _tex(x,y)=T[i,j]_ for a two-dimensional texture,

  * _tex(x,y,z)=T[i,j,k]_ for a three-dimensional texture,


where _i=floor(x)_ , _j=floor(y)_ , and _k=floor(z)_.

[Figure 36](#nearest-point-sampling-nearest-point-sampling-fig) illustrates nearest-point sampling for a one-dimensional texture with _N=4_.

![_images/nearest-point-sampling-of-1-d-texture-of-4-texels.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/nearest-point-sampling-of-1-d-texture-of-4-texels.png)

Figure 36 Nearest-Point Sampling Filtering Mode

For integer textures, the value returned by the texture fetch can be optionally remapped to [0.0, 1.0] (see [Texture Memory](#texture-memory)).


##  19.2. Linear Filtering 

In this filtering mode, which is only available for floating-point textures, the value returned by the texture fetch is

  * \\(tex(x)=(1-\alpha)T[i]+{\alpha}T[i+1]\\) for a one-dimensional texture,

  * \\(tex(x)=(1-\alpha)T[i]+{\alpha}T[i+1]\\) for a one-dimensional texture,

  * \\(tex(x,y)=(1-\alpha)(1-\beta)T[i,j]+\alpha(1-\beta)T[i+1,j]+(1-\alpha){\beta}T[i,j+1]+\alpha{\beta}T[i+1,j+1]\\) for a two-dimensional texture,

  * \\(tex(x,y,z)\\) =

\\((1-\alpha)(1-\beta)(1-\gamma)T[i,j,k]+\alpha(1-\beta)(1-\gamma)T[i+1,j,k]+\\)

\\((1-\alpha)\beta(1-\gamma)T[i,j+1,k]+\alpha\beta(1-\gamma)T[i+1,j+1,k]+\\)

\\((1-\alpha)(1-\beta){\gamma}T[i,j,k+1]+\alpha(1-\beta){\gamma}T[i+1,j,k+1]+\\)

\\((1-\alpha)\beta{\gamma}T[i,j+1,k+1]+\alpha\beta{\gamma}T[i+1,j+1,k+1]\\)

for a three-dimensional texture,


where:

  * \\(i=floor(x\ B)*, \alpha=frac(x\ B)*, *x\ B\ =x-0.5,\\)

  * \\(j=floor(y\ B)*, \beta=frac(y\ B)*, *y\ B\ =y-0.5,\\)

  * \\(k=floor(z\ B)*, \gamma=frac(z\ B)*, *z\ B\ = z-0.5,\\)


\\(\alpha\\), \\(\beta\\), and \\(\gamma\\) are stored in 9-bit fixed point format with 8 bits of fractional value (so 1.0 is exactly represented).

[Figure 37](#linear-filtering-of-1-d-texture-of-4-texels) illustrates linear filtering of a one-dimensional texture with _N=4_.

![_images/linear-filtering-of-1-d-texture-of-4-texels.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/linear-filtering-of-1-d-texture-of-4-texels.png)

Figure 37 Linear Filtering Mode


##  19.3. Table Lookup 

A table lookup _TL(x)_ where _x_ spans the interval _[0,R]_ can be implemented as _TL(x)=tex((N-1)/R)x+0.5)_ in order to ensure that _TL(0)=T[0]_ and _TL(R)=T[N-1]_.

[Figure 38](#table-lookup-1-d-table-lookup-using-linear-filtering) illustrates the use of texture filtering to implement a table lookup with _R=4_ or _R=1_ from a one-dimensional texture with _N=4_.

![_images/1-d-table-lookup-using-linear-filtering.png](https://docs.nvidia.com/cuda/cuda-c-programming-guide/_images/1-d-table-lookup-using-linear-filtering.png)

Figure 38 One-Dimensional Table Lookup Using Linear Filtering
