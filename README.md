- [mgcpp](#org8251436)
  - [Introduction](#org9dd78d2)
  - [Example](#orga8a25a1)
  - [Build](#org0bb6aa9)
  - [Dependencies](#orgd700710)
  - [TODO](#org6fefac1)
  - [Contact](#orgb25347e)
  - [License](#org9d65126)


<a id="org8251436"></a>

# mgcpp [![Build Status](https://travis-ci.org/MGfoundation/mgcpp.svg?branch=master)](https://travis-ci.org/MGfoundation/mgcpp)
<a id="org9dd78d2"></a>

## Introduction

mgcpp is a CUDA based C++ linear algebra library. <br />
It provides a standard C++ interface without any CUDA specific syntax. <br />


### Disclaimer 

This library is heavily under development and in pre-alpha state.<br />
msvc support is not properly tested and guarenteed to work.
For contribution, please refer to TODO or contact me.


<a id="orga8a25a1"></a>

## Example

```C++
#include <mgcpp/mgcpp.hpp>

#define GPU 0

mgcpp::device_matrix<float, GPU> A(4, 3, 2);
mgcpp::device_matrix<float, GPU> B(3, 2, 2);

auto C = mgcpp::eval(A * B);

```

The above code invokes cuBLAS's gemm function.
All mgcpp expressions are lazily computed and optimized using C++ expression template method.


<a id="org0bb6aa9"></a>

## Build

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -G "<Generator>"
make -j4
make install
```

for building without MAGMA or mgcpp native cuda kernels,

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -DUSE_MAGMA=OFF -DBUILD_CUSTOM_KERNELS=OFF -G "<Generator>"
make -j4
make install
```

The library probes your system cuda ```gpu-architecture``` and ```gpu-code``` code automatically.
However if you want to specify the computability code, <br />
add the argument below to the command line. 

``` shell
-DCUDA_ARCH=<arch>
```

In order to use a different C++ compiler for cuda code and native C++ code, <br />
provide the argument below.

``` shell
-DCUDA_HOST_COMPILER=<full path to compiler>
```

You must provide the __FULL PATH__ to the cuda host compiler in order to work.
Different cuda versions have different C++ compiler constraints. <br />
For example ```cuda 8.0``` only support gcc up to 5.3.

So, for an example case that you want to use cuda 9.0 for cuda code, gcc-6 for native C++ code,

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -DCUDA_HOST_COMPILER=/usr/bin/g++-6 -DCMAKE_CXX_COMPILER=g++7 -G "<Generator>"
make -j4
make install
```


<a id="orgd700710"></a>

## Dependencies

-   cmake 3.8 or later
-   gcc 6, clang 3.9, Visual Studio 14.0 (2015) or later
-   [boost-outcome](https://github.com/ned14/boost-outcome)
-   cuda 8.0 or later
-   [MAGMA](https://github.com/kjbartel/magma) (optional)
-   gtest (optional)
-   boost


<a id="org6fefac1"></a>

## Planned Features

- Fully tested msvc support.
- Epression template optimization for all operations.
- Tensor type and tensor operations.
- sparse matrix, sparse vector, sparse tensor types.
- Batch matrix type and batch matrix operations.
- cuSPARSE support.
- Full cuFFT support.
- Convolution operation.
- half precision type support.
- Full compatibility with [uBLAS](http://www.boost.org/doc/libs/1_59_0/libs/numeric/ublas/doc/), [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), [blaze](https://bitbucket.org/blaze-lib/blaze)


<a id="orgb25347e"></a>

## Contact

Red-Portal
-   msca8h@naver.com
-   msca8h@sogang.ac.kr

mujjingun
-   mujjingun@gmail.com
-   mujjingun@sogang.ac.kr


<a id="org9d65126"></a>

## License

Copyright RedPortal, mujjingun 2017 - 2018.

Distributed under the Boost Software License, Version 1.0. <br />
(See accompanying file LICENSE or copy at <http://www.boost.org/LICENSE_1_0.txt>)
