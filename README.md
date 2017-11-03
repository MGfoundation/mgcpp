- [mgcpp](#org8251436)
  - [Introduction](#org9dd78d2)
  - [Example](#orga8a25a1)
  - [Build](#org0bb6aa9)
  - [Dependencies](#orgd700710)
  - [TODO](#org6fefac1)
  - [Contact](#orgb25347e)
  - [License](#org9d65126)


<a id="org8251436"></a>

# mgcpp [![Build Status](https://travis-ci.org/Red-Portal/mgcpp.svg?branch=master)](https://travis-ci.org/Red-Portal/mgcpp)
C++ Multi GPU Math Library Based on CUDA


<a id="org9dd78d2"></a>

## Introduction

mgcpp is a GPGPU math library. Using various CUDA libraries as backends, <br />
it provides an optimized, standard C++ interface, abstracted C++ math library. <br />

This library is heavily under development and in pre-alpha state.<br />
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

The above code invokes cuBLAS's gemm function. All operations are lazely computed and graph optimized using C++ expression templates.


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
However if you to specify the code, 

``` shell
-DCUDA_ARCH=<arch>
```

In order to use different C++ compilers for compiling cuda code and native C++ code, <br />
provide the argument below.

``` shell
-DCUDA_HOST_COMPILER=<full path to compiler>
```

You must provide the __FULL PATH__ to the cuda host compiler in order to work.
Different cuda versions have different C++ compiler constraints. <br />
For example ```cuda 8.0``` only support gcc up to 5.3.

So, for an example case that you want to use cuda 8.0 for cuda code, gcc-7 for native C++ code,

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -DCUDA_HOST_COMPILER=/usr/bin/g++-6 -DCMAKE_CXX_COMPILER=g++7 -G "<Generator>"
make -j4
make install
```

it should look like this.


<a id="orgd700710"></a>

## Dependencies

-   cmake
-   gcc 6, clang 3.9, Visual Studio 14.0 (2015) or later
-   [outcome](https://github.com/ned14/outcome)
-   cuda 8.0 or later
-   [MAGMA](https://github.com/kjbartel/magma) (optional)
-   gtest (optional)


<a id="org6fefac1"></a>

## TODO 

-   [X] Finish thread context class for managing threads.
-   [ ] Support msvc build.
-   [ ] Add BLAS lv1 operations.
-   [ ] Add BLAS lv2 operations.
-   [ ] Add BLAS lv3 operations.
-   [ ] Add exprssions for BLAS lv1 operations.
-   [ ] Add exprssions for BLAS lv2 operations.
-   [ ] Add exprssions for BLAS lv3 operations.
-   [ ] Add dense vector type
-   [ ] Add sparse matrix type
-   [ ] Add sparse vector type
-   [ ] Add batch type
-   [ ] Add high level linear algebra operations.
-   [ ] Add benchmark
-   [ ] Add FFT
-   [ ] Add convolution


<a id="orgb25347e"></a>

## Contact

Red-Portal
-   msca8h@naver.com
-   msca8h@sogang.ac.kr


<a id="org9d65126"></a>

## License

Copyright RedPortal 2017.

Distributed under the Boost Software License, Version 1.0. <br />
(See accompanying file LICENSE or copy at <http://www.boost.org/LICENSE_1_0.txt>)
