- [mgcpp](#org8251436)
  - [Introduction](#org9dd78d2)
  - [Example](#orga8a25a1)
  - [Build](#org0bb6aa9)
  - [Dependencies](#orgd700710)
  - [TODO](#org6fefac1)
  - [Contact](#orgb25347e)
  - [License](#org9d65126)


<a id="org8251436"></a>

![logo](https://github.com/MGfoundation/mgcpp/blob/master/docs/logo.png)

# mgcpp [![Build Status](https://travis-ci.org/MGfoundation/mgcpp.svg?branch=master)](https://travis-ci.org/MGfoundation/mgcpp)
<a id="org9dd78d2"></a>

## Introduction

mgcpp is a CUDA based C++ linear algebra library. <br />
It provides a standard C++ interface without any CUDA specific syntax. <br />


### Disclaimer 

This library is heavily under development and in pre-alpha state.<br />
msvc support is not properly tested and guarenteed to work. <br />
For contribution, please contact us personally or join our [discord server](https://discord.gg/k5bxQT) <br />

If our library lacks a feature that you need, please leave an issue. <br />
We can raise the priority for most needed features.


<a id="orga8a25a1"></a>

## Example

```C++
#include <mgcpp/mgcpp.hpp>

#define GPU 0

mgcpp::device_matrix<float, GPU> A(4, 3, 2);
mgcpp::device_matrix<float, GPU> B(3, 2, 2);

auto C = mgcpp::eval(A * B);

```

The above code invokes cuBLAS's gemm function. <br />
All mgcpp expressions are lazily computed and optimized using C++ expression template method.


<a id="org0bb6aa9"></a>

## Build

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -G "<Generator>"
make -j4
make install
```

for building without half type support,

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -DUSE_HALF=OFF -G "<Generator>"
make -j4
make install
```

The library probes your system cuda ```gpu-architecture``` and ```gpu-code``` code automatically. <br />
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

You must provide the __FULL PATH__ to your cuda host compiler. <br />
Different cuda versions have different C++ compiler constraints. <br />
For example ```cuda 8.0``` only support gcc up to 5.3.

So, for an example case that you want to use cuda 8.0 for cuda code, gcc-5.3 for native C++ code,

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -DCUDA_HOST_COMPILER=/usr/bin/g++-5.3 -DCMAKE_CXX_COMPILER=g++7 -G "<Generator>"
make -j4
make install
```


<a id="orgd700710"></a>

## Dependencies

-   cmake 3.8 or later
-   gcc 6, clang 3.9, Visual Studio 14.0 (2015) or later
-   [boost-outcome](https://github.com/ned14/boost-outcome)
-   boost
-   cuda 8.0 or later
-   [half](http://half.sourceforge.net/index.html)(optional)
-   [magma](http://icl.cs.utk.edu/magma/)(optional)
-   gtest (optional)


<a id="org6fefac1"></a>

## Planned Features

- Fully tested msvc support.
- Expression template optimization for all operations.
- Tensor type and tensor operations.
- sparse matrix, sparse vector, sparse tensor types.
- Batch matrix type and batch matrix operations.
- cuSPARSE support.
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
