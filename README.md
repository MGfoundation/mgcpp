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

# mgcpp
[![Build Status](https://travis-ci.org/MGfoundation/mgcpp.svg?branch=master)](https://travis-ci.org/MGfoundation/mgcpp)
[![Documentation](https://codedocs.xyz/MGfoundation/mgcpp.svg)](https://codedocs.xyz/MGfoundation/mgcpp/)

<a id="org9dd78d2"></a>

## Introduction

mgcpp is a CUDA based C++ linear algebra library. <br />
It provides a standard C++ interface without any CUDA specific syntax. <br /> <br />

This library is heavily under development and in pre-alpha state.<br />
If our library lacks a feature that you need, please leave an issue. <br />
We can raise the priority for most needed features. <br /> <br />

For contribution, please contact us personally or join our [discord server](https://discord.gg/k5bxQT) <br />

<a id="orga8a25a1"></a>

## Example

```C++
#include <mgcpp/mgcpp.hpp>

mgcpp::device_matrix<float> A({4, 3}, 2);
mgcpp::device_matrix<float> B({3, 2}, 2);

auto C = ref(A) * ref(B);

// Lazy evaluation
auto result = mgcpp::eval(C);

```

The above code invokes cuBLAS's gemm function. <br />
All mgcpp expressions are lazily computed and optimized using C++ expression template method.


<a id="org0bb6aa9"></a>

## Notable Features

- cuBLAS backend for efficient BLAS computation.
- Expression templates notation for efficient, expressive code
- Half precision operation for extra performance on modern Nvidia GPU architectures.
- Automatic GPU memory management.
- Fill, reduce, outer-product and more custom BLAS extensions.

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


*-DISCLAIMER-* <br />
Windows build is currently not supported because CUDA 9.1 doesn't support latest version of Visual Studio 15. <br />
Windows support should be back when this issue is resolved. <br /> <br />
On Windows, after installing all dependencies, execute the following on the Developer Command Prompt for VS 2017:

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cd mgcpp
mkdir build
cd build
cmake -G "Visual Studio 15 2017 Win64" -T v140,host=x64 ..
msbuild ALL_BUILD.vcxproj
```

<a id="orgd700710"></a>

## Dependencies

-   cmake 3.8 or later
-   gcc 6, clang 3.9, ~~Visual Studio 14.0 (2015) or later~~(currently not supported)
-   [boost-outcome](https://github.com/ned14/boost-outcome)
-   boost
-   cuda 8.0 or later
-   [half](http://half.sourceforge.net/index.html)(optional)
-   [magma](http://icl.cs.utk.edu/magma/)(optional)
-   google test (optional)
-   google benchmark (optional)


<a id="org6fefac1"></a>

## Planned Features

- Various GPU memory allocators.
- Expression template optimization for all operations.
- Convolution operation.
- sparse matrix, sparse vector, sparse tensor types.
- Batch matrix type and batch matrix operations.
- Tensor type and tensor operations.
- cuSPARSE support.
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
