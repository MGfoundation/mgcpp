- [mgcpp [https://travis-ci.org/Red-Portal/mgcpp](https://travis-ci.org/Red-Portal/mgcpp.svg?branch=master)](#org9bc6c9e)
  - [Introduction](#org17c8cf8)
  - [Status](#org36e6ba8)
  - [Example](#org4ee4471)
  - [Build](#org66354a2)
  - [Dependencies](#org580f735)
  - [TODO](#orgfeabbd6)
  - [Contact](#org8b8c218)
  - [License](#orgbfe08a3)


<a id="org9bc6c9e"></a>

# mgcpp [https://travis-ci.org/Red-Portal/mgcpp](https://travis-ci.org/Red-Portal/mgcpp.svg?branch=master)

C++ Multi GPU Math Library Based on CUDA


<a id="org17c8cf8"></a>

## Introduction

mgcpp is a multi GPU linear algebra library. It is a wrapper for various CUDA libraries in standard C++.


<a id="org36e6ba8"></a>

## Status

This library is heavily under development and in pre-alpha state. For contribution, please refer to TODO or contact me.


<a id="org4ee4471"></a>

## Example

```C++
#include <mgcpp/mgcpp.hpp>

#define GPU 0

mgcpp::device_matrix<float, GPU> A(4, 3, 2);
mgcpp::device_matrix<float, GPU> A(3, 2, 2);

auto C = mgcpp::eval(A * B);

```

The above code invokes cuBLAS's gemm function. All operations are lazely computed and graph optimized using C++ expression templates.


<a id="org66354a2"></a>

## Build

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -G "<Generator>"
make -j4
make install
```

for building without MAGMA or cuSPARSE,

```shell
git clone --recursive https://github.com/Red-Portal/mgcpp.git
cmake -DUSE_MAGMA=OFF -DUSE_CUSPARSE=OFF -G "<Generator>"
make -j4
make install
```


<a id="org580f735"></a>

## Dependencies

-   cmake
-   gcc (>= 6) or clang (>= 3.9) or Visual Studio (>= 15)
-   [outcome](https://github.com/ned14/outcome)
-   cuda (>= 8.0)
-   cuBLAS
-   [MAGMA](https://github.com/kjbartel/magma) (optional)
-   cuSPARSE (optional)
-   gtest (optional)


<a id="orgfeabbd6"></a>

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


<a id="org8b8c218"></a>

## Contact

Red-Portal

-   msca8h@naver.com
-   msca8h@sogang.ac.kr


<a id="orgbfe08a3"></a>

## License

Copyright RedPortal 2017.

Distributed under the Boost Software License, Version 1.0. (See accompanying file LICENSE or copy at <http://www.boost.org/LICENSE_1_0.txt>)
