<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. mgcpp</a>
<ul>
<li><a href="#sec-1-1">1.1. Introduction</a></li>
<li><a href="#sec-1-2">1.2. Status</a></li>
<li><a href="#sec-1-3">1.3. Example</a></li>
<li><a href="#sec-1-4">1.4. Build</a></li>
<li><a href="#sec-1-5">1.5. Used Libraries</a></li>
<li><a href="#sec-1-6">1.6. Dependencies</a></li>
<li><a href="#sec-1-7">1.7. <span class="todo TODO">TODO</span> </a></li>
<li><a href="#sec-1-8">1.8. Contact</a></li>
<li><a href="#sec-1-9">1.9. License</a></li>
</ul>
</li>
</ul>
</div>
</div>

# mgcpp<a id="sec-1" name="sec-1"></a> [![Build Status](https://travis-ci.org/Red-Portal/mgcpp.svg?branch=master)](https://travis-ci.org/Red-Portal/mgcpp)

A CUDA based C++ Multi GPU Linear Algebra Library

## Introduction<a id="sec-1-1" name="sec-1-1"></a>

mgcpp is a multi GPU linear algebra library.
It is a wrapper for various CUDA libraries in standard C++.

## Status<a id="sec-1-2" name="sec-1-2"></a>

This library is heavily under development and in pre-alpha state.
For contribution, refer to TODO or contact me.

## Example<a id="sec-1-3" name="sec-1-3"></a>

    using namespace mgcpp;
    
    int const device = 0;
    
    gpu::matrix<float, device> A(2, 4, 2);
    gpu::matrix<float, device> B(4, 3, 4);
    
    gpu::matrix<float, device> C = A * B;
    
    /* or */
    
    auto C = eval(A * B);

The above code invokes the highly optimized cuBLAS library's gemm function.
All operation are lazely computed using expression templates.
GPU computation kernels are also called the least possible.

## Build<a id="sec-1-4" name="sec-1-4"></a>

    git clone --recursive https://github.com/Red-Portal/mgcpp.git
    cmake -G "<Generator>"
    make -j4
    make install

for building without MAGMA or cuSPARSE,

    git clone --recursive https://github.com/Red-Portal/mgcpp.git
    cmake -DUSE_MAGMA=OFF -DUSE_CUSPARSE=OFF -G "<Generator>"
    make -j4
    make install

## Used Libraries<a id="sec-1-5" name="sec-1-5"></a>

These dependencies are optional. 
Not including them might result in some featrues not working properly.

-   cuSPARSE for sparse operations
-   [MAGMA](https://github.com/kjbartel/magma) for hight level linear algebra operations.

## Dependencies<a id="sec-1-6" name="sec-1-6"></a>

-   cmake
-   gcc (>= 6) or clang (>= 3.9) or Visual Studio (>= 15)

-   [outcome](https://github.com/ned14/outcome)
-   cuda (>= 8.0)
-   cuBLAS
-   gtest (optional)

## TODO <a id="sec-1-7" name="sec-1-7"></a>

-   [X] Finish thread context class for managing threads.
-   [ ] Support msvc build.
-   [ ] Add BLAS operations.
-   [ ] Add sparse matrix
-   [ ] Add sparse operations.
-   [ ] Add high level linear algebra operations.
-   [ ] Add benchmark

## Contact<a id="sec-1-8" name="sec-1-8"></a>

Red-Portal
-   msca8h@naver.com
-   msca8h@sogang.ac.kr

## License<a id="sec-1-9" name="sec-1-9"></a>

Copyright RedPortal 2017.

Distributed under the Boost Software License, Version 1.0.
(See accompanying file LICENSE or copy at
<http://www.boost.org/LICENSE_1_0.txt>)
