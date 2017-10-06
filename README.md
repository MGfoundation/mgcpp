<div id="table-of-contents">
<h2>Table of Contents</h2>
<div id="text-table-of-contents">
<ul>
<li><a href="#sec-1">1. mgcpp</a>
<ul>
<li><a href="#sec-1-1">1.1. Introduction</a></li>
<li><a href="#sec-1-2">1.2. Libraries</a></li>
<li><a href="#sec-1-3">1.3. Dependencies</a></li>
<li><a href="#sec-1-4">1.4. <span class="todo TODO">TODO</span> </a></li>
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

## Libraries<a id="sec-1-2" name="sec-1-2"></a>

These dependencies are optional. 
Not including them might result in some featrues not working properly.

-   cuBLAS for various BLAS operations.
-   cuSPARSE for sparse operations
-   MAGMA for hight level linear algebra operations.

## Dependencies<a id="sec-1-3" name="sec-1-3"></a>

-   gtest (optional)
-   cuda (>8.0)

## TODO <a id="sec-1-4" name="sec-1-4"></a>

-   [X] Finish thread context class for managing threads.
-   [ ] Support msvc build.
-   [ ] Add BLAS operations.
-   [ ] Add sparse matrix
-   [ ] Add sparse operations.
-   [ ] Add high level linear algebra operations.
-   [ ] Add benchmark
