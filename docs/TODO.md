# MGCPP Todo List
## GPU Matrix
-   [ ] add constructor and assign operator for expressions
-   [ ] add test
The problem is, by implementing this using a 'enable_if'ed perfect forwarding overload, <br />
the overload attracts all type. <br />
Even cpu\:\:matrix that has a dedicated constructor does not work. <br />

-   [ ] add overload so std::pair can be used as dimension
-   [ ] add test
Making methods that receive std::pair instead of i and j would make sense <br />
since matrix has a method called 'shape' which returns std::pair. <br />


-   [ ] add tests for constructing and using gpu::matrix in multiple GPUs

-   [ ] add tests for column major storage order

## BLAS lv1
-   [ ] add sum array
This can be done by computing the dot product with a 1-initialized vector
