# MGCPP Todo List
## GPU Matrix
-   [ ] __add constructor and assign operator for expressions__ <br />
-   [ ] __add test for expression constructor, assign operator__ <br />
The problem is, by implementing this using a 'enable\_if'ed perfect forwarding overload, <br />
the overload attracts all type. <br />
Even cpu\:\:matrix that has a dedicated constructor does not work. <br />

-   [ ] __add overload for std::pair can represent dimension__  <br />
-   [ ] __add tests for std::pair dimension representation overloads__   <br />
Making methods that receive std::pair instead of i and j would make sense <br />
since matrix has a method called 'shape' which returns std::pair. <br />


-   [ ] __add tests for constructing and using gpu::matrix in multiple GPUs__ <br />

-   [ ] __add tests for column major storage order__ <br />

## BLAS lv1
-   [ ] __add sum array operation__
This can be done by computing the dot product with a 1-initialized vector
