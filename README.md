# SparseArrayKit.jl

[![Build
Status](https://github.com/Jutho/SparseArrayKit.jl/workflows/CI/badge.svg)](https://github.com/Jutho/SparseArrayKit.jl/actions)
[![Coverage](https://codecov.io/gh/Jutho/SparseArrayKit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Jutho/SparseArrayKit.jl)

A Julia package for sparse multidimensional arrays, aimed particularly at the setting of
very sparse and higher-dimensional arrays (e.g. tensor algebra). This is not a replacement
nor a competitor to Julia's `SparseArrays` standard library and the `SparseMatrixCSC`
format.

The current interface, which is subject to breaking changes, exports a type
`SparseArray{T,N,A} <: AbstractArray{T,N}`. Here, `A` is a type parameter for the underlying
storage format. There are two type aliases which give a concrete implementation to `A`. The
type `SparseDOKArray{T,N}` uses a hash table (`Dict` from Julia's `Base`, could change) to
store keys (of type `CartesianIndex{N}`) and values (of type `T`) of the non-zero data, and
is thus supposed to have O(1) access time for getting and setting individual values. The
type `SparseCOOArray{T,N}` uses an internally defined `SortedVectorDict`, which stores keys
and values of nonzero data in two matching vectors, such that the keys are sorted and
individual keys (and matching values) can be obtained in time `O(log L)`, with `L` the
number of nonzero entries. This format could potentially enable certain optimizations for
global array operations, though none are currently in place, and `SparseDOKArray` is the
preferred format. Indeed, the constructor `SparseArray{T}(undef, dims)` creates a
`SparseDOKArray`.

`SparseArray` instances have a number of method definitions, mostly indexing, basic
arithmetic and methods from the `LinearAlgebra` standard library. Aside from matrix
multiplication, there are no specific matrix methods (such as matrix factorizations) and you
are probably better of with `SparseMatrixCSC` from `SparseArrays` if your problem can be
cast in terms of matrices and vectors. Objects of type `SparseArray` are fully compatible
with the interface from [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl),
and thus with the `@tensor` macro for multidimensional tensor contractions.

here are only three new methods exported by this package, which are `nonzero_keys`,
`nonzero_values` and `nonzero_pairs` wich export iterators (not necessarily editable or
indexable vectors) over the keys, values and key-value pairs of the nonzero entries of the
array. These can be used to define new optimized methods.
