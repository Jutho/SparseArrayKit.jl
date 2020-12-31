# SparseArrayKit.jl

[![Build
Status](https://github.com/Jutho/SparseArrayKit.jl/workflows/CI/badge.svg)](https://github.com/Jutho/SparseArrayKit.jl/actions)
[![Coverage](https://codecov.io/gh/Jutho/SparseArrayKit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Jutho/SparseArrayKit.jl)

A Julia package for sparse multidimensional arrays, aimed particularly at the setting of
very sparse and higher-dimensional arrays (e.g. tensor algebra). This is not a replacement
nor a competitor to Julia's `SparseArrays` standard library and the `SparseMatrixCSC`
format.

The current interface, which is subject to breaking changes, exports a type
`SparseArray{T,N} <: AbstractArray{T,N}`. This type uses a hash table (`Dict` from Julia's 
`Base`, could change) to store keys (of type `CartesianIndex{N}`) and values (of type `T`)
of the non-zero data (i.e. a dictionary-of-keys storage format), and is thus supposed to
have O(1) access time for getting and setting individual values. Other storage formats for
sparse arrays could in the future be experimented with.

`SparseArray` instances have a number of method definitions, mostly indexing, basic
arithmetic and methods from the `LinearAlgebra` standard library. Aside from matrix
multiplication, there are no specific matrix methods (such as matrix factorizations) and you
are probably better off with `SparseMatrixCSC` from `SparseArrays` if your problem can be
cast in terms of matrices and vectors. There is a fast conversion path from
`SparseMatrixCSC` to `SparseArray` (but not yet the other way around).

Objects of type `SparseArray` are fully compatible with the interface from
[TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl), and thus with the
`@tensor` macro for multidimensional tensor contractions.

There are only three new methods exported by this package, which are `nonzero_keys`,
`nonzero_values` and `nonzero_pairs` which export iterators (not necessarily editable or
indexable vectors) over the keys, values and key-value pairs of the nonzero entries of the
array. These can be used to define new optimized methods.
