# Basic arithmetic
Base.:*(a::Number, x::SparseArray) = LinearAlgebra.lmul!(a, copy(x))
Base.:*(x::SparseArray, a::Number) = LinearAlgebra.rmul!(copy(x), a)
Base.:\(a::Number, x::SparseArray) = LinearAlgebra.ldiv!(a, copy(x))
Base.:/(x::SparseArray, a::Number) = LinearAlgebra.rdiv!(copy(x), a)
Base.:+(x::SparseArray, y::SparseArray) = LinearAlgebra.axpy!(+1, y, copy(x))
Base.:-(x::SparseArray, y::SparseArray) = LinearAlgebra.axpy!(-1, y, copy(x))
Base.:-(x::SparseArray) = LinearAlgebra.lmul!(-one(eltype(x)), copy(x))

Base.zero(x::SparseArray) = similar(x)
Base.iszero(x::SparseArray) = nonzero_length(x) == 0

function Base.one(x::SparseArray{<:Any,2})
    m, n = size(x)
    m == n ||
        throw(DimensionMismatch("multiplicative identity defined only for square matrices"))

    u = similar(x)
    @inbounds for i = 1:m
        u[i,i] = one(eltype(x))
    end
end

# comparison
function Base.:(==)(x::SparseArray, y::SparseArray)
    keys = collect(nonzero_keys(x))
    intersect!(keys, nonzero_keys(y))
    if !(length(keys) == length(nonzero_keys(x)) == length(nonzero_keys(y)))
        return false
    end
    for I in keys
        x[I] == y[I] || return false
    end
    return true
end

# TODO
# Base.permutedims!
# LinearAlgebra.mul!
# LinearAlgebra.adjoint!
# LinearAlgebra.transpose!

# Vector space functions
#------------------------
function LinearAlgebra.lmul!(a::Number, x::SparseArray)
    lmul!(a, x.data.vals)
    # typical occupation in a dict is about 30% from experimental testing
    # the benefits of scaling all values (e.g. SIMD) largely outweight the extra work
    return x
end
function LinearAlgebra.rmul!(x::SparseArray, a::Number)
    rmul!(x.data.vals, a)
    return x
end
function LinearAlgebra.ldiv!(a::Number, x::SparseArray)
    ldiv!(a, x.data.vals)
    return x
end
function LinearAlgebra.rdiv!(x::SparseArray, a::Number)
    rdiv!(x.data.vals, a)
    return x
end
function LinearAlgebra.axpby!(α, x::SparseArray, β, y::SparseArray)
    β == one(β) || (iszero(β) ? _zero!(y) : LinearAlgebra.lmul!(β, y))
    for (k, v) in nonzero_pairs(x)
        y[k] += α*v
    end
    return y
end
function LinearAlgebra.axpy!(α, x::SparseArray, y::SparseArray)
    for (k, v) in nonzero_pairs(x)
        y[k] += α*v
    end
    return y
end

function LinearAlgebra.norm(x::SparseArray, p::Real = 2)
    norm(nonzero_values(x), p)
end

function LinearAlgebra.dot(x::SparseArray, y::SparseArray)
    size(x) == size(y) || throw(DimensionMismatch("dot arguments have different size"))
    s = dot(zero(eltype(x)), zero(eltype(y)))
    if nonzero_length(x) >= nonzero_length(y)
        @inbounds for I in nonzero_keys(x)
            s += dot(x[I], y[I])
        end
    else
        @inbounds for I in nonzero_keys(y)
            s += dot(x[I], y[I])
        end
    end
    return s
end
