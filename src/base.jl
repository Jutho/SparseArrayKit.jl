# Julia Base functionality
#--------------------------
# arithmetic
Base.:*(a::Number, x::SparseArray) = mul!(similar(x, Base.promote_eltypeof(a, x)), a, x)
Base.:*(x::SparseArray, a::Number) = mul!(similar(x, Base.promote_eltypeof(a, x)), x, a)
function Base.:\(a::Number, x::SparseArray)
    return mul!(similar(x, Base.promote_eltypeof(a, x)), inv(a), x)
end
function Base.:/(x::SparseArray, a::Number)
    return mul!(similar(x, Base.promote_eltypeof(a, x)), x, inv(a))
end
function Base.:+(x::SparseArray, y::SparseArray)
    return (T = Base.promote_eltypeof(x, y); axpy!(+one(T), y, copy!(similar(x, T), x)))
end
function Base.:-(x::SparseArray, y::SparseArray)
    return (T = Base.promote_eltypeof(x, y); axpy!(-one(T), y, copy!(similar(x, T), x)))
end

Base.:-(x::SparseArray) = LinearAlgebra.lmul!(-one(eltype(x)), copy(x))

Base.zero(x::SparseArray) = similar(x)
Base.iszero(x::SparseArray) = nonzero_length(x) == 0

function Base.one(x::SparseArray{<:Any,2})
    m, n = size(x)
    m == n ||
        throw(DimensionMismatch("multiplicative identity defined only for square matrices"))

    u = similar(x)
    @inbounds for i in 1:m
        u[i, i] = one(eltype(x))
    end
    return u
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

# in-place conjugation
function Base.conj!(x::SparseArray)
    conj!(x.data.vals)
    return x
end

# array manipulation
function Base.permutedims!(dst::SparseArray, src::SparseArray, p)
    return tensoradd!(dst, src, (tuple(p...), ()), false)
end

function Base.reshape(parent::SparseArray{T}, dims::Dims) where {T}
    n = length(parent)
    n == prod(dims) ||
        throw(DimensionMismatch("parent has $n elements, which is incompatible with size $dims"))
    child = SparseArray{T}(undef, dims)
    lin_inds = LinearIndices(parent)
    new_cart_inds = CartesianIndices(dims)
    for (ky, vl) in nonzero_pairs(parent)
        child[new_cart_inds[lin_inds[ky]]] = vl
    end
    return child
end
