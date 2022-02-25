
# Basic arithmetic
Base.:*(a::Number, x::SparseArray) =
    mul!(similar(x, Base.promote_eltypeof(a, x)), a, x)
Base.:*(x::SparseArray, a::Number) =
    mul!(similar(x, Base.promote_eltypeof(a, x)), x, a)
Base.:\(a::Number, x::SparseArray) =
    mul!(similar(x, Base.promote_eltypeof(a, x)), inv(a), x)
Base.:/(x::SparseArray, a::Number) =
    mul!(similar(x, Base.promote_eltypeof(a, x)), x, inv(a))
Base.:+(x::SparseArray, y::SparseArray) =
    (T = Base.promote_eltypeof(x, y); axpy!(+one(T), y, copy!(similar(x, T), x)))
Base.:-(x::SparseArray, y::SparseArray) =
    (T = Base.promote_eltypeof(x, y); axpy!(-one(T), y, copy!(similar(x, T), x)))

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

# Vector space functions
#------------------------
function Base.conj!(x::SparseArray)
    conj!(x.data.vals)
    return x
end
function LinearAlgebra.mul!(dst::SparseArray, a::Number, src::SparseArray)
    _zero!(dst)
    for (k, v) in nonzero_pairs(src)
        dst[k] = a*v
    end
    return dst
end
function LinearAlgebra.mul!(dst::SparseArray, src::SparseArray, a::Number)
    _zero!(dst)
    for (k, v) in nonzero_pairs(src)
        dst[k] = v*a
    end
    return dst
end
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
function LinearAlgebra.axpby!(α::Number, x::SparseArray, β, y::SparseArray)
    β == one(β) || (iszero(β) ? _zero!(y) : LinearAlgebra.lmul!(β, y))
    for (k, v) in nonzero_pairs(x)
        increaseindex!(y, α*v, k)
    end
    return y
end
function LinearAlgebra.axpy!(α::Number, x::SparseArray, y::SparseArray)
    for (k, v) in nonzero_pairs(x)
        increaseindex!(y, α*v, k)
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

# permutedims
Base.permutedims!(dst::SparseArray, src::SparseArray, p) =
    add!(one(eltype(dst)), src, :N, zero(eltype(dst)), dst, tuple(p...))

# matrix functions
const SV{T} = SparseArray{T,1}
const SM{T} = SparseArray{T, 2}
const ASM{T} = Union{SparseArray{T, 2},
                    Transpose{T, <:SparseArray{T,2}},
                    Adjoint{T, <:SparseArray{T,2}}}

LinearAlgebra.mul!(C::SM, A::ASM, B::ASM) = mul!(C, A, B, one(eltype(C)), zero(eltype(C)))
function LinearAlgebra.mul!(C::SM, A::ASM, B::ASM, α::Number, β::Number)
    CA = A isa Adjoint ? :C : :N
    CB = B isa Adjoint ? :C : :N
    oindA = A isa Union{Adjoint,Transpose} ? (2,) : (1,)
    cindA = A isa Union{Adjoint,Transpose} ? (1,) : (2,)
    oindB = B isa Union{Adjoint,Transpose} ? (1,) : (2,)
    cindB = B isa Union{Adjoint,Transpose} ? (2,) : (1,)

    AA = A isa Union{Adjoint,Transpose} ? parent(A) : A
    BB = B isa Union{Adjoint,Transpose} ? parent(B) : B

    contract!(α, AA, CA, BB, CB, β, C, oindA, cindA, oindB, cindB, (1, 2))
end

function LinearAlgebra.adjoint!(C::SM, A::SM)
    add!(one(eltype(C)), A, :C, zero(eltype(C)), C, (2, 1))
    return C
end
function LinearAlgebra.transpose!(C::SM, A::SM)
    add!(one(eltype(C)), A, :N, zero(eltype(C)), C, (2, 1))
    return C
end
