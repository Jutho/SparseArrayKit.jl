# Julia LinearAlgebra functionality
# ----------------------------------
# vector space behavior
function LinearAlgebra.mul!(dst::SparseArray, a::Number, src::SparseArray)
    _zero!(dst)
    for (k, v) in nonzero_pairs(src)
        dst[k] = a * v
    end
    return dst
end
function LinearAlgebra.mul!(dst::SparseArray, src::SparseArray, a::Number)
    _zero!(dst)
    for (k, v) in nonzero_pairs(src)
        dst[k] = v * a
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
        increaseindex!(y, α * v, k)
    end
    return y
end
function LinearAlgebra.axpy!(α::Number, x::SparseArray, y::SparseArray)
    for (k, v) in nonzero_pairs(x)
        increaseindex!(y, α * v, k)
    end
    return y
end

function LinearAlgebra.norm(x::SparseArray, p::Real=2)
    return norm(nonzero_values(x), p)
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

# matrix functions
const SV{T} = SparseArray{T,1}
const SM{T} = SparseArray{T,2}
const ASM{T} = Union{SparseArray{T,2},
                     Transpose{T,<:SparseArray{T,2}},
                     Adjoint{T,<:SparseArray{T,2}}}

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

    return tensorcontract!(C, (1, 2), AA, CA, oindA, cindA, BB, CB, oindB, cindB, α, β)
end

LinearAlgebra.adjoint!(C::SM, A::SM) = tensoradd!(C, (2, 1), A, :C, true, false)
LinearAlgebra.transpose!(C::SM, A::SM) = tensoradd!(C, (2, 1), A, :N, true, false)
