# Julia LinearAlgebra functionality
# ----------------------------------
function LinearAlgebra.mul!(y::SparseArray, α::Number, x::SparseArray)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    _zero!(y)
    for (k, v) in nonzero_pairs(x)
        y[k] = α * v
    end
    return y
end
LinearAlgebra.mul!(y::SparseArray, x::SparseArray, α::Number) = scale!(y, x, α)

function LinearAlgebra.lmul!(α::Number, x::SparseArray)
    # typical occupation in a dict is about 30% from experimental testing
    # the benefits of scaling all values (e.g. SIMD) largely outweight the extra work
    lmul!(α, x.data.vals)
    return x
end
LinearAlgebra.rmul!(x::SparseArray, α::Number) = scale!(x, α)

function LinearAlgebra.ldiv!(a::Number, x::SparseArray)
    ldiv!(a, x.data.vals)
    return x
end
function LinearAlgebra.rdiv!(x::SparseArray, a::Number)
    rdiv!(x.data.vals, a)
    return x
end

function LinearAlgebra.axpby!(α::Number, x::SparseArray, β::Number, y::SparseArray)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    β == one(β) || (iszero(β) ? _zero!(y) : LinearAlgebra.lmul!(β, y))
    for (k, v) in nonzero_pairs(x)
        increaseindex!(y, α * v, k)
    end
    return y
end
# if VERSION < v"1.9"
#     using LinearAlgebra: BlasFloat
#     function LinearAlgebra.axpby!(α::Number, x::SparseArray{T,1}, β::Number, y::SparseArray{T,1}) where {T<:BlasFloat}
#         ax = axes(x)
#         ay = axes(y)
#         ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
#         β == one(β) || (iszero(β) ? _zero!(y) : LinearAlgebra.lmul!(β, y))
#         for (k, v) in nonzero_pairs(x)
#             increaseindex!(y, α * v, k)
#         end
#         return y
#     end
# end

function LinearAlgebra.axpy!(α::Number, x::SparseArray, y::SparseArray)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    for (k, v) in nonzero_pairs(x)
        increaseindex!(y, α * v, k)
    end
    return y
end

function LinearAlgebra.norm(x::SparseArray, p::Real=2)
    return norm(nonzero_values(x), p)
end
LinearAlgebra.dot(x::SparseArray, y::SparseArray) = inner(x, y)

# matrix functions
const SV{T} = SparseArray{T,1}
const SM{T} = SparseArray{T,2}
const ASM{T} = Union{SparseArray{T,2},
                     Transpose{T,<:SparseArray{T,2}},
                     Adjoint{T,<:SparseArray{T,2}}}

LinearAlgebra.mul!(C::SM, A::ASM, B::ASM) = mul!(C, A, B, one(eltype(C)), zero(eltype(C)))
function LinearAlgebra.mul!(C::SM, A::ASM, B::ASM, α::Number, β::Number)
    conjA = A isa Adjoint
    conjB = B isa Adjoint
    oindA = A isa Union{Adjoint,Transpose} ? (2,) : (1,)
    cindA = A isa Union{Adjoint,Transpose} ? (1,) : (2,)
    oindB = B isa Union{Adjoint,Transpose} ? (1,) : (2,)
    cindB = B isa Union{Adjoint,Transpose} ? (2,) : (1,)

    AA = A isa Union{Adjoint,Transpose} ? parent(A) : A
    BB = B isa Union{Adjoint,Transpose} ? parent(B) : B

    return tensorcontract!(C, AA, conjA, oindA, cindA, BB, conjB, oindB, cindB, (1, 2), α,
                           β)
end

LinearAlgebra.adjoint!(C::SM, A::SM) = tensoradd!(C, A, true, (2, 1), One(), Zero())
LinearAlgebra.transpose!(C::SM, A::SM) = tensoradd!(C, A, false, (2, 1), One(), Zero())
