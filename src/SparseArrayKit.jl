module SparseArrayKit

using LinearAlgebra
using TupleTools
using Requires

const IndexTuple{N} = NTuple{N,Int}

export SparseArray
export nonzero_pairs, nonzero_keys, nonzero_values, nonzero_length

include("sparsearray.jl")
include("linearalgebra.jl")
include("tensoroperations.jl")

# Initialization
#-----------------
function __init__()
    @require TensorOperations="6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2" begin
        const TO = SparseArrayKit.TensorOperations
        TO.memsize(a::SparseArray) = Base.summarysize(a.data)

        TO.add!(α, A::SparseArray, CA::Symbol, β, C::SparseArray, indCinA) =
            add!(α, A, CA, β, C, indCinA)
        TO.trace!(α, A::SparseArray, CA::Symbol,
                                    β, C::SparseArray,indCinA, cindA1, cindA2) =
            trace!(α, A, CA, β, C, indCinA, cindA1, cindA2)
        TO.contract!(α, A::SparseArray, CA::Symbol, B::SparseArray, CB::Symbol,
                        β, C::SparseArray,
                        oindA::IndexTuple, cindA::IndexTuple,
                        oindB::IndexTuple, cindB::IndexTuple,
                        indCinoAB::IndexTuple,
                        syms::Union{Nothing, NTuple{3,Symbol}} = nothing) =
            contract!(α, A, CA, B, CB, β, C, oindA, cindA, oindB, cindB, indCinoAB, syms)
    end
    @require SparseArrays="2f01184e-e22b-5df5-ae63-d93ebab69eaf" begin
        using .SparseArrays: SparseMatrixCSC, nonzeros, rowvals, nzrange
        Base.convert(T::Type{<:SparseArray}, a::SparseMatrixCSC) = T(a)
        SparseArray(a::SparseMatrixCSC) = SparseArray{eltype(a)}(a)
        function SparseArray{T}(a::SparseMatrixCSC) where T
            b = SparseArray{T}(undef, size(a))
            rv = rowvals(a)
            nzv = nonzeros(a)
            @inbounds for col = 1:size(a, 2)
                for i = nzrange(a, col)
                    row = rv[i]
                    v = nzv[i]
                    b[row, col] = v
                end
            end
            return b
        end
    end
end

end
