# TensorOperations compatiblity
#-------------------------------
import .TensorOperations: memsize, add!, trace!, contract!
using .TensorOperations: IndexTuple, TupleTools

memsize(A::SparseArray) = memsize(A.data)

function add!(α, A::SparseArray{<:Any, N}, CA::Symbol,
                β, C::SparseArray{<:Any, N}, indCinA) where {N}

    (N == length(indCinA) && TupleTools.isperm(indCinA)) ||
        throw(IndexError("Invalid permutation of length $N: $indCinA"))
    size(C) == TupleTools.getindices(size(A), indCinA) ||
        throw(DimensionMismatch("non-matching sizes while adding arrays"))

    β == one(β) || (iszero(β) ? _zero!(C) : LinearAlgebra.lmul!(β, C))
    for (IA, vA) in A.data
        IC = CartesianIndex(TupleTools.getindices(IA.I, indCinA))
        C[IC] += α* (CA == :C ? conj(vA) : vA)
    end
    C
end

function trace!(α, A::SparseArray{<:Any, NA}, CA::Symbol, β, C::SparseArray{<:Any, NC},
                indCinA, cindA1, cindA2) where {NA,NC}

    NC == length(indCinA) ||
        throw(IndexError("Invalid selection of $NC out of $NA: $indCinA"))
    NA-NC == 2*length(cindA1) == 2*length(cindA2) ||
        throw(IndexError("invalid number of trace dimension"))
    pA = (indCinA..., cindA1..., cindA2...)
    TupleTools.isperm(pA) ||
        throw(IndexError("invalid permutation of length $(ndims(A)): $pA"))

    sizeA = size(A)
    sizeC = size(C)

    TupleTools.getindices(sizeA, cindA1) == TupleTools.getindices(sizeA, cindA2) ||
        throw(DimensionMismatch("non-matching trace sizes"))
    sizeC == TupleTools.getindices(sizeA, indCinA) ||
        throw(DimensionMismatch("non-matching sizes"))

    β == one(β) || (iszero(β) ? _zero!(C) : LinearAlgebra.lmul!(β, C))
    for (IA, v) in A.data
        IAc1 = CartesianIndex(TupleTools.getindices(IA.I, cindA1))
        IAc2 = CartesianIndex(TupleTools.getindices(IA.I, cindA2))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(TupleTools.getindices(IA.I, indCinA))
        C[IC] += α * (CA == :C ? conj(v) : v)
    end
    return C
end

function contract!(α, A::SparseArray, CA::Symbol, B::SparseArray, CB::Symbol,
                    β, C::SparseArray,
                    oindA::IndexTuple, cindA::IndexTuple,
                    oindB::IndexTuple, cindB::IndexTuple,
                    indCinoAB::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing)

    pA = (oindA...,cindA...)
    (length(pA) == ndims(A) && TupleTools.isperm(pA)) ||
        throw(IndexError("invalid permutation of length $(ndims(A)): $pA"))
    pB = (oindB...,cindB...)
    (length(pB) == ndims(B) && TupleTools.isperm(pB)) ||
        throw(IndexError("invalid permutation of length $(ndims(B)): $pB"))
    (length(oindA) + length(oindB) == ndims(C)) ||
        throw(IndexError("non-matching output indices in contraction"))
    (ndims(C) == length(indCinoAB) && isperm(indCinoAB)) ||
        throw(IndexError("invalid permutation of length $(ndims(C)): $indCinoAB"))

    sizeA = size(A)
    sizeB = size(B)
    sizeC = size(C)

    csizeA = TupleTools.getindices(sizeA, cindA)
    csizeB = TupleTools.getindices(sizeB, cindB)
    osizeA = TupleTools.getindices(sizeA, oindA)
    osizeB = TupleTools.getindices(sizeB, oindB)

    csizeA == csizeB ||
        throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
    TupleTools.getindices((osizeA..., osizeB...), indCinoAB) == size(C) ||
        throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))

    β == one(β) || (iszero(β) ? _zero!(C) : LinearAlgebra.lmul!(β, C))

    keysA = sort!(collect(nonzero_keys(A)),
                    by = IA->CartesianIndex(TupleTools.getindices(IA.I, cindA)))
    keysB = sort!(collect(nonzero_keys(B)),
                    by = IB->CartesianIndex(TupleTools.getindices(IB.I, cindB)))

    iA = iB = 1
    @inbounds while iA <= length(keysA) && iB <= length(keysB)
        IA = keysA[iA]
        IB = keysB[iB]
        IAc = CartesianIndex(TupleTools.getindices(IA.I, cindA))
        IBc = CartesianIndex(TupleTools.getindices(IB.I, cindB))
        if IAc == IBc
            Ic = IAc
            jA = iA
            while jA < length(keysA)
                if CartesianIndex(TupleTools.getindices(keysA[jA+1].I, cindA)) == Ic
                    jA += 1
                else
                    break
                end
            end
            jB = iB
            while jB < length(keysB)
                if CartesianIndex(TupleTools.getindices(keysB[jB+1].I, cindB)) == Ic
                    jB += 1
                else
                    break
                end
            end
            for kB in iB:jB, kA in iA:jA
                IA = keysA[kA]
                IB = keysB[kB]
                IAo = CartesianIndex(TupleTools.getindices(IA.I, oindA))
                IBo = CartesianIndex(TupleTools.getindices(IB.I, oindB))
                IABo = CartesianIndex(IAo, IBo)
                IC = CartesianIndex(TupleTools.getindices(IABo.I, indCinoAB))
                vA = A[IA]
                vB = B[IB]
                C[IC] += α * (CA == :C ? conj(vA) : vA) * (CB == :C ? conj(vB) : vB)
            end
            iA = jA+1
            iB = jB+1
        elseif IAc < IBc
            iA += 1
        else
            iB += 1
        end
    end
    C
end
