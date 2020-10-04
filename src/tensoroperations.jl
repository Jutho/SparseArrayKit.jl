# TensorOperations compatiblity
#-------------------------------
import .TensorOperations: memsize, add!, trace!, contract!

memsize(A::SparseArray) = memsize(A.data)

function add!(α, A::SparseArray{<:Any, N}, CA::Symbol,
                β, C::SparseArray{<:Any, N}, indCinA) where {N}

    (N == length(indCinA) && TupleTools.isperm(indCinA)) ||
        throw(IndexError("Invalid permutation of length $N: $indCinA"))
    size(C) == TupleTools.getindices(size(A), indCinA) ||
        throw(DimensionMismatch("non-matching sizes while adding arrays"))

    β == one(β) || LinearAlgebra.lmul!(β, C);
    for (kA, vA) in A.data
        kC = CartesianIndex(TupleTools.getindices(kA.I, indCinA))
        C[kC] += α* (conjA == :C ? conj(vA) : vA)
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

    β == one(β) || LinearAlgebra.lmul!(β, C);

    for (kA, v) in A.data
        kAc1 = CartesianIndex(TupleTools.getindices(kA.I, cindA1))
        kAc2 = CartesianIndex(TupleTools.getindices(kA.I, cindA2))
        kAc1 == kAc2 || continue

        kC = CartesianIndex(TupleTools.getindices(kC.I, indCinA))
        C[kC] += α * (conjA == :C ? conj(v) : v)
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

    β == one(β) || LinearAlgebra.lmul!(β, C);

    for (kA, vA) in A.data
        kAc = CartesianIndex(TupleTools.getindices(kA.I, cindA))
        kAo = CartesianIndex(TupleTools.getindices(kA.I, oindA))
        for (kB, vB) in B.data
            kBc = CartesianIndex(TupleTools.getindices(kB.I, cindB))
            kAc == kBc || continue

            kBo = CartesianIndex(TupleTools.getindices(kB.I, oindB))

            kABo = CartesianIndex(kAo, kBo)

            kC = CartesianIndex(TupleTools.getindices(kABo.I, indCinoAB))

            C[kC] += α * (CA == :C ? conj(vA) : vA) * (CB == :C ? conj(vB) : vB)
        end
    end
    C
end
