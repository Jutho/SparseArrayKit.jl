# TensorOperations compatiblity
#-------------------------------
function tensoradd!(C::SparseArray{<:Any,N},
                    A::SparseArray{<:Any,N}, conjA::Bool, pA,
                    α::Number=One(), β::Number=One()) where {N}
    (N == length(pA) && TupleTools.isperm(pA)) ||
        throw(ArgumentError("Invalid permutation of length $N: $pA"))
    size(C) == TupleTools.getindices(size(A), pA) ||
        throw(DimensionMismatch("non-matching sizes while adding arrays"))
    scale!(C, β)
    for (IA, vA) in A.data
        IC = CartesianIndex(TupleTools.getindices(IA.I, pA))
        C[IC] += α * (conjA ? conj(vA) : vA)
    end
    return C
end

function tensortrace!(C::SparseArray{<:Any,NC},
                      A::SparseArray{<:Any,NA}, conjA::Bool, p, q1, q2,
                      α::Number=One(), β::Number=Zero()) where {NA,NC}
    NC == length(p) ||
        throw(ArgumentError("Invalid selection of $NC out of $NA: $p"))
    NA - NC == 2 * length(q1) == 2 * length(q2) ||
        throw(ArgumentError("invalid number of trace dimension"))
    pA = (p..., q1..., q2...)
    TupleTools.isperm(pA) ||
        throw(ArgumentError("invalid permutation of length $(ndims(A)): $pA"))

    sizeA = size(A)
    sizeC = size(C)

    TupleTools.getindices(sizeA, q1) == TupleTools.getindices(sizeA, q2) ||
        throw(DimensionMismatch("non-matching trace sizes"))
    sizeC == TupleTools.getindices(sizeA, p) ||
        throw(DimensionMismatch("non-matching sizes"))

    scale!(C, β)
    for (IA, v) in A.data
        IAc1 = CartesianIndex(TupleTools.getindices(IA.I, q1))
        IAc2 = CartesianIndex(TupleTools.getindices(IA.I, q2))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(TupleTools.getindices(IA.I, p))
        C[IC] += α * (conjA ? conj(v) : v)
    end
    return C
end

function tensorcontract!(C::SparseArray,
                         A::SparseArray, conjA::Bool, oindA, cindA,
                         B::SparseArray, conjB::Bool, oindB, cindB,
                         indCinoAB,
                         α::Number=One(), β::Number=Zero())
    pA = (oindA..., cindA...)
    (length(pA) == ndims(A) && TupleTools.isperm(pA)) ||
        throw(ArgumentError("invalid permutation of length $(ndims(A)): $pA"))
    pB = (oindB..., cindB...)
    (length(pB) == ndims(B) && TupleTools.isperm(pB)) ||
        throw(ArgumentError("invalid permutation of length $(ndims(B)): $pB"))
    (length(oindA) + length(oindB) == ndims(C)) ||
        throw(ArgumentError("non-matching output indices in contraction"))
    (ndims(C) == length(indCinoAB) && isperm(indCinoAB)) ||
        throw(ArgumentError("invalid permutation of length $(ndims(C)): $indCinoAB"))

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

    scale!(C, β)

    keysA = sort!(collect(nonzero_keys(A));
                  by=IA -> CartesianIndex(TupleTools.getindices(IA.I, cindA)))
    keysB = sort!(collect(nonzero_keys(B));
                  by=IB -> CartesianIndex(TupleTools.getindices(IB.I, cindB)))

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
                if CartesianIndex(TupleTools.getindices(keysA[jA + 1].I, cindA)) == Ic
                    jA += 1
                else
                    break
                end
            end
            jB = iB
            while jB < length(keysB)
                if CartesianIndex(TupleTools.getindices(keysB[jB + 1].I, cindB)) == Ic
                    jB += 1
                else
                    break
                end
            end
            rA = iA:jA
            rB = iB:jB
            if length(rA) < length(rB)
                for kB in rB
                    IB = keysB[kB]
                    IBo = CartesianIndex(TupleTools.getindices(IB.I, oindB))
                    vB = B[IB]
                    for kA in rA
                        IA = keysA[kA]
                        IAo = CartesianIndex(TupleTools.getindices(IA.I, oindA))
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(TupleTools.getindices(IABo.I, indCinoAB))
                        vA = A[IA]
                        v = α * (conjA ? conj(vA) : vA) * (conjB ? conj(vB) : vB)
                        increaseindex!(C, v, IC)
                    end
                end
            else
                for kA in rA
                    IA = keysA[kA]
                    IAo = CartesianIndex(TupleTools.getindices(IA.I, oindA))
                    vA = A[IA]
                    for kB in rB
                        IB = keysB[kB]
                        IBo = CartesianIndex(TupleTools.getindices(IB.I, oindB))
                        vB = B[IB]
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(TupleTools.getindices(IABo.I, indCinoAB))
                        v = α * (conjA ? conj(vA) : vA) * (conjB ? conj(vB) : vB)
                        increaseindex!(C, v, IC)
                    end
                end
            end
            iA = jA + 1
            iB = jB + 1
        elseif IAc < IBc
            iA += 1
        else
            iB += 1
        end
    end
    return C
end
