# TensorOperations compatiblity
#-------------------------------
function tensoradd!(C::SparseArray{<:Any,N}, indCinA,
                    A::SparseArray{<:Any,N}, CA::Symbol,
                    α::Number=One(), β::Number=One()) where {N}
    (N == length(indCinA) && TupleTools.isperm(indCinA)) ||
        throw(ArgumentError("Invalid permutation of length $N: $indCinA"))
    size(C) == TupleTools.getindices(size(A), indCinA) ||
        throw(DimensionMismatch("non-matching sizes while adding arrays"))
    scale!(C, β)
    for (IA, vA) in A.data
        IC = CartesianIndex(TupleTools.getindices(IA.I, indCinA))
        C[IC] += α * (CA == :C ? conj(vA) : vA)
    end
    return C
end

function tensortrace!(C::SparseArray{<:Any,NC}, indCinA,
                      A::SparseArray{<:Any,NA}, CA::Symbol, cindA1, cindA2,
                      α::Number=One(), β::Number=Zero()) where {NA,NC}
    NC == length(indCinA) ||
        throw(ArgumentError("Invalid selection of $NC out of $NA: $indCinA"))
    NA - NC == 2 * length(cindA1) == 2 * length(cindA2) ||
        throw(ArgumentError("invalid number of trace dimension"))
    pA = (indCinA..., cindA1..., cindA2...)
    TupleTools.isperm(pA) ||
        throw(ArgumentError("invalid permutation of length $(ndims(A)): $pA"))

    sizeA = size(A)
    sizeC = size(C)

    TupleTools.getindices(sizeA, cindA1) == TupleTools.getindices(sizeA, cindA2) ||
        throw(DimensionMismatch("non-matching trace sizes"))
    sizeC == TupleTools.getindices(sizeA, indCinA) ||
        throw(DimensionMismatch("non-matching sizes"))

    scale!(C, β)
    for (IA, v) in A.data
        IAc1 = CartesianIndex(TupleTools.getindices(IA.I, cindA1))
        IAc2 = CartesianIndex(TupleTools.getindices(IA.I, cindA2))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(TupleTools.getindices(IA.I, indCinA))
        C[IC] += α * (CA == :C ? conj(v) : v)
    end
    return C
end

function tensorcontract!(C::SparseArray, indCinoAB,
                         A::SparseArray, CA::Symbol, oindA, cindA,
                         B::SparseArray, CB::Symbol, oindB, cindB,
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
                        increaseindex!(C,
                                       α * (CA == :C ? conj(vA) : vA) *
                                       (CB == :C ? conj(vB) : vB), IC)
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
                        increaseindex!(C,
                                       α * (CA == :C ? conj(vA) : vA) *
                                       (CB == :C ? conj(vB) : vB), IC)
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
