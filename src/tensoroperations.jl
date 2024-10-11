const TO = TensorOperations

struct SparseArrayBackend <: TO.AbstractBackend end

# TensorOperations compatiblity
#-------------------------------
function TO.select_backend(::typeof(TO.tensoradd!), C::SparseArray, A::SparseArray)
    return SparseArrayBackend()
end

function TO.tensoradd!(C::SparseArray,
                       A::SparseArray, pA::Index2Tuple, conjA::Bool,
                       α::Number, β::Number, ::SparseArrayBackend, allocator)
    TO.argcheck_tensoradd(C, A, pA)
    TO.dimcheck_tensoradd(C, A, pA)

    scale!(C, β)
    pA_lin = linearize(pA)
    for (IA, vA) in A.data
        IC = CartesianIndex(TupleTools.getindices(IA.I, pA_lin))
        C[IC] += α * (conjA ? conj(vA) : vA)
    end

    return C
end

function TO.select_backend(::typeof(TO.tensortrace!), C::SparseArray, A::SparseArray)
    return SparseArrayBackend()
end

function TO.tensortrace!(C::SparseArray,
                         A::SparseArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                         α::Number, β::Number, ::SparseArrayBackend, allocator)
    TO.argcheck_tensortrace(C, A, p, q)
    TO.dimcheck_tensortrace(C, A, p, q)

    scale!(C, β)
    p_lin = linearize(p)
    for (IA, v) in A.data
        IAc1 = CartesianIndex(TupleTools.getindices(IA.I, q[1]))
        IAc2 = CartesianIndex(TupleTools.getindices(IA.I, q[2]))
        IAc1 == IAc2 || continue

        IC = CartesianIndex(TupleTools.getindices(IA.I, p_lin))
        C[IC] += α * (conjA ? conj(v) : v)
    end

    return C
end

function TO.select_backend(::typeof(TO.tensorcontract!),
                           C::SparseArray, A::SparseArray, B::SparseArray)
    return SparseArrayBackend()
end

function TO.tensorcontract!(C::SparseArray,
                            A::SparseArray, pA::Index2Tuple, conjA::Bool,
                            B::SparseArray, pB::Index2Tuple, conjB::Bool,
                            pAB::Index2Tuple,
                            α::Number, β::Number, ::SparseArrayBackend, allocator)
    TO.argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    TO.dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    scale!(C, β)
    pAB_lin = linearize(pAB)

    keysA = sort!(collect(nonzero_keys(A));
                  by=IA -> CartesianIndex(TupleTools.getindices(IA.I, pA[2])))
    keysB = sort!(collect(nonzero_keys(B));
                  by=IB -> CartesianIndex(TupleTools.getindices(IB.I, pB[1])))

    iA = iB = 1
    @inbounds while iA <= length(keysA) && iB <= length(keysB)
        IA = keysA[iA]
        IB = keysB[iB]
        IAc = CartesianIndex(TupleTools.getindices(IA.I, pA[2]))
        IBc = CartesianIndex(TupleTools.getindices(IB.I, pB[1]))
        if IAc == IBc
            Ic = IAc
            jA = iA
            while jA < length(keysA)
                if CartesianIndex(TupleTools.getindices(keysA[jA + 1].I, pA[2])) == Ic
                    jA += 1
                else
                    break
                end
            end
            jB = iB
            while jB < length(keysB)
                if CartesianIndex(TupleTools.getindices(keysB[jB + 1].I, pB[1])) == Ic
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
                    IBo = CartesianIndex(TupleTools.getindices(IB.I, pB[2]))
                    vB = B[IB]
                    for kA in rA
                        IA = keysA[kA]
                        IAo = CartesianIndex(TupleTools.getindices(IA.I, pA[1]))
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(TupleTools.getindices(IABo.I, pAB_lin))
                        vA = A[IA]
                        v = α * (conjA ? conj(vA) : vA) * (conjB ? conj(vB) : vB)
                        increaseindex!(C, v, IC)
                    end
                end
            else
                for kA in rA
                    IA = keysA[kA]
                    IAo = CartesianIndex(TupleTools.getindices(IA.I, pA[1]))
                    vA = A[IA]
                    for kB in rB
                        IB = keysB[kB]
                        IBo = CartesianIndex(TupleTools.getindices(IB.I, pB[2]))
                        vB = B[IB]
                        IABo = CartesianIndex(IAo, IBo)
                        IC = CartesianIndex(TupleTools.getindices(IABo.I, pAB_lin))
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

function TO.tensoradd_type(TC, ::SparseArray, pA::Index2Tuple, ::Bool)
    return SparseArray{TC,TO.numind(pA)}
end

function TO.tensorcontract_type(TC,
                                ::SparseArray, pA::Index2Tuple, conjA::Bool,
                                ::SparseArray, pB::Index2Tuple, conjB::Bool,
                                pAB::Index2Tuple)
    return SparseArray{TC,TO.numind(pAB)}
end
