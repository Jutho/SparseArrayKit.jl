module SparseArrayTensorOperations

@static if isdefined(Base, :get_extension)
    using TensorOperations: TensorOperations, Index2Tuple, linearize
else
    using ..TensorOperations: TensorOperations, Index2Tuple, linearize
end

using SparseArrayKit

TensorOperations.memsize(a::SparseArray) = Base.summarysize(a.data)

Backends = Union{TensorOperations.AbstractBackend,TensorOperations.StridedBackend,
                 TensorOperations.StridedBLASBackend}
for backend in (TensorOperations.AbstractBackend, TensorOperations.StridedBackend,
                TensorOperations.StridedBLASBackend)
    @eval function TensorOperations.tensoradd!(::($backend),
                                               C::SparseArray,
                                               A::SparseArray, pA::Index2Tuple,
                                               conjA::Symbol,
                                               α::Number, β::Number)
        return SparseArrayKit.add!(α, A, conjA, β, C, linearize(pA))
    end

    @eval function TensorOperations.tensortrace!(::($backend),
                                                 C::SparseArray,
                                                 pC::Index2Tuple, A::SparseArray,
                                                 pA::Index2Tuple,
                                                 conjA::Symbol, α::Number,
                                                 β::Number)
        return SparseArrayKit.trace!(α, A, conjA, β, C, linearize(pC), pA[1], pA[2])
    end

    @eval function TensorOperations.tensorcontract!(::($backend),
                                                    C::SparseArray, pC::Index2Tuple,
                                                    A::SparseArray,
                                                    pA::Index2Tuple, conjA::Symbol,
                                                    B::SparseArray,
                                                    pB::Index2Tuple, conjB::Symbol,
                                                    α::Number,
                                                    β::Number)
        return SparseArrayKit.contract!(α, A, conjA, B, conjB, β, C, pA[1], pA[2],
                                        pB[2], pB[1],
                                        linearize(pC))
    end
end

function TensorOperations.tensoradd_type(TC, A::SparseArray, pA::Index2Tuple,
                                         conjA::Symbol)
    return SparseArray{TC,sum(length.(pA))}
end

function TensorOperations.tensorcontract_type(TC, pC, A::SparseArray, pA, conjA,
                                              B::SparseArray, pB, conjB)
    return SparseArray{TC,sum(length.(pC))}
end

end
