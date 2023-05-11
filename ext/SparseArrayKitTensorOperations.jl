module SparseArrayKitTensorOperations

@static if isdefined(Base, :get_extension)
    using TensorOperations: TensorOperations, Index2Tuple, linearize
else
    using ..TensorOperations: TensorOperations, Index2Tuple, linearize
end
const TO = TensorOperations

using SparseArrayKit: add!, trace!, contract!, SparseArray

function TO.tensoradd!(C::SparseArray,
                       A::SparseArray, pA::Index2Tuple, conjA::Symbol,
                       α::Number, β::Number)
    return add!(α, A, conjA, β, C, linearize(pA))
end

function TO.tensortrace!(C::SparseArray, pC::Index2Tuple,
                         A::SparseArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number)
    return trace!(α, A, conjA, β, C, linearize(pC), pA[1], pA[2])
end

function TO.tensorcontract!(C::SparseArray, pC::Index2Tuple,
                            A::SparseArray, pA::Index2Tuple, conjA::Symbol,
                            B::SparseArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    return contract!(α, A, conjA, B, conjB, β, C, pA[1], pA[2], pB[2], pB[1], linearize(pC))
end

function TO.tensoradd_type(TC, ::SparseArray, pA::Index2Tuple, ::Symbol)
    return SparseArray{TC,sum(length.(pA))}
end

function TO.tensorcontract_type(TC, pC, ::SparseArray, pA, conjA,
                                ::SparseArray, pB, conjB)
    return SparseArray{TC,sum(length.(pC))}
end

end
