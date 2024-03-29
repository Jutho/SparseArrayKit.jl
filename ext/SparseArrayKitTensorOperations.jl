module SparseArrayKitTensorOperations

import TensorOperations as TO
using TensorOperations: Index2Tuple, linearize, numind
using SparseArrayKit: tensoradd!, tensortrace!, tensorcontract!, SparseArray

function TO.tensoradd!(C::SparseArray, pC::Index2Tuple,
                       A::SparseArray, conjA::Symbol,
                       α::Number, β::Number)
    return tensoradd!(C, linearize(pC), A, conjA, α, β)
end

function TO.tensortrace!(C::SparseArray, pC::Index2Tuple,
                         A::SparseArray, pA::Index2Tuple, conjA::Symbol,
                         α::Number, β::Number)
    return tensortrace!(C, linearize(pC), A, conjA, pA[1], pA[2], α, β)
end

function TO.tensorcontract!(C::SparseArray, pC::Index2Tuple,
                            A::SparseArray, pA::Index2Tuple, conjA::Symbol,
                            B::SparseArray, pB::Index2Tuple, conjB::Symbol,
                            α::Number, β::Number)
    return tensorcontract!(C, linearize(pC), A, conjA, pA[1], pA[2], B, conjB, pB[2], pB[1],
                           α, β)
end

function TO.tensoradd_type(TC, pA::Index2Tuple, ::SparseArray, ::Symbol)
    return SparseArray{TC,numind(pA)}
end

function TO.tensorcontract_type(TC, pC, ::SparseArray, pA, conjA,
                                ::SparseArray, pB, conjB)
    return SparseArray{TC,numind(pC)}
end

end
