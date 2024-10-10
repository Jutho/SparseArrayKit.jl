module SparseArrayKitTensorOperations

import TensorOperations as TO
using TensorOperations: Index2Tuple, linearize, numind
using SparseArrayKit: tensoradd!, tensortrace!, tensorcontract!, SparseArray

struct SparseArrayBackend <: TO.AbstractBackend
end

# ------------------------------------------------------------------------------------------
# Default backend selection mechanism for AbstractArray instances
# ------------------------------------------------------------------------------------------
function TO.select_backend(::typeof(TO.tensoradd!), C::SparseArray, A::SparseArray)
    return SparseArrayBackend()
end

function TO.select_backend(::typeof(TO.tensortrace!), C::SparseArray, A::SparseArray)
    return SparseArrayBackend()
end

function TO.select_backend(::typeof(TO.tensorcontract!), C::SparseArray, A::SparseArray,
                           B::SparseArray)
    return SparseArrayBackend()
end

function TO.tensoradd!(C::SparseArray,
                       A::SparseArray, pA::Index2Tuple, conjA::Bool,
                       α::Number, β::Number,
                       ::SparseArrayBackend, allocator)
    return tensoradd!(C, A, conjA, linearize(pA), α, β)
end

function TO.tensortrace!(C::SparseArray,
                         A::SparseArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
                         α::Number, β::Number,
                         ::SparseArrayBackend, allocator)
    return tensortrace!(C, A, conjA, linearize(p), q[1], q[2], α, β)
end

function TO.tensorcontract!(C::SparseArray,
                            A::SparseArray, pA::Index2Tuple, conjA::Bool,
                            B::SparseArray, pB::Index2Tuple, conjB::Bool,
                            pAB::Index2Tuple,
                            α::Number, β::Number,
                            ::SparseArrayBackend, allocator)
    return tensorcontract!(C, A, conjA, pA[1], pA[2], B, conjB, pB[2], pB[1],
                           linearize(pAB), α, β)
end

function TO.tensoradd_type(TC, ::SparseArray, pA::Index2Tuple, ::Bool)
    return SparseArray{TC,numind(pA)}
end

function TO.tensorcontract_type(TC,
                                ::SparseArray, pA::Index2Tuple, conjA::Bool,
                                ::SparseArray, pB::Index2Tuple, conjB::Bool,
                                pAB::Index2Tuple)
    return SparseArray{TC,numind(pAB)}
end

end
