# TensorOperations compatiblity
#-------------------------------
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
                        increaseindex!(C, α * (CA == :C ? conj(vA) : vA) * (CB == :C ? conj(vB) : vB), IC)
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
                        increaseindex!(C, α * (CA == :C ? conj(vA) : vA) * (CB == :C ? conj(vB) : vB), IC)
                    end
                end
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

# function contract_CSC!(α, A::SparseArray, CA::Symbol, B::SparseArray, CB::Symbol,
#                     β, C::SparseArray,
#                     oindA::IndexTuple, cindA::IndexTuple,
#                     oindB::IndexTuple, cindB::IndexTuple,
#                     indCinoAB::IndexTuple, syms::Union{Nothing, NTuple{3,Symbol}} = nothing)
#
#     pA = (oindA...,cindA...)
#     (length(pA) == ndims(A) && TupleTools.isperm(pA)) ||
#         throw(IndexError("invalid permutation of length $(ndims(A)): $pA"))
#     pB = (oindB...,cindB...)
#     (length(pB) == ndims(B) && TupleTools.isperm(pB)) ||
#         throw(IndexError("invalid permutation of length $(ndims(B)): $pB"))
#     (length(oindA) + length(oindB) == ndims(C)) ||
#         throw(IndexError("non-matching output indices in contraction"))
#     (ndims(C) == length(indCinoAB) && isperm(indCinoAB)) ||
#         throw(IndexError("invalid permutation of length $(ndims(C)): $indCinoAB"))
#
#     sizeA = size(A)
#     sizeB = size(B)
#     sizeC = size(C)
#
#     csizeA = TupleTools.getindices(sizeA, cindA)
#     csizeB = TupleTools.getindices(sizeB, cindB)
#     osizeA = TupleTools.getindices(sizeA, oindA)
#     osizeB = TupleTools.getindices(sizeB, oindB)
#
#     csizeA == csizeB ||
#         throw(DimensionMismatch("non-matching sizes in contracted dimensions"))
#     TupleTools.getindices((osizeA..., osizeB...), indCinoAB) == size(C) ||
#         throw(DimensionMismatch("non-matching sizes in uncontracted dimensions"))
#
#     β == one(β) || (iszero(β) ? _zero!(C) : LinearAlgebra.lmul!(β, C))
#     if isempty(nonzero_pairs(A)) || isempty(nonzero_pairs(B))
#         return C
#     end
#
#     # Build CSC-like representation, using internal Dict structure
#     # perform multiplication based on Gustafson's algorithm
#     Anz = nonzero_pairs(A)
#     Bnz = nonzero_pairs(B)
#     sortbyA = i->_subind(i, Anz, cindA)
#     sortbyB = i->_subind(i, Bnz, oindB)
#
#     indexA = sort!(findall(!iszero, Anz.slots); by = sortbyA)
#     indexB = sort!(findall(!iszero, Bnz.slots); by = sortbyB)
#     rowvalsA = map(i->_subind(i, Anz, oindA), indexA)
#     rowvalsB = map(i->_subind(i, Bnz, cindB), indexB)
#     colvalsA, colptrA = _uniqueranges(sortbyA, indexA)
#     colmapA = Dict(colvalsA[i]=>i for i in 1:length(colvalsA))
#     colvalsB, colptrB = _uniqueranges(sortbyB, indexB)
#     nzvalsA = Anz.vals[indexA]
#     nzvalsB = Bnz.vals[indexB]
#
#     colptrC = similar(colptrB)
#     rowvalsC = similar(rowvalsA, 0)
#     nzvalsC = Vector{eltype(C)}()
#     localrowsC = Dict{eltype(rowvalsA), Int}()
#
#     offset = 0
#     counter = 0
#     @inbounds for jB = 1:length(colvalsB)
#         IBo = colvalsB[jB]
#         colptrC[jB] = offset+1
#         rowsB = colptrB[jB]:(colptrB[jB+1]-1)
#         for kB in rowsB
#             IBc = rowvalsB[kB]
#             vB = CB == :C ? conj(nzvalsB[kB]) : nzvalsB[kB]
#             kA = get(colmapA, IBc, 0)
#             kA == 0 && continue
#             rowsA = colptrA[kA]:(colptrA[kA+1]-1)
#             for iA in rowsA
#                 IAo = rowvalsA[iA]
#                 vA = CA == :C ? conj(nzvalsA[iA]) : nzvalsA[iA]
#                 iC = get!(localrowsC, IAo, counter+1)
#                 if iC <= counter
#                     nzvalsC[offset + iC] += vA*vB
#                 else
#                     push!(nzvalsC, vA*vB)
#                     push!(rowvalsC, IAo)
#                     counter += 1
#                 end
#             end
#         end
#         offset += counter
#         counter = 0
#         empty!(localrowsC)
#     end
#     colptrC[end] = offset+1
#
#     # transfer CSC data to C
#     _sizehint!(C, 2*length(nzvalsC))
#     @inbounds for j in 1:length(colvalsB)
#         IBo = colvalsB[j]
#         for i in colptrC[j]:(colptrC[j+1]-1)
#             IAo = rowvalsC[i]
#             vC = nzvalsC[i]
#             IABo = CartesianIndex(IAo, IBo)
#             IC = CartesianIndex(TupleTools.getindices(IABo.I, indCinoAB))
#             increaseindex!(C, α * vC, IC)
#         end
#     end
#     return C
# end
#
# @inbounds _subind(i, d::Dict{<:CartesianIndex}, ind::IndexTuple) =
#     CartesianIndex(TupleTools.getindices(d.keys[i].I, ind))
#
# # for a sorted vector a, return a list of unique elements and a vector whose i'th entry is
# # the starting index of the i'th unique element in a
# function _uniqueranges(f, a::AbstractVector{T}) where T
#     isempty(a) && error("does not work on empty array")
#     i = 1
#     startindex = [1]
#     uniquevals = [f(a[1])]
#     @inbounds for j in 2:length(a)
#         v = f(a[j])
#         v == uniquevals[i] && continue
#         push!(uniquevals, v)
#         push!(startindex, j)
#         i += 1
#     end
#     push!(startindex, length(a)+1)
#     return uniquevals, startindex
# end
