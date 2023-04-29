module SparseArrayKitSparseArrays

@static if isdefined(Base, :get_extension)
    using SparseArrays: SparseMatrixCSC, nonzeros, rowvals, nzrange
else
    using ..SparseArrays: SparseMatrixCSC, nonzeros, rowvals, nzrange
end

using SparseArrayKit

Base.convert(T::Type{<:SparseArray}, a::SparseMatrixCSC) = T(a)

SparseArrayKit.SparseArray(a::SparseMatrixCSC) = SparseArray{eltype(a)}(a)
function SparseArrayKit.SparseArray{T}(a::SparseMatrixCSC) where {T}
    b = SparseArray{T}(undef, size(a))
    rv = rowvals(a)
    nzv = nonzeros(a)
    @inbounds for col in 1:size(a, 2)
        for i in nzrange(a, col)
            row = rv[i]
            v = nzv[i]
            b[row, col] = v
        end
    end
    return b
end

end
