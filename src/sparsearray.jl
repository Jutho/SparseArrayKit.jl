Base.@pure defaultdict(N::Int, T::Type) = Dict{CartesianIndex{N}, T}
# Base.@pure defaultdict(N::Int, T::Type) = SortedVectorDict{CartesianIndex{N}, T}

struct SparseArray{T,N,A<:AbstractDict{CartesianIndex{N},T}} <: AbstractArray{T,N}
    data::A
    dims::NTuple{N,Int64}
    function SparseArray{T,N,A}(::UndefInitializer, dims::Dims{N}) where {T,N,A}
        return new{T,N,A}(A(), dims)
    end
    function SparseArray(a::SparseArray{T,N}) where {T,N}
        data = copy(a.data)
        A = typeof(data)
        new{T,N,A}(data, a.dims)
    end
end
SparseArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    SparseArray{T,N,defaultdict(N,T)}(undef, dims)
SparseArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    SparseArray{T,N,defaultdict(N,T)}(undef, dims)
SparseArray{T}(::UndefInitializer, dims...) where {T} = SparseArray{T}(undef, dims)

const SparseDOKArray{T,N} = SparseArray{T,N,Dict{CartesianIndex{N},T}}
const SparseCOOArray{T,N} = SparseArray{T,N,SortedVectorDict{CartesianIndex{N},T}}

SparseDOKArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    SparseDOKArray{T,N}(undef, dims)
SparseCOOArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    SparseCOOArray{T,N}(undef, dims)

NonZeroIndices(a::SparseArray) = keys(a.data)
nonzeros(A::SparseArray) = values(A.data)

# elementary getindex and setindex!
@inline function Base.getindex(a::SparseArray{T,N}, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(a, I)
    return get(a.data, I, zero(T))
end
Base.@propagate_inbounds Base.getindex(a::SparseArray{T,N}, I::Vararg{Int,N}) where {T,N} =
                                        getindex(a, CartesianIndex(I))

@inline function Base.setindex!(a::SparseArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(a, I)
    if v != zero(v)
        a.data[I] = v
    else
        delete!(a.data, I) # does not do anything if there was no key I
    end
    return v
end
Base.@propagate_inbounds Base.setindex!(a::SparseArray{T,N},
                                        v, I::Vararg{Int,N}) where {T,N} =
                                            setindex!(a, v, CartesianIndex(I))

# following code is used to index with ranges etc
_newindex(i::Int, range::Int) = i == range ? () : nothing
function _newindex(i::Int, range::AbstractVector{Int})
    k = findfirst(==(i), range)
    k === nothing ? nothing : (k,)
end
_newindices(I::Tuple{}, indices::Tuple{}) = ()
function _newindices(I::Tuple, indices::Tuple)
    i = _newindex(I[1], indices[1])
    Itail = _newindices(Base.tail(I), Base.tail(indices))
    (i === nothing || Itail === nothing) && return nothing
    return (i..., Itail...)
end

_findfirstvalue(v, r) = findfirst(==(v), r)
# slicing should produce SparseArray
function Base._unsafe_getindex(::IndexCartesian, a::SparseArray{T,N},
                                I::Vararg{<:Union{Int,AbstractVector{Int}},N}) where {T,N}
    @boundscheck checkbounds(a, I...)
    indices = Base.to_indices(a, I)
    b = SparseArray{T}(undef, length.(Base.index_shape(indices...)))
    for (k, v) in a.data
        newI = _newindices(k.I, indices)
        if newI !== nothing
            b[newI...] = v
        end
    end
    return b
end

Base.Array(a::SparseArray{T,N}) where {T,N} = Array{T,N}(a)
function Base.Array{T,N}(a::SparseArray) where {T,N}
    d = fill(zero(T), size(a))
    for (I, v) in a.data
        d[I] = v
    end
    d
end

SparseArray(a::AbstractArray{T,N}) where {T,N} = SparseArray{T,N}(a)
SparseCOOArray(a::AbstractArray{T,N}) where {T,N} = SparseCOOArray{T,N}(a)
SparseDOKArray(a::AbstractArray{T,N}) where {T,N} = SparseDOKArray{T,N}(a)
function (::Type{S})(a::AbstractArray) where {S<:SparseArray}
    d = S(undef, size(a))
    for I in CartesianIndices(a)
        iszero(a[I]) && continue
        d[I] = a[I]
    end
    return d
end

function SparseArray(A::Adjoint{T,<:SparseArray{T,2}}) where T
    B = SparseArray{T}(undef, size(A))
    for (I, v) in parent(A).data
        B[I[2], I[1]] = v
    end
    return B
end

Base.copy(a::SparseArray) = SparseArray(a)

Base.size(a::SparseArray) = a.dims

Base.similar(a::SparseCOOArray, ::Type{S}, dims::Dims{N}) where {S,N} =
    SparseCOOArray{S}(undef, dims)
Base.similar(a::SparseDOKArray, ::Type{S}, dims::Dims{N}) where {S,N} =
    SparseDOKArray{S}(undef, dims)
Base.similar(a::SparseArray, ::Type{S}, dims::Dims{N}) where {S,N} =
    SparseArray{S}(undef, dims)
