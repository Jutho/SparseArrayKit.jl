# simple wrapper to give indices a custom wrapping behaviour
struct SparseArray{T,N} <: AbstractArray{T,N}
    data::Dict{CartesianIndex{N},T}
    dims::NTuple{N,Int64}
    function SparseArray{T,N}(::UndefInitializer, dims::Dims{N}) where {T,N}
        return new{T,N}(Dict{CartesianIndex{N},T}(), dims)
    end
    function SparseArray(a::SparseArray{T,N}) where {T,N}
        return new{T,N}(copy(a.data), a.dims)
    end
    function SparseArray{T,N}(a::Dict{CartesianIndex{N},T},
                              dims::NTuple{N,Int64}) where {T,N}
        return new{T,N}(a, dims)
    end
end
function SparseArray{T}(::UndefInitializer, dims::Dims{N}) where {T,N}
    return SparseArray{T,N}(undef, dims)
end
SparseArray{T}(::UndefInitializer, dims...) where {T} = SparseArray{T}(undef, dims)
function SparseArray{T}(a::UniformScaling, dims::Dims{2}) where {T}
    A = SparseArray{T}(undef, dims)
    for i in 1:min(dims...)
        A[i, i] = a[1, 1]
    end
    return A
end
SparseArray(a::UniformScaling, dims::Dims{2}) = SparseArray{eltype(a)}(a, dims)
SparseArray(a::UniformScaling, d1::Int, d2::Int) = SparseArray{eltype(a)}(a, (d1, d2))

nonzero_pairs(a::SparseArray) = pairs(a.data)
nonzero_keys(a::SparseArray) = keys(a.data)
nonzero_values(a::SparseArray) = values(a.data)
nonzero_length(a::SparseArray) = length(a.data)

_zero!(x::SparseArray) = (empty!(x.data); return x)
_sizehint!(x::SparseArray, n) = sizehint!(x.data, n)

# elementary getindex and setindex!
@inline function Base.getindex(a::SparseArray{T,N}, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(a, I)
    return get(a.data, I, zero(T))
end
Base.@propagate_inbounds function Base.getindex(a::SparseArray{T,N},
                                                I::Vararg{Int,N}) where {T,N}
    return getindex(a, CartesianIndex(I))
end

@inline function Base.setindex!(a::SparseArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(a, I)
    if v !== zero(v)
        a.data[I] = v
    else
        delete!(a.data, I) # does not do anything if there was no key corresponding to I
    end
    return v
end
Base.@propagate_inbounds function Base.setindex!(a::SparseArray{T,N},
                                                 v, I::Vararg{Int,N}) where {T,N}
    return setindex!(a, v, CartesianIndex(I))
end

@inline function increaseindex!(a::SparseArray{T,N}, v, I::CartesianIndex{N}) where {T,N}
    @boundscheck checkbounds(a, I)
    v == zero(v) && return
    h = a.data
    index = Base.ht_keyindex2!(h, I)
    @inbounds begin
        if index > 0
            currentv = h.vals[index]
            newv = currentv + convert(T, v)
            if v == zero(newv)
                Base._delete!(h, index)
            else
                h.age += 1
                h.keys[index] = I
                h.vals[index] = newv
            end
        else
            newv = convert(T, v)
            Base._setindex!(h, newv, I, -index)
        end
    end
    return newv
end

# following code is used to index with ranges etc
_newindex(i::Int, range::Int) = i == range ? () : nothing
function _newindex(i::Int, range::AbstractVector{Int})
    k = findfirst(==(i), range)
    return k === nothing ? nothing : (k,)
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
                               I::Vararg{Union{Int,AbstractVector{Int}},N}) where {T,N}
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
    return d
end

SparseArray(a::AbstractArray{T,N}) where {T,N} = SparseArray{T,N}(a)
SparseArray{T}(a::AbstractArray{<:Any,N}) where {T,N} = SparseArray{T,N}(a)
function SparseArray{T,N}(a::AbstractArray{<:Any,N}) where {T,N}
    d = SparseArray{T,N}(undef, size(a))
    for I in CartesianIndices(a)
        v == zero(a[I]) && continue
        d[I] = a[I]
    end
    return d
end
Base.convert(::Type{S}, a::S) where {S<:SparseArray} = a
Base.convert(S::Type{<:SparseArray}, a::AbstractArray) = S(a)

function SparseArray(A::Adjoint{T,<:SparseArray{T,2}}) where {T}
    B = SparseArray{T}(undef, size(A))
    for (I, v) in parent(A).data
        B[I[2], I[1]] = conj(v)
    end
    return B
end
function SparseArray(A::Transpose{T,<:SparseArray{T,2}}) where {T}
    B = SparseArray{T}(undef, size(A))
    for (I, v) in parent(A).data
        B[I[2], I[1]] = v
    end
    return B
end

Base.copy(a::SparseArray) = SparseArray(a)

Base.size(a::SparseArray) = a.dims

function Base.copy!(dst::SparseArray, src::SparseArray)
    axes(dst) == axes(src) ||
        throw(ArgumentError("arrays must have the same axes for copy! (consider using `copyto!`)"))

    _zero!(dst)
    for (I, v) in nonzero_pairs(src)
        dst[I] = v
    end
    return dst
end

function Base.similar(::SparseArray, ::Type{S}, dims::Dims{N}) where {S,N}
    return SparseArray{S}(undef, dims)
end

# show and friends
function Base.show(io::IO, ::MIME"text/plain", x::SparseArray)
    xnnz = nonzero_length(x)
    print(io, join(size(x), "×"), " ", typeof(x), " with ", xnnz, " stored ",
          xnnz == 1 ? "entry" : "entries")
    if xnnz != 0
        println(io, ":")
        show(IOContext(io, :typeinfo => eltype(x)), x)
    end
end
Base.show(io::IO, x::SparseArray) = show(convert(IOContext, io), x)
function Base.show(io::IOContext, x::SparseArray)
    nzind = nonzero_keys(x)
    if isempty(nzind)
        return show(io, MIME("text/plain"), x)
    end
    limit = get(io, :limit, false)::Bool
    half_screen_rows = limit ? div(displaysize(io)[1] - 8, 2) : typemax(Int)
    pads = map(1:ndims(x)) do i
        return ndigits(maximum(getindex.(nzind, i)))
    end
    if !haskey(io, :compact)
        io = IOContext(io, :compact => true)
    end
    for (k, (ind, val)) in enumerate(nonzero_pairs(x))
        if k < half_screen_rows || k > length(nzind) - half_screen_rows
            print(io, "  ", '[', join(lpad.(Tuple(ind), pads), ","), "]  =  ", val)
            k != length(nzind) && println(io)
        elseif k == half_screen_rows
            println(io, "   ", join(" " .^ pads, " "), "   \u22ee")
        end
    end
end
