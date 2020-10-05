struct SortedVectorDict{K, V} <: AbstractDict{K, V}
    keys::Vector{K}
    vals::Vector{V}
    function SortedVectorDict{K, V}(pairs::Vector{Pair{K, V}}) where {K, V}
        pairs = sort!(pairs, by=first)
        return new{K, V}(first.(pairs), last.(pairs))
    end
    function SortedVectorDict{K, V}(keys::Vector{K}, values::Vector{V}) where {K, V}
        @assert issorted(keys)
        new{K, V}(keys, values)
    end
    SortedVectorDict{K, V}() where {K, V} = new{K, V}(Vector{K}(undef, 0), Vector{V}(undef, 0))
end
SortedVectorDict{K, V}(kv::Pair{K, V}...) where {K, V} = SortedVectorDict{K, V}(kv)
function SortedVectorDict{K, V}(kv) where {K, V}
    d = SortedVectorDict{K, V}()
    if Base.IteratorSize(kv) !== SizeUnknown()
        sizehint!(d, length(kv))
    end
    for (k, v) in kv
        push!(d, k=>v)
    end
    return d
end
SortedVectorDict(pairs::Vector{Pair{K, V}}) where {K, V} = SortedVectorDict{K, V}(pairs)
@noinline function _no_pair_error()
    msg = "SortedVectorDict(kv): kv needs to be an iterator of pairs"
    throw(ArgumentError(msg))
end
function SortedVectorDict(pairs::Vector)
    all(p->isa(p, Pair), pairs) || _no_pair_error()
    pairs = sort!(pairs, by=first)
    keys = map(first, pairs)
    values = map(last, pairs)
    return SortedVectorDict{eltype(keys), eltype(values)}(keys, values)
end

SortedVectorDict(kv::Pair{K, V}...) where {K, V} = SortedVectorDict{K, V}(kv)

Base.@pure _getKV(::Type{Pair{K, V}}) where {K, V} = (K, V)
function SortedVectorDict(kv)
    if Base.IteratorEltype(kv) === Base.HasEltype()
        P = eltype(kv)
    elseif kv isa Base.Generator && kv.f isa Type
        P = kv.f
    else
        P = Base.Core.Compiler.return_type(first, Tuple{typeof(kv)})
    end
    if P <: Pair && Base.isconcretetype(P)
        K, V = _getKV(P)
        return SortedVectorDict{K, V}(kv)
    else
        return SortedVectorDict(collect(kv))
    end
end
SortedVectorDict() = SortedVectorDict{Any, Any}()

Base.length(d::SortedVectorDict) = length(d.keys)
Base.sizehint!(d::SortedVectorDict, newsz) =
    (sizehint!(d.keys, newsz); sizehint!(d.vals, newsz); return d)

Base.copy(d::SortedVectorDict{K, V}) where {K, V} =
    SortedVectorDict{K, V}(copy(d.keys), copy(d.vals))
Base.empty(::SortedVectorDict, ::Type{K}, ::Type{V}) where {K, V} = SortedVectorDict{K, V}()
Base.empty!(d::SortedVectorDict) = (empty!(d.keys); empty!(d.vals); return d)

# experiment with binary versus linear search
_searchsortedfirst(v::Vector, k) = searchsortedfirst(v, k)
# function _searchsortedfirst(v::Vector, k)
#     i = 1
#     @inbounds while i <= length(v) && isless(v[i], k)
#         i += 1
#     end
#     return i
# end

Base.keys(d::SortedVectorDict) = d.keys
Base.values(d::SortedVectorDict) = d.vals

function Base.haskey(d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return false
    end
    i = _searchsortedfirst(d.keys, key)
    return (i <= length(d) && isequal(d.keys[i], key))
end
function Base.getindex(d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        throw(KeyError(k))
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds if (i <= length(d) && isequal(d.keys[i], key))
        return d.vals[i]
    else
        throw(KeyError(key))
    end
end
function Base.setindex!(d::SortedVectorDict{K}, v, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        throw(ArgumentError("$k is not a valid key for type $K"))
    end
    i = _searchsortedfirst(d.keys, key)
    if i <= length(d) && isequal(d.keys[i], key)
        d.vals[i] = v
    else
        insert!(d.keys, i, key)
        insert!(d.vals, i, v)
    end
    return d
end
function Base.delete!(d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return d
    end
    i = _searchsortedfirst(d.keys, key)
    if i <= length(d) && isequal(d.keys[i], key)
        deleteat!(d.keys, i)
        deleteat!(d.vals, i)
    end
    return d
end

function Base.get(d::SortedVectorDict{K}, k, default) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return default
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds begin
        return (i <= length(d) && isequal(d.keys[i], key)) ? d.vals[i] : default
    end
end
function Base.get(default::Base.Callable, d::SortedVectorDict{K}, k) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        return f()
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds begin
        return (i <= length(d) && isequal(d.keys[i], key)) ? d.vals[i] : default()
    end
end

function Base.get!(d::SortedVectorDict{K}, k, default) where {K}
    key = convert(K, k)
    if !isequal(k, key)
        throw(ArgumentError("$k is not a valid key for type $K"))
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds begin
        if (i <= length(d) && isequal(d.keys[i], key))
            return d.vals[i]
        else
            insert!(d.keys, i, key)
            insert!(d.vals, i, default)
            return default
        end
    end
end
function Base.get!(default::Base.Callable, d::SortedVectorDict{K,V}, k) where {K,V}
    key = convert(K, k)
    if !isequal(k, key)
        throw(ArgumentError("$k is not a valid key for type $K"))
    end
    i = _searchsortedfirst(d.keys, key)
    @inbounds begin
        if (i <= length(d) && isequal(d.keys[i], key))
            return d.vals[i]
        else
            v = convert(V, default())
            insert!(d.keys, i, key)
            insert!(d.vals, i, v)
            return v
        end
    end
end

function Base.iterate(d::SortedVectorDict, i = 1)
    @inbounds if i > length(d)
        return nothing
    else
        return (d.keys[i] => d.vals[i]), i+1
    end
end
