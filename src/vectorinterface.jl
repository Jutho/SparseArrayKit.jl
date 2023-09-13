# Vector interface implementation for SparseArray
##################################################################
# zerovector & zerovector!!
#---------------------------
function VectorInterface.zerovector(x::SparseArray, ::Type{S}) where {S<:Number}
    return SparseArray{S}(undef, size(x))
end
VectorInterface.zerovector!(x::SparseArray) = _zero!(x)
VectorInterface.zerovector!!(x::SparseArray) = zerovector!(x)

# scale, scale! & scale!!
#-------------------------
function VectorInterface.scale(x::SparseArray, α::Number)
    α === One() && return copy(x)
    α === Zero() && return zerovector(x)
    return x * α
end
function VectorInterface.scale!(x::SparseArray, α::Number)
    iszero(α) && return zerovector!(x)
    # typical occupation in a dict is about 30% from experimental testing
    # the benefits of scaling all values (e.g. SIMD) largely outweight the extra work
    scale!(x.data.vals, α)
    return x
end
function VectorInterface.scale!(y::SparseArray, x::SparseArray, α::Number)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    zerovector!(y)
    for (k, v) in nonzero_pairs(x)
        y[k] = scale!!(v, α)
    end
    return y
end
function VectorInterface.scale!!(x::SparseArray, α::Number)
    α === One() && return x
    if VectorInterface.promote_scale(x, α) <: scalartype(x)
        return scale!(x, α)
    else
        return scale(x, α)
    end
end
function VectorInterface.scale!!(y::SparseArray, x::SparseArray, α::Number)
    if VectorInterface.promote_scale(x, α) <: scalartype(y)
        return scale!(y, x, α)
    else
        return scale(x, α)
    end
end

# add, add! & add!!
#-------------------
function VectorInterface.add(y::SparseArray, x::SparseArray, α::Number, β::Number)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    T = VectorInterface.promote_add(y, x, α, β)
    z = SparseArray{T}(undef, size(y))
    scale!(z, y, β)
    add!(z, x, α)
    return z
end

function VectorInterface.add!(y::SparseArray, x::SparseArray, α::Number, β::Number)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    scale!(y, β)
    for (k, v) in nonzero_pairs(x)
        increaseindex!(y, scale!!(v, α), k)
    end
    return y
end

function VectorInterface.add!!(y::SparseArray, x::SparseArray, α::Number, β::Number)
    if VectorInterface.promote_add(y, x, α, β) <: scalartype(y)
        return add!(y, x, α, β)
    else
        return add(y, x, α, β)
    end
end

# inner
#-------
function VectorInterface.inner(x::SparseArray, y::SparseArray)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("dot arguments have non-matching axes $ax and $ay"))
    s = zero(VectorInterface.promote_inner(x, y))
    if nonzero_length(x) >= nonzero_length(y)
        @inbounds for I in nonzero_keys(x)
            s += VectorInterface.inner(x[I], y[I])
        end
    else
        @inbounds for I in nonzero_keys(y)
            s += VectorInterface.inner(x[I], y[I])
        end
    end
    return s
end
