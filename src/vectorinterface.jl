# Vector interface implementation for 'SparseArray
##################################################################
# zerovector & zerovector!!
#---------------------------
function VectorInterface.zerovector(x::SparseArray, ::Type{S}) where {S<:Number}
    T = typeof(zero(eltype(x)) * zero(S))
    return SparseArray{T}(undef, size(x))
end

VectorInterface.zerovector!(x::SparseArray) = _zero!(x)
VectorInterface.zerovector!!(x::SparseArray) = zerovector!(x)

# scale, scale! & scale!!
#-------------------------
VectorInterface.scale(x::SparseArray, α::ONumber) = α === _one ? copy(x) : x * α

function VectorInterface.scale!(x::SparseArray, α::ONumber)
    # typical occupation in a dict is about 30% from experimental testing
    # the benefits of scaling all values (e.g. SIMD) largely outweight the extra work
    scale!(x.data.vals, α)
    return x
end
function VectorInterface.scale!(y::SparseArray, x::SparseArray, α::ONumber)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    _zero!(y)
    for (k, v) in nonzero_pairs(x)
        y[k] = scale!!(v, α)
    end
    return y
end

function VectorInterface.scale!!(x::SparseArray, α::Number)
    T = scalartype(x)
    if promote_type(T, typeof(α)) <: T
        return scale!(x, α)
    else
        return scale(x, α)
    end
end
function VectorInterface.scale!!(y::SparseArray, x::SparseArray, α::Number)
    T = scalartype(y)
    if promote_type(T, typeof(α), scalartype(x)) <: T
        return scale!(y, x, α)
    else
        return scale(x, α)
    end
end

# add, add! & add!!
#-------------------
function VectorInterface.add(y::SparseArray, x::SparseArray, α::ONumber=_one,
                             β::ONumber=_one)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    T = promote_type(scalartype(y), scalartype(x), typeof(α), typeof(β))
    z = SparseArray{T}(undef, size(y))
    scale!(z, y, β)
    add!(z, x, α)
    return z
end

function VectorInterface.add!(y::SparseArray, x::SparseArray, α::ONumber=_one,
                              β::ONumber=_one)
    ax = axes(x)
    ay = axes(y)
    ax == ay || throw(DimensionMismatch("output axes $ay differ from input axes $ax"))
    _isone(β) || (iszero(β) ? _zero!(y) : scale!(y, β))
    for (k, v) in nonzero_pairs(x)
        increaseindex!(y, scale!!(v, α), k)
    end
    return y
end

function VectorInterface.add!!(y::SparseArray, x::SparseArray, α::ONumber=_one,
                               β::ONumber=_one)
    T = scalartype(y)
    if promote_type(T, typeof(α), typeof(β), scalartype(x)) <: T
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
    s = dot(zero(eltype(x)), zero(eltype(y)))
    if nonzero_length(x) >= nonzero_length(y)
        @inbounds for I in nonzero_keys(x)
            s += dot(x[I], y[I])
        end
    else
        @inbounds for I in nonzero_keys(y)
            s += dot(x[I], y[I])
        end
    end
    return s
end
