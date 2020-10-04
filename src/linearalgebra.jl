# TODO: Basic arithmetic, comparison, ...

# Vector space functions
#------------------------
function LinearAlgebra.lmul!(a::Number, x::SparseArray)
    lmul!(a, x.data.vals)
    # typical occupation in a dict is about 30% from experimental testing
    # the benefits of scaling all values (e.g. SIMD) largely outweight the extra work
    return x
end
function LinearAlgebra.rmul!(x::SparseArray, a::Number)
    rmul!(x.data.vals, a)
    return x
end
function LinearAlgebra.axpby!(α, x::SparseArray, β, y::SparseArray)
    lmul!(y, β)
    for (k, v) in x.data
        y[k] += α*v
    end
    return y
end
function LinearAlgebra.axpy!(α, x::SparseArray, y::SparseArray)
    for (k, v) in x.data
        y[k] += α*v
    end
    return y
end

function LinearAlgebra.norm(x::SparseArray, p::Real = 2)
    norm(Base.Generator(last, x.data), p)
end

function LinearAlgebra.dot(x::SparseArray, y::SparseArray)
    size(x) == size(y) || throw(DimensionMismatch("dot arguments have different size"))
    s = dot(zero(eltype(x)), zero(eltype(y)))
    if length(x.data) >= length(y.data)
        iter = keys(x.data)
    else
        iter = keys(y.data)
    end
    @inbounds for I in iter
        s += dot(x[I], y[I])
    end
    return s
end
