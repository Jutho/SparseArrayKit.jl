module SparseArrayKit

using VectorInterface
using LinearAlgebra
using TupleTools

if !isdefined(Base, :get_extension)
    using Requires
end

const IndexTuple{N} = NTuple{N,Int}

export SparseArray
export nonzero_pairs, nonzero_keys, nonzero_values, nonzero_length

include("sparsearray.jl")
include("base.jl")
include("vectorinterface.jl")
include("tensoroperations.jl")
include("linearalgebra.jl")

# Initialization
#-----------------
using PackageExtensionCompat
function __init__()
    @require_extensions
end

end
