module SparseArrayKit

using VectorInterface
using TensorOperations
using LinearAlgebra
using TupleTools

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
