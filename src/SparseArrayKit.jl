module SparseArrayKit

using LinearAlgebra
using TupleTools

if !isdefined(Base, :get_extension)
    using Requires
end

const IndexTuple{N} = NTuple{N,Int}

export SparseArray
export nonzero_pairs, nonzero_keys, nonzero_values, nonzero_length

include("sparsearray.jl")
include("linearalgebra.jl")
include("tensoroperations.jl")

# Initialization
#-----------------
function __init__()
    @static if !isdefined(Base, :get_extension)
        @require TensorOperations="6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2" begin
            include("../ext/SparseArrayKitTensorOperations.jl")
        end
        
        @require SparseArrays="2f01184e-e22b-5df5-ae63-d93ebab69eaf" begin
            include("../ext/SparseArrayKitSparseArrays.jl")
        end
    end
end

end
