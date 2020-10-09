module SparseArrayKit

using LinearAlgebra
using Requires

export SparseArray, SparseDOKArray, SparseCOOArray
export nonzero_pairs, nonzero_keys, nonzero_values, nonzero_length

include("sortedvectordict.jl")
include("sparsearray.jl")
include("linearalgebra.jl")

# Initialization
#-----------------
function __init__()
    @require TensorOperations="6aa20fa7-93e2-5fca-9bc0-fbd0db3c71a2" begin
        include("tensoroperations.jl")
    end
end

end
