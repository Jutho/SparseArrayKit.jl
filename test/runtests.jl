println("Testing basic algebra from Julia Base and Julia LinearAlgebra")
println("=============================================================")
include("basic.jl")

println("Testing VectorInterface")
println("=======================")
include("vectorinterface.jl")

println("Testing Tensor Contraction")
println("==========================")
include("contractions.jl")

module AquaSparseArrayKit
using SparseArrayKit, Aqua, Test
@testset "Aqua" verbose = true begin
    Aqua.test_all(SparseArrayKit;
                  project_toml_formatting=(VERSION >= v"1.9"))
end
end
