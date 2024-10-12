module ContractionTests
using SparseArrayKit
using Test, TestExtras, LinearAlgebra, Random
using TensorOperations
#=
generate a whole bunch of random contractions, compare with the dense result
=#
function randn_sparse(T::Type{<:Number}, sz::Dims, p=0.5)
    a = SparseArray{T}(undef, sz)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end

@timedtestset "random contractions with eltype = $T" for T in (Float64, ComplexF64)
    MAX_CONTRACTED_INDICES = 10
    MAX_OPEN_INDICES = 5
    MAX_DIM = 5
    MAX_IND_PER_TENS = 3
    NUM_TESTS = 100

    for i in 1:NUM_TESTS
        contracted_indices = repeat(collect(1:rand(1:MAX_CONTRACTED_INDICES)), 2)
        open_indices = collect(1:rand(1:MAX_OPEN_INDICES))
        dimensions = [repeat(rand(1:MAX_DIM, Int(length(contracted_indices) / 2)), 2);
                      rand(1:MAX_DIM, length(open_indices))]

        #generate a random tensor network contraction
        tensors = SparseArray[]
        indices = Vector{Int64}[]
        conjlist = Bool[]
        while !isempty(contracted_indices) || !isempty(open_indices)
            num_inds = rand(1:min(MAX_IND_PER_TENS,
                                  length(contracted_indices) + length(open_indices)))

            cur_inds = Int64[]
            cur_dims = Int64[]

            for ind in 1:num_inds
                curind_index = rand(1:(length(contracted_indices) + length(open_indices)))

                if curind_index <= length(contracted_indices)
                    push!(cur_inds, contracted_indices[curind_index])
                    push!(cur_dims, dimensions[curind_index])
                    deleteat!(contracted_indices, curind_index)
                    deleteat!(dimensions, curind_index)
                else
                    tind = curind_index - length(contracted_indices)
                    push!(cur_inds, -open_indices[tind])
                    push!(cur_dims, dimensions[curind_index])
                    deleteat!(open_indices, tind)
                    deleteat!(dimensions, curind_index)
                end
            end

            push!(tensors, randn_sparse(T, tuple(cur_dims...)))
            push!(indices, cur_inds)
            push!(conjlist, rand([true, false]))
        end

        sparse_result = TensorOperations.ncon(tensors, indices, conjlist)
        # only check nonzero results via dense tensor contractions to reduce test time
        if SparseArrayKit.nonzero_length(sparse_result) > 0
            dense_result = TensorOperations.ncon(Array.(tensors), indices, conjlist)
            @test Array(sparse_result) ≈ dense_result
        end
    end
end

end
