using SparseArrayKit
using Test, TestExtras, LinearAlgebra, Random
import TensorOperations

#=
generate a whole bunch of random contractions, compare with the dense result
=#
function randn_sparse(S::Type{<:SparseArray}, sz::Dims, p = 0.5)
    a = S(undef, sz)
    T = eltype(a)
    for I in keys(a)
        if rand() < p
            a[I] = randn(T)
        end
    end
    return a
end



@timedtestset "Basic linear algebra"  for arraytype in (SparseCOOArray, SparseDOKArray)
    MAX_DIM = 20;
    MAX_LEGS = 4;

    dims = ntuple(l->rand(1:MAX_DIM), MAX_LEGS)
    ar = randn_sparse(arraytype{Float64}, dims)
    ac = randn_sparse(arraytype{ComplexF64}, dims)

    α = randn(ComplexF64)
    β = randn(Float64)
    γ = 2
    @test @constinferred(α*ar) == α*Array(ar)
    @test @constinferred(β*ar) == β*Array(ar)
    @test @constinferred(γ*ar) == γ*Array(ar)
    @test @constinferred(ac*β) == Array(ac)*β
    @test @constinferred(ar + ac) == Array(ar) + Array(ac)

    @test norm(ar + @constinferred(α*ar)) ≈ norm(ar)*abs(1+α)
    @test @constinferred(norm(ac + ac*α)) ≈ norm(ac)*abs(1+α)
    @test ar*α ≈ α*ar
    @test norm(ar*α) ≈ norm(ar)*abs(α);
    @test ac/β ≈ β\ac;
    @test norm(ac/γ) ≈ norm(ac)/abs(γ);
    @test norm(ac) ≈ sqrt(real(@constinferred(dot(ac, ac))))

    @test @constinferred(lmul!(β, copy(ar))) == β * ar == β * Array(ar)
    @test @constinferred(rmul!(copy(ac), α)) == ac * α == Array(ac) * α
    @test @constinferred(rmul!(copy(ac), β)) == ac * β == Array(ac) * β
    @test_throws InexactError rmul!(copy(ar), α)

    p = randperm(MAX_LEGS)
    @test permutedims(ar, p) == permutedims(Array(ar), p)

    @test @constinferred(axpy!(α, copy(ar), copy(ac))) == α * ar + ac
    @test @constinferred(axpby!(α, copy(ar), β, copy(ac))) == α * ar + β * ac
end

@timedtestset "random contractions" for (eltype,arraytype) in Any[(ComplexF64,SparseCOOArray),
                                                                (Float64,SparseDOKArray)]

    MAX_CONTRACTED_INDICES = 10;
    MAX_OPEN_INDICES = 5;
    MAX_DIM = 5;
    MAX_IND_PER_TENS = 3;
    NUM_TESTS = 100

    for i in 1:NUM_TESTS
        contracted_indices = repeat(collect(1:rand(1:MAX_CONTRACTED_INDICES)),2);
        open_indices = collect(1:rand(1:MAX_OPEN_INDICES));
        dimensions = [repeat(rand(1:MAX_DIM,Int(length(contracted_indices)/2)),2);rand(1:MAX_DIM,length(open_indices))]

        #generate a random tensor network contraction
        tensors = SparseArray[];
        indices = Vector{Int64}[];
        conjlist = Bool[];
        while !isempty(contracted_indices) || !isempty(open_indices)
            num_inds = rand(1:min(MAX_IND_PER_TENS,length(contracted_indices)+length(open_indices)));

            cur_inds = Int64[];
            cur_dims = Int64[];

            for ind in 1:num_inds
                curind_index = rand(1:(length(contracted_indices)+length(open_indices)));

                if curind_index <= length(contracted_indices)
                    push!(cur_inds,contracted_indices[curind_index]);
                    push!(cur_dims,dimensions[curind_index]);
                    deleteat!(contracted_indices,curind_index);
                    deleteat!(dimensions,curind_index);
                else
                    tind = curind_index - length(contracted_indices)
                    push!(cur_inds,-open_indices[tind]);
                    push!(cur_dims,dimensions[curind_index]);
                    deleteat!(open_indices,tind);
                    deleteat!(dimensions,curind_index);
                end
            end

            push!(tensors, randn_sparse(arraytype{eltype}, tuple(cur_dims...)))
            push!(indices, cur_inds);
            push!(conjlist, rand([true,false]));
        end

        length(tensors) == 1 && continue # very rare - but possible

        sparse_result = TensorOperations.ncon(tensors,indices,conjlist);
        dense_result = TensorOperations.ncon(Array.(tensors),indices,conjlist);

        @test Array(sparse_result) ≈ dense_result
    end
end

#
#
# @test old_norm*abs(a) ≈ norm(arr);
#     ldiv!(a,arr);
#     @test old_norm ≈ norm(arr);
#
#     old_norm = norm(arr);
#     a = randn(eltype);
#     rmul!(arr,a);
#     @test old_norm*abs(a) ≈ norm(arr);
#     rdiv!(arr,a);
#     @test old_norm ≈ norm(arr);
#
#     brr = arraytype(rand(eltype,dims...));
#     @test arr + brr ≈ brr + arr
#     @test dot(arr,brr) ≈ dot(brr,arr)'
#
#     b = randn(eltype);
#
#     c = a*arr+brr
#     axpy!(a,arr,brr)
#     @test brr ≈ c
#
#     c = a*arr+b*brr
#     axpby!(a,arr,b,brr)
#     @test brr ≈ c
# end
