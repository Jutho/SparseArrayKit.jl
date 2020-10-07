using SparseArrayKit,Test,TestExtras,LinearAlgebra
import TensorOperations;

#=
generate a whole bunch of random contractions, compare with the dense result
=#
@timedtestset "random contractions" for (eltype,arraytype) in [(ComplexF64,SparseCOOArray),
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

            push!(tensors,arraytype(rand(eltype,cur_dims...)))
            push!(indices,cur_inds);
            push!(conjlist,rand([true,false]));
        end

        length(tensors) == 1 && continue # very rare - but possible

        sparse_result = TensorOperations.ncon(tensors,indices,conjlist);
        dense_result = TensorOperations.ncon(Array.(tensors),indices,conjlist);

        @test Array(sparse_result) ≈ dense_result
    end
end

@timedtestset "Basic linear algebra"  for (eltype,arraytype) in [(ComplexF64,SparseCOOArray),
                                                                (Float64,SparseDOKArray)]
    MAX_DIM = 5;
    MAX_LEGS = 5;

    dims = [rand(1:MAX_DIM) for l in 1:MAX_LEGS];

    arr = arraytype(rand(eltype,dims...));

    T = typeof(arr);

    @test isa(arr*2,T);
    @test isa(2*arr,T);
    @test isa(arr+arr,T);

    a = randn(eltype);
    @test norm(arr + a*arr) ≈ norm(arr)*abs(1+a);
    @test norm(arr + arr*a) ≈ norm(arr)*abs(1+a);
    @test arr*a ≈ a*arr;
    @test norm(a*arr) ≈ norm(arr)*abs(a);
    @test arr/a ≈ a\arr;
    @test norm(arr/a) ≈ norm(arr)/abs(a);
    @test norm(arr) ≈ sqrt(real(dot(arr,arr)))

    old_norm = norm(arr);
    lmul!(a,arr);
    @test old_norm*abs(a) ≈ norm(arr);
    ldiv!(a,arr);
    @test old_norm ≈ norm(arr);

    old_norm = norm(arr);
    a = randn(eltype);
    rmul!(arr,a);
    @test old_norm*abs(a) ≈ norm(arr);
    rdiv!(arr,a);
    @test old_norm ≈ norm(arr);

    brr = arraytype(rand(eltype,dims...));
    @test arr + brr ≈ brr + arr
    @test dot(arr,brr) ≈ dot(brr,arr)'

    b = randn(eltype);

    c = a*arr+brr
    axpy!(a,arr,brr)
    @test brr ≈ c

    c = a*arr+b*brr
    axpby!(a,arr,b,brr)
    @test brr ≈ c
end
