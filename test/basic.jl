module BasicTests
using SparseArrayKit
using Test, TestExtras, LinearAlgebra, Random

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

@timedtestset "Basic linear algebra" begin
    MAX_DIM = 20
    MAX_LEGS = 4

    dims = ntuple(l -> rand(1:MAX_DIM), MAX_LEGS)
    ar = randn_sparse(Float64, dims)
    ac = randn_sparse(ComplexF64, dims)

    slice = (1:rand(1:dims[1]), rand(1:dims[2]), rand(1:dims[3]):dims[3], rand(1:dims[4]))
    @test ar[slice...] == Array(ar)[slice...]

    α = randn(ComplexF64)
    β = randn(Float64)
    γ = 2
    @test @constinferred(α * ar) == α * Array(ar)
    @test @constinferred(β * ar) == β * Array(ar)
    @test @constinferred(γ * ar) == γ * Array(ar)
    @test @constinferred(ac * β) == Array(ac) * β
    @test @constinferred(ar + ac) == Array(ar) + Array(ac)
    @test @constinferred(ac - ar) == Array(ac) - Array(ar)
    mar = @constinferred(-ar)
    @test iszero(ar + mar)
    ar2 = copy(ar)
    ar2[first(nonzero_keys(ar2))] = 0
    @test !(ar2 == ar)
    @test @constinferred(zero(ar)) + ac == ac
    @test ac == convert(SparseArray, Array(ac))

    @test norm(ar + @constinferred(α * ar)) ≈ norm(ar) * abs(1 + α)
    @test @constinferred(norm(ac + ac * α)) ≈ norm(ac) * abs(1 + α)
    @test ar * α ≈ α * ar
    @test norm(ar * α) ≈ norm(ar) * abs(α)
    @test ac / β ≈ β \ ac
    @test norm(ac / γ) ≈ norm(ac) / abs(γ)
    @test norm(ac) ≈ sqrt(real(@constinferred(dot(ac, ac))))
    @test @constinferred(dot(ac, ar)) ≈ dot(Array(ac), Array(ar))
    @test dot(ar, ac) ≈ conj(dot(ac, ar))

    @test @constinferred(lmul!(β, copy(ar))) == β * ar == β * Array(ar)
    @test @constinferred(rmul!(copy(ac), α)) == ac * α == Array(ac) * α
    @test @constinferred(rmul!(copy(ac), β)) == ac * β == Array(ac) * β
    @test @constinferred(ldiv!(β, copy(ar))) == ldiv!(β, Array(ar)) ≈ β \ ar
    @test @constinferred(rdiv!(copy(ac), α)) == rdiv!(Array(ac), α) ≈ ac / α
    @test @constinferred(rdiv!(copy(ac), β)) == rdiv!(Array(ac), β) ≈ ac / β
    @test @constinferred(conj!(copy(ac))) == conj!(Array(ac))
    @test_throws InexactError rmul!(copy(ar), α)
    @test_throws InexactError ldiv!(α, copy(ar))

    p = randperm(MAX_LEGS)
    @test permutedims(ar, p) == permutedims(Array(ar), p)

    @test @constinferred(axpy!(α, copy(ar), copy(ac))) == α * ar + ac
    @test @constinferred(axpby!(α, copy(ar), β, copy(ac))) == α * ar + β * ac
end

@timedtestset "Basic matrix algebra" begin
    using SparseArrays
    N = 100
    a = sprandn(ComplexF64, N, N, 0.1)
    b = sprandn(N, N, 0.1)
    aa = @constinferred(SparseArray(a))
    bb = SparseArray(b)
    @test aa == a
    @test @constinferred(aa * bb) ≈ SparseArray(a * b)
    @test aa * bb ≈ Array(aa) * Array(bb)
    @test adjoint(aa) == adjoint(Array(aa))
    @test transpose(aa) == transpose(Array(aa))
    @test aa' * bb ≈ Array(aa)' * Array(bb)
    @test aa' * bb' ≈ Array(aa)' * Array(bb)'
    @test aa * bb' ≈ Array(aa) * Array(bb)'
    @test @constinferred(one(aa)) == one(Array(aa))
    @test norm(one(aa)) ≈ sqrt(N)
    @test one(aa) + zero(aa) == one(bb)
    @test adjoint!(copy(aa), bb) == SparseArray(adjoint(bb)) == adjoint(Array(bb))
    @test transpose!(copy(aa), bb) == SparseArray(transpose(bb)) == transpose(Array(bb))
end

end
