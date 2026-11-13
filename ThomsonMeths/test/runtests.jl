using Test
using ThomsonMeths
using LinearAlgebra
using Random

@testset "ThomsonMeths.jl Test Suite" begin
    @testset "Energy and Gradient Primitives" begin
        m = 10
        X = random_configuration(m)
        E = thomson_energy(X)
        @test E > 0.0

        # Fast version consistency
        E_fast = thomson_energy_fast(X)
        @test isapprox(E, E_fast; atol=1e-8)

        # Gradient computation
        G = zeros(3, m)
        E_grad = thomson_energy_and_grad!(G, X)
        @test isapprox(E, E_grad; atol=1e-8)
        @test norm(G) > 0.0
    end

    @testset "Subset Selection Bases" begin
        include("test_sel_bases.jl")
    end
end
