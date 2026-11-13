using Test
using PTShared
using Random
using LinearAlgebra
using Statistics

@testset "PTShared.jl Test Suite" begin
    @testset "Data Structures" begin
        betas = [0.1, 0.5, 1.0, 2.0]
        sampler = PTSampler(betas)
        @test sampler.nb_sim == 4
        @test sampler.betas == betas

        init_states = [randn(5) for _ in 1:4]
        state = PTState{Vector{Float64}}(init_states)
        @test length(state.sims) == 4
        @test state.indices == [1, 2, 3, 4]
        @test state.energy_min == Inf
    end

    @testset "1D Gaussian Sampling with PT" begin
        # Target: standard normal distribution N(0, 1), E(x) = 0.5*x^2
        gauss_energy(x::Float64) = 0.5 * x^2
        gauss_obs(x::Float64) = x

        function gauss_local_update!(x::Float64, beta::Float64; step_size::Float64=1.5)
            x_prop = x + randn() * step_size
            dE = gauss_energy(x_prop) - gauss_energy(x)
            if rand() < exp(-beta * dE)
                return x_prop
            end
            return x
        end

        Random.seed!(1234)
        betas = [0.1, 0.3, 0.7, 1.0]
        sampler = PTSampler(betas)
        state = PTState{Float64}([randn() for _ in 1:length(betas)])

        algo_pt_complet!(sampler, state, gauss_energy, gauss_local_update!, gauss_obs, 1000, 5000)

        cold_samples = state.history.obs_hist[end]
        @test length(cold_samples) == 5000
        @test isapprox(mean(cold_samples), 0.0; atol=0.15)
        @test isapprox(var(cold_samples), 1.0; atol=0.25)
    end

    @testset "Non-Reversible PT (NRPT) Deterministic Even-Odd Swaps" begin
        gauss_energy(x::Float64) = 0.5 * x^2
        gauss_obs(x::Float64) = x

        function gauss_local_update!(x::Float64, beta::Float64; step_size::Float64=1.5)
            x_prop = x + randn() * step_size
            dE = gauss_energy(x_prop) - gauss_energy(x)
            if rand() < exp(-beta * dE)
                return x_prop
            end
            return x
        end

        Random.seed!(4321)
        betas = [0.1, 0.3, 0.7, 1.0]
        sampler = PTSampler(betas)
        state = PTState{Float64}([randn() for _ in 1:length(betas)])

        algo4_nrpt_complet!(sampler, state, gauss_energy, gauss_local_update!, gauss_obs, 1000, 5000)

        cold_samples = state.history.obs_hist[end]
        @test length(cold_samples) == 5000
        @test isapprox(mean(cold_samples), 0.0; atol=0.15)
        @test isapprox(var(cold_samples), 1.0; atol=0.25)
    end
end
