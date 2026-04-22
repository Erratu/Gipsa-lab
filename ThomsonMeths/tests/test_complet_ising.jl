using Test, Statistics, Printf, LinearAlgebra
using IsingSimu

@testset "Suite Complète de Tests Ising avec PT et NRPT" begin

    @testset "Validation Analytique Ising 1D" begin
        @testset "Énergie Interne : Chaîne Ouverte" begin
            L = 20
            N = L
            beta = 0.6
            Q = adjacency_1d(N, periodic=false)
            m = zeros(N)
            e_theo = -((N-1)/N) * tanh(beta)
            
            betas = [0.1, 0.3, beta]
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, N); return X end
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}([rand([-1,1], N) for _ in 1:3])
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 2000, 10000)
            e_sim_pt = mean(state_pt.history.ene_hist[end]) / N
            @test abs(e_sim_pt - e_theo) < 0.02
            
            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}([rand([-1,1], N) for _ in 1:3])
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 2000, 10000)
            e_sim_nrpt = mean(state_nrpt.history.ene_hist[end]) / N
            @test abs(e_sim_nrpt - e_theo) < 0.02

            @info "1D Chaîne Ouverte (N=$N, beta=$beta)"
            @info @sprintf("E/N théorique : %.5f", e_theo)
            @info @sprintf("E/N simulé PT : %.5f", e_sim_pt)
            @info @sprintf("Erreur PT     : %.2f%%", abs(e_sim_pt - e_theo)/abs(e_theo)*100)
        end

        @testset "Énergie Interne : Cercle (Périodique)" begin
            N = 30
            beta = 0.4
            Q = adjacency_1d(N, periodic=true)
            m = zeros(N)
            e_theo = -tanh(beta)
            
            betas = [0.1, 0.25, beta]
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, N); return X end
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}([rand([-1,1], N) for _ in 1:3])
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 2000, 10000)
            e_sim_pt = mean(state_pt.history.ene_hist[end]) / N
            @test abs(e_sim_pt - e_theo) < 0.02
            
            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}([rand([-1,1], N) for _ in 1:3])
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 2000, 10000)
            e_sim_nrpt = mean(state_nrpt.history.ene_hist[end]) / N
            @test abs(e_sim_nrpt - e_theo) < 0.02

            @info "1D Cercle Périodique (N=$N, beta=$beta)"
            @info @sprintf("E/N théorique : %.5f", e_theo)
            @info @sprintf("E/N simulé PT : %.5f", e_sim_pt)
        end

        @testset "Corrélations Spatiales (Cercle)" begin
            N = 20
            beta = 0.8
            Q = adjacency_1d(N, periodic=true)
            m = zeros(N)
            corr_theo = tanh(beta)
            
            betas = [0.1, 0.4, beta]
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, N); return X end
            obs_func = X -> Float64(X[1] * X[2])
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}([rand([-1,1], N) for _ in 1:3])
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, obs_func, 2000, 10000)
            corr_sim_pt = mean(state_pt.history.obs_hist[end])
            @test abs(corr_sim_pt - corr_theo) < 0.05
            
            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}([rand([-1,1], N) for _ in 1:3])
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, obs_func, 2000, 10000)
            corr_sim_nrpt = mean(state_nrpt.history.obs_hist[end])
            @test abs(corr_sim_nrpt - corr_theo) < 0.05

            @info "Corrélation voisin direct (r=1)"
            @info @sprintf("Théorie tanh(beta) : %.5f", corr_theo)
            @info @sprintf("Simulé PT <s1 s2>  : %.5f", corr_sim_pt)
        end
    end

    @testset "Loi de Probabilité Exacte N=4" begin
        n = 4
        beta = 0.5
        Q = zeros(n,n); for i=1:n-1 Q[i,i+1]=Q[i+1,i]=1 end
        m = zeros(n)
        
        states = [ [(i >> j) & 1 == 1 ? 1 : -1 for j in 0:n-1] for i in 0:(2^n-1) ]
        energies = [energy(s, Q, m) for s in states]
        Z = sum(exp.(-beta .* energies))
        probs_theo = exp.(-beta .* energies) ./ Z

        betas = [0.1, 0.3, beta]
        state_to_idx(X) = Float64(sum((X .+ 1) .÷ 2 .* (2 .^ (0:n-1))) + 1)
        e_func = X -> energy(X, Q, m)
        u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n); return X end

        sampler_pt = PTSampler(copy(betas))
        state_pt = PTState{Vector{Int}}([rand([-1,1], n) for _ in 1:3])
        algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, state_to_idx, 2000, 20000)
        
        counts_pt = zeros(2^n)
        for val in state_pt.history.obs_hist[end] counts_pt[Int(val)] += 1 end
        diff_max_pt = maximum(abs.( (counts_pt ./ sum(counts_pt)) .- probs_theo))
        @test diff_max_pt < 0.05

        sampler_nrpt = PTSampler(copy(betas))
        state_nrpt = PTState{Vector{Int}}([rand([-1,1], n) for _ in 1:3])
        algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, state_to_idx, 2000, 20000)
        
        counts_nrpt = zeros(2^n)
        for val in state_nrpt.history.obs_hist[end] counts_nrpt[Int(val)] += 1 end
        diff_max_nrpt = maximum(abs.( (counts_nrpt ./ sum(counts_nrpt)) .- probs_theo))
        @test diff_max_nrpt < 0.05

        @info @sprintf("[Loi de Proba N=4] Différence max PT : %.4f (Seuil : 0.05)", diff_max_pt)
    end

    @testset "Conditions aux Limites et Alignement (1D et 2D)" begin
        @testset "1D Dirichlet (Bords fixes)" begin
            n = 10
            Q = adjacency_1d(n, periodic=false)
            m = zeros(n)
            fixed = [1, n]
            betas = [0.5, 5.0]
            
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n, fixed_indices=fixed); return X end
            
            init_states = [rand([-1,1], n) for _ in 1:2]
            for s in init_states s[1], s[n] = 1, 1 end
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}(copy.(init_states))
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 500, 1000)
            err_align_pt = n - abs(sum(state_pt.sims[end]))
            @test all(state_pt.sims[end] .== 1)
            
            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}(copy.(init_states))
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 500, 1000)
            @test all(state_nrpt.sims[end] .== 1)

            @info @sprintf("[1D Dirichlet] Erreur d'alignement PT : %d (Attendu : 0)", err_align_pt)
        end

        @testset "Tests 2D (Basse T)" begin
            @info "[Tests 2D Basse T]"
            for L in [3, 4, 5]
                n = L*L
                Q = adjacency_2d(L, periodic=true)
                m = zeros(n)
                betas = [0.5, 3.0]
                
                e_func = X -> energy(X, Q, m)
                u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n); return X end
                
                sampler_pt = PTSampler(copy(betas))
                state_pt = PTState{Vector{Int}}([rand([-1,1], n) for _ in 1:2])
                algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 500, 500)
                res_2d_pt = abs(sum(state_pt.sims[end]))
                @test res_2d_pt == n
                
                sampler_nrpt = PTSampler(copy(betas))
                state_nrpt = PTState{Vector{Int}}([rand([-1,1], n) for _ in 1:2])
                algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 500, 500)
                res_2d_nrpt = abs(sum(state_nrpt.sims[end]))
                @test res_2d_nrpt == n

                @info @sprintf(" -> L=%d : Obtenu %d / Attendu %d", L, res_2d_pt, n)
            end
        end
    end

    @testset "Transition de Phase (2D) avec PT et NRPT" begin
        L = 10
        n = L*L
        Q = adjacency_2d(L, periodic=true)
        m = zeros(n)
        
        betas = collect(range(0.1, 1.5, length=8))
        initial_states = [rand([-1,1], n) for _ in 1:8]
        
        e_func = X -> energy(X, Q, m)
        u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n); return X end
        obs_func = X -> abs(sum(X)) / n
        
        sampler_pt = PTSampler(copy(betas))
        state_pt = PTState{Vector{Int}}(copy.(initial_states))
        algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, obs_func, 300, 800)
        
        mag_froid_pt = mean(state_pt.history.obs_hist[end])
        mag_chaud_pt = mean(state_pt.history.obs_hist[1])
        
        @test mag_froid_pt > 0.8
        @test mag_chaud_pt < 0.3

        sampler_nrpt = PTSampler(copy(betas))
        state_nrpt = PTState{Vector{Int}}(copy.(initial_states))
        algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, obs_func, 300, 800)
        
        mag_froid_nrpt = mean(state_nrpt.history.obs_hist[end])
        mag_chaud_nrpt = mean(state_nrpt.history.obs_hist[1])
        
        @test mag_froid_nrpt > 0.8
        @test mag_chaud_nrpt < 0.3

        @info @sprintf("[Transition 2D PT] |M| Froid : %.3f (Seuil > 0.8) | |M| Chaud : %.3f (Seuil < 0.3)", mag_froid_pt, mag_chaud_pt)
    end

    @testset "Validation des Comportements PT et NRPT" begin
        @testset "Champ Fort (m > 0 -> X = 1)" begin
            n = 10
            Q = zeros(n, n)
            m = fill(20.0, n)
            
            betas = collect(range(0.1, 1.0, length=5))
            initial_states = [rand([-1,1], n) for _ in 1:5]
            
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n); return X end
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}(copy.(initial_states))
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 50, 150)
            @test all(state_pt.sims[end] .== 1)

            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}(copy.(initial_states))
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 50, 150)
            @test all(state_nrpt.sims[end] .== 1)

            @info @sprintf("Champ Fort : OK (Somme spins = %d / %d)", sum(state_pt.sims[end]), n)
        end

        @testset "Anti-ferro (Q=-1 -> Spins opposés)" begin
            n = 2
            Q = [0.0 -1.0; -1.0 0.0]
            m = zeros(n)
            
            betas = collect(range(0.1, 2.0, length=5))
            initial_states = [rand([-1,1], n) for _ in 1:5]
            
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n); return X end
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}(copy.(initial_states))
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 50, 150)
            @test state_pt.sims[end][1] != state_pt.sims[end][2]
            @test energy(state_pt.sims[end], Q, m) == -1.0

            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}(copy.(initial_states))
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 50, 150)
            @test state_nrpt.sims[end][1] != state_nrpt.sims[end][2]
            @test energy(state_nrpt.sims[end], Q, m) == -1.0

            @info "Anti-ferro : OK (X[1]=$(state_pt.sims[end][1]), X[2]=$(state_pt.sims[end][2]))"
        end

        @testset "Optimisation Globale" begin
            n = 15
            Q = randn(n, n)
            Q = (Q + Q') / 2 
            for i in 1:n Q[i,i] = 0 end
            m = randn(n)
            
            X_rand = rand([-1, 1], n)
            E_rand = energy(X_rand, Q, m)
            
            betas = collect(range(0.1, 3.0, length=8))
            initial_states = [rand([-1,1], n) for _ in 1:8]
            
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n); return X end
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}(copy.(initial_states))
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 100, 400)
            E_opt_pt = energy(state_pt.sims[end], Q, m)
            @test E_opt_pt <= E_rand

            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}(copy.(initial_states))
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 100, 400)
            @test energy(state_nrpt.sims[end], Q, m) <= E_rand

            @info @sprintf("Optimisation : OK (E_init: %.2f -> E_opt: %.2f | Delta: %.2f)", E_rand, E_opt_pt, E_rand - E_opt_pt)
        end

        @testset "Mixage des Répliques" begin
            n = 16
            L = 4
            Q = adjacency_2d(L, periodic=true)
            m = zeros(n)
            nb_sim = 6
            nb_it = 1000
            
            betas = collect(range(0.1, 1.5, length=nb_sim))
            initial_states = [rand([-1,1], n) for _ in 1:nb_sim]
            
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, n); return X end
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}(copy.(initial_states))
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, X->1.0, 200, nb_it)
            @test state_pt.metrics.round_trips > 0

            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}(copy.(initial_states))
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, X->1.0, 200, nb_it)
            @test state_nrpt.metrics.round_trips > 0

            @info "Mixage : OK (Structure validée, RT: $(state_pt.metrics.round_trips))"
        end
    end

    @testset "Benchmarking et Validation Statistique" begin
        L = 6
        N = L * L
        beta = 0.5
        n_essais = 3
        n_steps = 2000
        Q = adjacency_2d(L, periodic=true)
        m = zeros(N)

        E_pt = []
        E_nrpt = []

        for essai in 1:n_essais
            X_init = rand([-1, 1], N)
            nb_sim = 8
            betas = collect(range(0.1, beta, length=nb_sim))
            initial_states = [copy(X_init) for _ in 1:nb_sim]
            
            e_func = X -> energy(X, Q, m)
            u_func = (X, b) -> begin heat_bath_ising!(X, Q, m, b, N); return X end
            obs_func = X -> abs(sum(X))/N
            
            sampler_pt = PTSampler(copy(betas))
            state_pt = PTState{Vector{Int}}(copy.(initial_states))
            algo_pt_complet!(sampler_pt, state_pt, e_func, u_func, obs_func, div(n_steps, 2), div(n_steps, 2))
            push!(E_pt, mean(state_pt.history.ene_hist[end]) / N)

            sampler_nrpt = PTSampler(copy(betas))
            state_nrpt = PTState{Vector{Int}}(copy.(initial_states))
            algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func, obs_func, div(n_steps, 2), div(n_steps, 2))
            push!(E_nrpt, mean(state_nrpt.history.ene_hist[end]) / N)
        end

        @info @sprintf("Énergie moyenne par spin sur %d essais :", n_essais)
        @info @sprintf("Cible       : -1.7473")
        @info @sprintf("PT          : %.4f", mean(E_pt))
        @info @sprintf("NRPT        : %.4f", mean(E_nrpt))
        
        @test abs(mean(E_pt) - (-1.7473)) < 0.1
        @test abs(mean(E_nrpt) - (-1.7473)) < 0.1
    end
end;