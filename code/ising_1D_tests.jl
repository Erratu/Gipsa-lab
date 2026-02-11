using Test, Statistics, Printf

@testset "Validation Analytique Ising 1D" begin
    
    @testset "Énergie Interne : Chaîne Ouverte" begin
        # Pour une chaîne de N spins, l'énergie moyenne est connue exactement
        L = 20
        N = L
        beta = 0.6
        Q = adjacency_1d(N, periodic=false)
        m = zeros(N)
        
        # Théorie : -(N-1)/N * tanh(beta)
        e_theo = -((N-1)/N) * tanh(beta)
        
        # Simulation (Heat Bath)
        X = rand([-1,1], N)
        # Thermalisation
        heat_bath_ising!(X, Q, m, beta, 5000)
        
        # Mesure
        energies = Float64[]
        for _ in 1:10000
            heat_bath_ising!(X, Q, m, beta, 1)
            push!(energies, energy(X, Q, m) / N)
        end
        e_sim = mean(energies)
        
        println("\n--- 1D Chaîne Ouverte (N=$N, beta=$beta) ---")
        @printf("E/N théorique : %.5f\n", e_theo)
        @printf("E/N simulé    : %.5f\n", e_sim)
        @printf("Erreur        : %.2f%%\n", abs(e_sim - e_theo)/abs(e_theo)*100)
        
        @test abs(e_sim - e_theo) < 0.01
    end

    @testset "Énergie Interne : Cercle (Périodique)" begin
        # Pour un cercle, l'énergie est -tanh(beta) quand N est grand
        N = 30
        beta = 0.4
        Q = adjacency_1d(N, periodic=true)
        m = zeros(N)
        
        # Théorie pour le cercle : -tanh(beta) * [1 - (tanh(beta)^(N-2) * ...)] 
        # Pour N=30, -tanh(beta) est une excellente approximation
        e_theo = -tanh(beta)
        
        X = rand([-1,1], N)
        heat_bath_ising!(X, Q, m, beta, 5000)
        
        energies = Float64[]
        for _ in 1:2000
            heat_bath_ising!(X, Q, m, beta, N)
            push!(energies, energy(X, Q, m) / N)
        end
        e_sim = mean(energies)
        
        println("\n--- 1D Cercle Périodique (N=$N, beta=$beta) ---")
        @printf("E/N théorique : %.5f\n", e_theo)
        @printf("E/N simulé    : %.5f\n", e_sim)
        
        @test abs(e_sim - e_theo) < 0.01
    end

    @testset "Corrélations Spatiales (Cercle)" begin
        # On vérifie si la corrélation entre voisins suit [tanh(beta)]^r
        N = 20
        beta = 0.8
        Q = adjacency_1d(N, periodic=true)
        m = zeros(N)
        
        # Corrélation théorique voisin direct (r=1) : tanh(beta)
        corr_theo = tanh(beta)
        
        X = rand([-1,1], N)
        heat_bath_ising!(X, Q, m, beta, 10000)
        
        corrs = Float64[]
        for _ in 1:10000
            heat_bath_ising!(X, Q, m, beta, 1)
            # On mesure le produit de deux voisins s_i * s_{i+1}
            push!(corrs, X[1] * X[2])
        end
        corr_sim = mean(corrs)
        
        println("\n--- Corrélation voisin direct (r=1) ---")
        @printf("Théorie tanh(beta) : %.5f\n", corr_theo)
        @printf("Simulé <s1 s2>     : %.5f\n", corr_sim)
        
        @test abs(corr_sim - corr_theo) < 0.05
    end
end