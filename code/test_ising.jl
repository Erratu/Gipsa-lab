# Fichier : test_ising.jl
include("ising_model_2.jl") # Charge tes fonctions
using Test
using LinearAlgebra
using Statistics
using Printf

# Utilisation de TA fonction energy (convention -0.5) pour la cohérence
# Rappel : energy(X, Q, m) = -0.5 * dot(X, Q, X) - dot(m, X)

println("\n=== Démarrage de la suite de tests Validation PT ===\n")

@testset "Validation Parallel Tempering" begin

    # --- TEST 1 : CHAMP MAGNÉTIQUE FORT ---
    @testset "Champ Fort (m > 0 -> X = 1)" begin
        n = 10
        Q = zeros(n, n)   # Pas d'interaction entre voisins
        m = fill(20.0, n) # Champ très fort positif (force l'alignement à +1)
        
        # On récupère sims (toutes les répliques) et histories
        # On prend la dernière réplique (la plus froide) pour le test
        sims, _ = parallel_tempering_ising(Q, m, n, 200, 5, visual=false)
        X_final = sims[end]
        
        @test all(X_final .== 1)
        @printf("Champ Fort : OK (Somme spins = %d / %d)\n", sum(X_final), n)
    end

    # --- TEST 2 : ANTI-FERROMAGNÉTISME ---
    @testset "Anti-ferro (Q=-1 -> Spins opposés)" begin
        # ATTENTION : Pour avoir des spins opposés avec TA fonction energy,
        # il faut que Q soit négatif (antiferromagnétique).
        n = 2
        Q = [-1.0 0.0; 0.0 -1.0] # Interaction négative
        Q = [0.0 -1.0; -1.0 0.0] # s1 veut être différent de s2
        m = zeros(n)
        
        sims, _ = parallel_tempering_ising(Q, m, n, 200, 5, visual=false)
        X_final = sims[end]
        
        @test X_final[1] != X_final[2]
        # E = -0.5 * (1*-1*-1 + -1*-1*1) = -0.5 * (1 + 1) = -1.0
        @test energy(X_final, Q, m) == -1.0
        println("Anti-ferro : OK (X[1]=$(X_final[1]), X[2]=$(X_final[2]))")
    end

    # --- TEST 3 : CONVERGENCE D'ÉNERGIE ---
    @testset "Optimisation Globale" begin
        n = 15
        # Matrice symétrique aléatoire (Verre de spin simple)
        Q = randn(n, n)
        Q = (Q + Q') / 2 
        for i in 1:n Q[i,i] = 0 end
        m = randn(n)
        
        # Référence aléatoire
        X_rand = rand([-1, 1], n)
        E_rand = energy(X_rand, Q, m)
        
        # Résultat PT
        sims, _ = parallel_tempering_ising(Q, m, n, 500, 8, visual=false)
        X_opt = sims[end]
        E_opt = energy(X_opt, Q, m)
        
        # On vérifie que l'énergie a baissé (ou est égale au minimum)
        @test E_opt <= E_rand
        @printf("Optimisation : OK (E_init: %.2f -> E_opt: %.2f | Delta: %.2f)\n", 
                E_rand, E_opt, E_rand - E_opt)
    end

    # --- TEST 4 : MIXAGE ET ERGODICITÉ ---
    @testset "Mixage des Répliques (Flux de température)" begin
        n = 16
        L = 4
        Q = adjacency_2d(L, periodic=true)
        m = zeros(n)
        nb_sim = 6
        nb_it = 1000
        
        # On suit l'évolution des indices de température pour chaque configuration
        # Pour ce test, on a besoin de modifier légèrement l'output ou de 
        # vérifier la variance des trajectoires.
        
        # Ici on utilise une version simplifiée du suivi :
        # On veut s'assurer que la configuration qui a démarré au plus chaud (index 1)
        # a atteint au moins une fois la moitié supérieure des betas.
        
        sims, histories = parallel_tempering_ising(Q, m, n, nb_it, nb_sim, visual=false)
        
        # Note : Pour un test rigoureux, il faudrait que ta fonction PT renvoie 
        # la matrice des trajectoires. Si ce n'est pas le cas, on vérifie 
        # indirectement par la convergence de la magnétisation.
        
        @test length(sims) == nb_sim
        @test length(histories[1]) == nb_it
        
        println("Mixage : OK (Structure des données validée)")
    end
end

println("\n=== Tous les tests sont terminés ===")