# Fichier : test_ising.jl
include("implem_ising.jl") # Charge tes fonctions
using Test
using LinearAlgebra
using Statistics

# Fonction utilitaire pour vérifier l'énergie dans les tests
function get_energy(X, Q, m)
    return 0.5 * dot(X, Q, X) - dot(m, X)
end

println("Démarrage de la suite de tests Ising\n")

@testset "Validation Parallel Tempering" begin

    # --- TEST 1 : CHAMP MAGNÉTIQUE FORT ---
    # Si m est positif et très fort, tous les spins doivent s'aligner à +1
    # Energie : -m * X. Minimale si X = 1.
    @testset "Champ Fort (m > 0 -> X = 1)" begin
        n = 10
        Q = zeros(n, n) # Pas d'interaction
        m = fill(20.0, n) # Champ très fort positif
        
        # On lance la simu
        X_final = parallel_tempering_ising(Q, m, n, 200, 5)
        
        # Vérification
        @test all(X_final .== 1)
        println("Champ Fort : OK (Tous les spins à 1)")
    end

    # --- TEST 2 : ANTI-FERROMAGNÉTISME ---
    # Si Qij = 1 (répulsif), les voisins doivent être opposés.
    # Pour n=2, cela veut dire X=[1, -1] ou [-1, 1]
    @testset "Anti-ferro (Q=1 -> Spins opposés)" begin
        n = 2
        Q = ones(n, n); Q[1,1]=0; Q[2,2]=0
        m = zeros(n)
        
        X_final = parallel_tempering_ising(Q, m, n, 200, 5)
        
        @test X_final[1] != X_final[2]
        @test get_energy(X_final, Q, m) == -1.0
        println("Anti-ferro : OK (Spins opposés)")
    end

    # --- TEST 3 : CONVERGENCE D'ÉNERGIE ---
    # L'algorithme doit trouver une énergie plus basse qu'une configuration aléatoire
    @testset "Optimisation Globale" begin
        n = 15
        # Matrice symétrique aléatoire
        Q = randn(n, n)
        Q = (Q + Q') / 2 
        Q[diagind(Q)] .= 0
        m = randn(n)
        
        # Référence aléatoire (bruit)
        X_rand = rand([-1, 1], n)
        E_rand = get_energy(X_rand, Q, m)
        
        # Résultat optimisé
        X_opt = parallel_tempering_ising(Q, m, n, 300, 8)
        E_opt = get_energy(X_opt, Q, m)
        
        # On vérifie que l'énergie a baissé
        @test E_opt < E_rand
        println("Optimisation : OK (E_init: $(round(E_rand, digits=2)) -> E_opt: $(round(E_opt, digits=2)))")
    end

end

println("\nTous les tests sont terminés.")