using ThomsonMeths
using Test, Printf, LinearAlgebra, Random, StatsBase

function bp_ising_min_sum(J::Matrix{Float64}, h::Vector{Float64}; max_iter=100)
    N = size(J, 1)
    
    # Initialisation des messages
    u = zeros(N, N)
    u_new = zeros(N, N)
    
    # Sur un graphe 1D (arbre), BP converge en exactement N itérations.
    # Pas besoin de "damping" (inertie) sur un arbre.
    for iter in 1:max_iter
        total_u_in = [sum(u[k, i] for k in 1:N if k != i) for i in 1:N]
        
        for i in 1:N
            for j in 1:N
                # On ne calcule des messages que s'il y a une interaction
                if J[i, j] == 0.0 || i == j 
                    continue 
                end
                
                # Le champ de cavité classique
                cavity_field_i = h[i] + total_u_in[i] - u[j, i]
                
                # Règle de mise à jour Min-Sum exacte
                u_new[i, j] = sign(J[i, j]) * sign(cavity_field_i) * min(abs(J[i, j]), abs(cavity_field_i))
            end
        end
        
        diff = norm(u_new - u)
        u = copy(u_new)
        
        if diff < 1e-6
            break
        end
    end
    
    # Calcul des croyances (Beliefs) finales
    beliefs = zeros(N)
    for i in 1:N
        beliefs[i] = h[i] + sum(u[k, i] for k in 1:N if k != i)
    end
    
    # L'état fondamental est simplement le signe des croyances
    best_state = sign.(beliefs)
    # Sécurité : si un champ est exactement 0, on le met à 1
    best_state[best_state .== 0.0] .= 1.0 
    
    # Calcul de l'énergie d'Ising standard : E = - sum(J_ij s_i s_j) - sum(h_i s_i)
    # Le 0.5 corrige le double comptage de la matrice symétrique J
    E_min = -0.5 * sum(J[i, j] * best_state[i] * best_state[j] for i in 1:N, j in 1:N) - dot(h, best_state)
    
    return best_state, E_min
end

@testset "Validation Exacte des Méthodes de Sélection (n parmi m)" begin

    @testset "1D Anti-Ferromagnétique (Répulsion stricte)" begin
        # Contexte : m points en ligne. Répulsion entre voisins directs.
        # But : Choisir n = m/2 points.
        # Solution exacte : Prendre 1 point sur 2 (ex: 1, 3, 5, 7, 9) pour une énergie de 0.0.
        m = 10
        n = 5
        H = zeros(m, m)
        for i in 1:m-1
            H[i, i+1] = 1.0
            H[i+1, i] = 1.0
        end
        
        idx_ex, E_ex = methode_exhaustive(H, n)
        @test E_ex == 0.0
        
        idx_heu, E_heu = methode_heuristique(H, n)
        @test E_heu == 0.0
        
        idx_bp, E_bp = methode_belief_propagation(H, n)
        @test E_bp == 0.0

        @info "[1D Anti-Ferro] OK (E_exact = 0.0 | Heuristique = $E_heu | BP = $E_bp)"
    end

    @testset "1D Ferromagnétique (Attraction stricte)" begin
        # Contexte : m points en ligne. Attraction entre voisins directs.
        # But : Choisir n points.
        # Solution exacte : Choisir n points adjacents (un bloc compact).
        # Énergie : n points adjacents forment n-1 liaisons. Chaque liaison vaut -1. 
        # Énergie totale = (n-1) * (-1.0)
        m = 10
        n = 4
        H = zeros(m, m)
        for i in 1:m-1
            H[i, i+1] = -1.0
            H[i+1, i] = -1.0
        end
        
        E_theo = (n - 1) * (-1.0) # -3.0
        
        idx_ex, E_ex = methode_exhaustive(H, n)
        @test E_ex == E_theo
        
        idx_heu, E_heu = methode_heuristique(H, n)
        @test E_heu == E_theo
        
        idx_bp, E_bp = methode_belief_propagation(H, n)
        @test E_bp == E_theo

        @info "[1D Ferro] OK (E_exact = $E_theo | Heuristique = $E_heu | BP = $E_bp)"
    end

    @testset "2D Grille Anti-Ferromagnétique (Damier)" begin
        # Contexte : Grille 4x4, répulsion entre voisins haut/bas/gauche/droite.
        # But : Choisir 8 points sur 16 (la moitié).
        # Solution exacte : Le motif en damier. Aucune particule ne touche une autre. Énergie = 0.0.
        L = 4
        m = L * L
        n = div(m, 2)
        H = zeros(m, m)
        
        # Construction de l'adjacence 2D (sans conditions périodiques pour simplifier)
        for i in 1:m
            x, y = (i-1)%L, div(i-1, L)
            if x < L-1 H[i, i+1] = 1.0; H[i+1, i] = 1.0 end # Droite
            if y < L-1 H[i, i+L] = 1.0; H[i+L, i] = 1.0 end # Bas
        end

        idx_ex, E_ex = methode_exhaustive(H, n)
        @test E_ex == 0.0
        
        idx_heu, E_heu = methode_heuristique(H, n)
        @test E_heu == 0.0
        
        idx_bp, E_bp = methode_belief_propagation(H, n)
        @test E_bp == 0.0

        @info "[2D Grille Anti-Ferro (Damier)] OK (E_exact = 0.0 | BP = $E_bp)"
    end

    @testset "Robustesse sur Graphe Dense (Test type Thomson petit format)" begin
        # Contexte : Matrice symétrique aléatoire (dense, 100% de connexions)
        # But : S'assurer que BP et Heuristique ne font pas n'importe quoi comparé à la vérité absolue.
        m = 16
        n = 8
        
        # Génération d'un graphe dense aléatoire
        H = randn(m, m)
        H = (H + H') / 2
        for i in 1:m H[i, i] = 0.0 end
        
        idx_ex, E_ex = methode_exhaustive(H, n)
        idx_heu, E_heu = methode_heuristique(H, n)
        idx_bp, E_bp = methode_belief_propagation(H, n)
        
        # L'heuristique et BP sont des approximations sur des graphes denses.
        # Ils ne trouveront pas toujours E_ex, mais ils doivent trouver une énergie valide.
        # Le test valide simplement que l'énergie renvoyée correspond bien aux indices.
        @test evaluer_energie(H, idx_heu) ≈ E_heu
        @test evaluer_energie(H, idx_bp) ≈ E_bp
        
        # Statistiquement, pour m=16, les algos trouvent souvent le minimum absolu
        diff_heu = E_heu - E_ex
        diff_bp = E_bp - E_ex

        @info "[Graphe Dense Aléatoire m=$m, n=$n]"
        @info @sprintf(" -> E_exact      : %.4f", E_ex)
        @info @sprintf(" -> E_heuristique: %.4f (Diff: +%.4f)", E_heu, diff_heu)
        @info @sprintf(" -> E_BP         : %.4f (Diff: +%.4f)", E_bp, diff_bp)
    end

    @testset "Validation BP Min-Sum sur Ising 1D pur" begin
        # N=15 permet un calcul exhaustif instantané (2^15 = 32768 combinaisons)
        N = 15
        
        # Création d'une chaîne 1D avec des interactions aléatoires
        J = zeros(N, N)
        for i in 1:N-1
            # Des couplages au hasard (attractifs et répulsifs de forces diverses)
            couplage = randn() 
            J[i, i+1] = couplage
            J[i+1, i] = couplage
        end

        # Champs magnétiques locaux aléatoires
        h = randn(N)

        # --- 1. Résolution Exhaustive (La vérité absolue) ---
        # Génération de tous les états possibles {-1, 1}^N
        states = [[(i >> j) & 1 == 1 ? 1.0 : -1.0 for j in 0:N-1] for i in 0:(2^N-1)]

        # Fonction d'énergie d'Ising
        function calc_E_ising(s)
            return -0.5 * sum(J[i, j] * s[i] * s[j] for i in 1:N, j in 1:N) - dot(h, s)
        end

        energies = calc_E_ising.(states)
        E_exact = minimum(energies)
        idx_exact = argmin(energies)
        state_exact = states[idx_exact]

        # --- 2. Résolution par Belief Propagation Min-Sum ---
        state_bp, E_bp = bp_ising_min_sum(J, h)

        # --- 3. Les Tests ---
        @test E_bp ≈ E_exact atol=1e-7
        @test state_bp == state_exact

        @info "[Ising 1D Aléatoire] OK"
        @info @sprintf(" -> Énergie Exacte : %.5f", E_exact)
        @info @sprintf(" -> Énergie BP     : %.5f", E_bp)
    end
end;