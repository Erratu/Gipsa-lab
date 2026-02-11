using Test, Printf
include("ising_model_2.jl")

@testset "Ising Model Suite" begin

    @testset "Validation Loi de Proba N=4" begin
        n = 4
        beta = 0.5
        # Chaîne simple (Q)
        Q = zeros(n,n); for i=1:n-1 Q[i,i+1]=Q[i+1,i]=1 end
        m = zeros(n)
        
        # Énumération exhaustive des 16 états
        states = [ [(i >> j) & 1 == 1 ? 1 : -1 for j in 0:n-1] for i in 0:(2^n-1) ]
        energies = [energy(s, Q, m) for s in states]
        Z = sum(exp.(-beta .* energies))
        probs_theo = exp.(-beta .* energies) ./ Z

        # Simulation
        counts = zeros(2^n)
        X = rand([-1,1], n)
        for _ in 1:20000
            heat_bath_ising!(X, Q, m, beta, n)
            idx = findfirst(s -> s == X, states)
            counts[idx] += 1
        end
        probs_emp = counts ./ sum(counts)

        # Calcul de l'écart maximal pour affichage
        diff_max = maximum(abs.(probs_emp .- probs_theo))
        @printf("\n[Loi de Proba N=4] Différence max : %.4f (Seuil : 0.05)\n", diff_max)

        @test diff_max < 0.05
    end

    @testset "Conditions aux limites & Minima" begin
        n = 10
        # TEST 1 : Chaîne 1D bords identiques -> Minimum global aligné
        Q = adjacency_1d(n, periodic=false)
        m = zeros(n)
        fixed = [1, n]
        X = rand([-1,1], n)
        X[1], X[n] = 1, 1 
        heat_bath_ising!(X, Q, m, 5.0, 5000, fixed_indices=fixed)
        
        err_align = n - abs(sum(X))
        @printf("\n[1D Dirichlet] Erreur d'alignement : %d (Attendu : 0)\n", err_align)
        @test all(X .== 1)

        # TEST 2 : Cercle 1D (Périodique)
        Q_cercle = adjacency_1d(n, periodic=true)
        X = rand([-1,1], n)
        heat_bath_ising!(X, Q_cercle, m, 5.0, 5000)
        
        res_cercle = abs(sum(X))
        @printf("[Cercle 1D] Magnétisation totale : %d (Cible : %d)\n", res_cercle, n)
        # Sur un cercle à basse température, tous les spins doivent être identiques
        @test res_cercle == n 
    end

    @testset "Tests 2D (N = 3, 4, 5)" begin
        @printf("\n[Tests 2D Basse T]")
        for L in [3, 4, 5]
            n = L*L
            Q = adjacency_2d(L, periodic=true)
            m = zeros(n)
            X = rand([-1,1], n)
            # Basse température
            heat_bath_ising!(X, Q, m, 2.0, n*100)
            
            res_2d = abs(sum(X))
            @printf("\n -> L=%d : Obtenu %d / Attendu %d", L, res_2d, n)
            @test res_2d == n # Alignement complet
        end
        println()
    end

    @testset "Transition de Phase (2D)" begin
        L = 10
        n = L*L
        Q = adjacency_2d(L, periodic=true)
        m = zeros(n)
        
        # On utilise le Parallel Tempering pour éviter de rester bloqué
        # Le replica le plus froid doit montrer une magnétisation spontanée
        sims, histories = parallel_tempering_ising(Q, m, n, 1000, 8, visual=false)
        
        # Beta max est autour de 1.5 (froid), Beta min autour de 0.1 (chaud)
        mag_froid = abs(sum(sims[end]) / n)
        mag_chaud = abs(sum(sims[1]) / n)
        
        @printf("\n[Transition 2D PT] |M| Froid : %.3f (Seuil > 0.8) | |M| Chaud : %.3f (Seuil < 0.3)\n", mag_froid, mag_chaud)
        
        @test mag_froid > 0.8  # Phase ordonnée (Ferromagnétique)
        @test mag_chaud < 0.3  # Phase désordonnée (Paramagnétique)
        
        # Visualisation du phénomène de transition
        # La courbe de histories[end] doit être stable et haute
        # La courbe de histories[1] doit fluctuer autour de 0
    end

    @testset "Magnétisation Spontanée 2D heat bath" begin
        L = 8; n = L*L; Q = adjacency_2d(L, periodic=true)
        X = rand([-1,1], n)
        # À basse température, le système doit s'ordonner (|M| proche de 1)
        heat_bath_ising!(X, Q, zeros(n), 1.0, 5000)
        
        mag_hb = abs(sum(X)/n)
        @printf("\n[Spontanée HB 2D] |M| final : %.3f (Seuil > 0.8)\n", mag_hb)
        @test mag_hb > 0.8
    end

end