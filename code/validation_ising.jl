using Statistics, LinearAlgebra, Printf

include("ising_model_2.jl")

# Metropolis, Heat Bath, etc. (on suppose qu'elles sont chargées)

L = 6
N = L * L
beta = 0.5
n_essais = 5
n_steps = 4000
Q = adjacency_2d(L, periodic=true)
m = zeros(N)

# Tableaux pour stocker le résultat final de chaque essai
results_metro = []
results_hb = []
results_pt = []

E_metro = []
E_hb = []
E_pt = []

for essai in 1:n_essais
    # On repart d'une config aléatoire à chaque essai
    X_init = rand([-1, 1], N)
    energies_pt_froid = Float64[]
    
    # --- Metropolis ---
    X_m = copy(X_init)
    _, energies_metro = metropolis_ising!(X_m, Q, m, beta, n_steps)
    push!(results_metro, energy(X_m, Q, m) / N)
    half = div(length(energies_metro), 2)
    push!(E_metro, mean(energies_metro[half:end]) / N)
    
    # --- Heat Bath ---
    X_h = copy(X_init)
    _, energies_hb = heat_bath_ising!(X_h, Q, m, beta, n_steps)
    push!(results_hb, energy(X_h, Q, m) / N)
    half = div(length(energies_hb), 2)
    push!(E_hb, mean(energies_hb[half:end]) / N)
    
    # --- Parallel Tempering ---
    # Pour PT, on lance 8 répliques en parallèle pendant n_steps
    nb_sim = 8
    betas = collect(range(0.1, beta, length=nb_sim))
    sims = [rand([-1, 1], N) for _ in 1:nb_sim]
    
    for t in 1:n_steps
        for s in 1:nb_sim
            # Ici 1 step PT = 1 itération par réplique
            heat_bath_ising!(sims[s], Q, m, betas[s], 1)
        end
        # Tentative de swap tous les N steps (pour donner une chance de bouger)
        if t % N == 0
            for s in 1:nb_sim-1
                E1, E2 = energy(sims[s], Q, m), energy(sims[s+1], Q, m)
                if rand() < exp((betas[s] - betas[s+1]) * (E1 - E2))
                    sims[s], sims[s+1] = sims[s+1], sims[s]
                end
            end
        end
        push!(energies_pt_froid, energy(sims[end], Q, m))
    end
    push!(results_pt, energy(sims[end], Q, m) / N)
    half = div(length(energies_pt_froid), 2)
    push!(E_pt, mean(energies_pt_froid[half:end]) / N)
end

# --- Affichage des moyennes ---
println("Moyenne sur $n_essais essais ($n_steps steps chacun) :")
@printf("Cible : -1.7473\n")
@printf("Metropolis : %.4f\n", mean(results_metro))
@printf("Heat Bath  : %.4f\n", mean(results_hb))
@printf("PT         : %.4f\n", mean(results_pt))
println("\nÉnergie moyenne par spin :")
@printf("Cible : -1.7473\n")
@printf("Metropolis : %.4f\n", mean(E_metro))
@printf("Heat Bath  : %.4f\n", mean(E_hb))
@printf("PT         : %.4f\n", mean(E_pt))