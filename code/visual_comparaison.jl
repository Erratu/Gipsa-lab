using Plots
include("ising_model_2.jl")

# Force l'utilisation du backend GR pour la performance
gr()

function run_full_comparison(L=16, nb_it=2000, beta_froid=0.8)
    n = L*L
    Q = adjacency_2d(L, periodic=true)
    m = zeros(n)
    
    # Init Heat Bath (HB)
    X_hb = rand([-1, 1], n)
    mag_hb = Float64[]
    
    # Init Parallel Tempering (PT)
    nb_sim = 12
    betas = collect(range(0.1, beta_froid, length=nb_sim))
    sims_pt = [rand([-1, 1], n) for _ in 1:nb_sim]
    mag_pt_froid = Float64[]
    
    # Suivi de TOUTES les trajectoires
    trajectoires = [Int[] for _ in 1:nb_sim]
    indices_betas = collect(1:nb_sim) 

    println("Lancement de la comparaison (L=$L, répliques=$nb_sim)...")

    for t in 1:nb_it
        # 1. Update HB
        heat_bath_ising!(X_hb, Q, m, beta_froid, n)
        push!(mag_hb, abs(sum(X_hb)/n))

        # 2. Update PT
        for s in 1:nb_sim
            heat_bath_ising!(sims_pt[s], Q, m, betas[s], n)
        end
        
        # Swaps
        for s in 1:nb_sim-1
            E1, E2 = energy(sims_pt[s], Q, m), energy(sims_pt[s+1], Q, m)
            if rand() < exp((betas[s] - betas[s+1]) * (E1 - E2))
                sims_pt[s], sims_pt[s+1] = sims_pt[s+1], sims_pt[s]
                indices_betas[s], indices_betas[s+1] = indices_betas[s+1], indices_betas[s]
            end
        end
        
        # Enregistrement de TOUTES les positions
        for i in 1:nb_sim
            push!(trajectoires[i], findfirst(==(i), indices_betas))
        end
        push!(mag_pt_froid, abs(sum(sims_pt[end])/n))

        if t % 50 == 0
            p1 = Plots.plot([mag_hb, mag_pt_froid], labels=["HB" "PT"], title="Magnétisation")
            
            # Graphe de trajectoires multiples
            p2 = Plots.plot(title="Trajectoires (Echanges)", ylabel="Beta Index", legend=false)
            for i in 1:nb_sim
                Plots.plot!(p2, trajectoires[i], alpha=0.4)
            end
            
            p3 = Plots.heatmap(reshape(sims_pt[end], L, L), c=:binary, title="PT Froid", ratio=1)
            p4 = Plots.heatmap(reshape(X_hb, L, L), c=:binary, title="HB Froid", ratio=1)
            
            display(Plots.plot(p1, p2, p3, p4, layout=(2,2), size=(900, 700)))
        end
    end
end

# Exécution
# run_full_comparison(16, 2000)

L_difficile = 40 
nb_it_difficile = 5000
beta_tres_froid = 1.2 # Plus beta est fort, plus le HB est "rigide"

run_full_comparison(L_difficile, nb_it_difficile, beta_tres_froid) # magnétisation spontanée
#run_full_comparison(L_difficile, nb_it_difficile, 0.2) # pas de magnétisation spontanée
#run_full_comparison(L_difficile, nb_it_difficile, 0.44) # transition de phase