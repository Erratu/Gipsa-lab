using Plots
include("ising_model_2.jl")

#Plots.gr() # Backend rapide
plotlyjs()

function run_full_comparison(L=20, nb_it=4000, beta_froid=0.8, m=zeros(L*L))
    n = L*L
    Q = adjacency_2d(L, periodic=true)
    
    # Historiques HB
    X_hb = rand([-1, 1], n)
    mag_hb = Float64[]
    ene_hb = Float64[]
    
    # Historiques PT
    nb_sim = 12
    betas = collect(range(0.1, beta_froid, length=nb_sim))
    sims_pt = [rand([-1, 1], n) for _ in 1:nb_sim]
    mag_pt_froid = Float64[]
    ene_pt_froid = Float64[]
    
    trajectoires = [Int[] for _ in 1:nb_sim]
    indices_betas = collect(1:nb_sim) 

    println("Simulation en cours (L=$L, m=$(m[1]))...")

    for t in 1:nb_it
        # 1. Update Heat Bath (HB)
        heat_bath_ising!(X_hb, Q, m, beta_froid, n)
        push!(mag_hb, abs(sum(X_hb)/n))
        push!(ene_hb, energy(X_hb, Q, m) / n)

        # 2. Update Parallel Tempering (PT)
        for s in 1:nb_sim
            heat_bath_ising!(sims_pt[s], Q, m, betas[s], n)
        end
        
        # Swaps entre répliques
        for s in 1:nb_sim-1
            E1, E2 = energy(sims_pt[s], Q, m), energy(sims_pt[s+1], Q, m)
            if rand() < exp((betas[s] - betas[s+1]) * (E1 - E2))
                sims_pt[s], sims_pt[s+1] = sims_pt[s+1], sims_pt[s]
                indices_betas[s], indices_betas[s+1] = indices_betas[s+1], indices_betas[s]
            end
        end
        
        # Enregistrement des données PT froid
        push!(mag_pt_froid, abs(sum(sims_pt[end])/n))
        push!(ene_pt_froid, energy(sims_pt[end], Q, m) / n)
        
        # Suivi de la trajectoire des configurations
        for i in 1:nb_sim
            push!(trajectoires[i], findfirst(==(i), indices_betas))
        end

        # Mise à jour du dashboard
        if t % 100 == 0
            # Graphiques de Magnétisation et Énergie
            p_mag = plot([mag_hb, mag_pt_froid], labels=["HB" "PT"], title="Magnétisation |M|", lw=2)
            p_ene = plot([ene_hb, ene_pt_froid], labels=["HB" "PT"], title="Énergie par spin (E/N)", lw=2, color=[:blue :red])
            
            # Trajectoires
            p_traj = plot(title="Trajectoires des configurations", ylabel="Niveau Beta", legend=false)
            for i in 1:nb_sim
                plot!(p_traj, trajectoires[i], alpha=0.3)
            end
            
            # Heatmaps finales
            p_pt = heatmap(reshape(sims_pt[end], L, L), c=:binary, title="PT Final (Froid)", ratio=1, axis=false)
            p_hb = heatmap(reshape(X_hb, L, L), c=:binary, title="HB Final", ratio=1, axis=false)
            
            # Organisation du Dashboard
            # Layout : 2 en haut, 1 large au milieu, 2 en bas
            layout = @layout [grid(1,2); a{0.3h} ; grid(1,2)]
            display(plot(p_mag, p_ene, p_traj, p_pt, p_hb, layout=layout, size=(1000, 1000)))
        end
    end
end

L_difficile = 40 
nb_it_difficile = 5000
beta_tres_froid = 1.2 # Plus beta est fort, plus le HB est "rigide"

#run_full_comparison(L_difficile, nb_it_difficile, beta_tres_froid) # magnétisation spontanée
#run_full_comparison(20, 4000, 0.8) # magnétisation spontanée visible en flip-flop

m = fill(0.02, 20*20) # Champ très faible (ne force pas l'alignement)
run_full_comparison(20, 4000, 0.8, m) # magnétisation spontanée visible sans flip-flop

#run_full_comparison(L_difficile, nb_it_difficile, 0.2) # pas de magnétisation spontanée
#run_full_comparison(L_difficile, nb_it_difficile, 0.44) # transition de phase