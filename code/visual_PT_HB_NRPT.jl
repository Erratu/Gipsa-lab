using Plots
include("ising_model_2.jl")

# Utilisation de PlotlyJS pour pouvoir zoomer sur les trajectoires
plotlyjs()

function run_full_comparison(L=20, nb_it=4000, beta_froid=0.8, m=zeros(L*L))
    n = L*L
    Q = adjacency_2d(L, periodic=true)
    
    # --- 1. INITIALISATION DES MODÈLES ---
    
    # Heat Bath (HB)
    X_hb = rand([-1, 1], n)
    mag_hb, ene_hb = Float64[], Float64[]
    
    # Parallel Tempering (PT)
    nb_sim = 12
    betas = collect(range(0.1, beta_froid, length=nb_sim))
    sims_pt = [rand([-1, 1], n) for _ in 1:nb_sim]
    mag_pt, ene_pt = Float64[], Float64[]
    indices_pt = collect(1:nb_sim) 
    trajectoires_pt = [Int[] for _ in 1:nb_sim]

    # Non-Reversible Parallel Tempering (NRPT) 
    sims_nrpt = [rand([-1, 1], n) for _ in 1:nb_sim]
    mag_nrpt, ene_nrpt = Float64[], Float64[]
    indices_nrpt = collect(1:nb_sim) 
    trajectoires_nrpt = [Int[] for _ in 1:nb_sim]

    println("Lancement du duel : HB vs PT vs NRPT (L=$L, m=$(m[1]))...")

    for t in 1:nb_it
        heat_bath_ising!(X_hb, Q, m, beta_froid, n)
        for s in 1:nb_sim
            heat_bath_ising!(sims_pt[s], Q, m, betas[s], n)
            heat_bath_ising!(sims_nrpt[s], Q, m, betas[s], n)
        end
        
        # SWAPS PT CLASSIQUE (Réversible)
        for _ in 1:nb_sim
            s = rand(1:nb_sim-1)
            E1, E2 = energy(sims_pt[s], Q, m), energy(sims_pt[s+1], Q, m)
            if rand() < exp((betas[s] - betas[s+1]) * (E1 - E2))
                sims_pt[s], sims_pt[s+1] = sims_pt[s+1], sims_pt[s]
                indices_pt[s], indices_pt[s+1] = indices_pt[s+1], indices_pt[s]
            end
        end

        # SWAPS NRPT (Non-Réversible)
        offset = (t % 2 == 0) ? 1 : 2
        for s in offset:2:nb_sim-1
            E1, E2 = energy(sims_nrpt[s], Q, m), energy(sims_nrpt[s+1], Q, m)
            if rand() < exp((betas[s] - betas[s+1]) * (E1 - E2))
                sims_nrpt[s], sims_nrpt[s+1] = sims_nrpt[s+1], sims_nrpt[s]
                indices_nrpt[s], indices_nrpt[s+1] = indices_nrpt[s+1], indices_nrpt[s]
            end
        end
        
        # --- ENREGISTREMENT DES DONNÉES ---
        push!(mag_hb, abs(sum(X_hb)/n)); push!(ene_hb, energy(X_hb, Q, m)/n)
        push!(mag_pt, abs(sum(sims_pt[end])/n)); push!(ene_pt, energy(sims_pt[end], Q, m)/n)
        push!(mag_nrpt, abs(sum(sims_nrpt[end])/n)); push!(ene_nrpt, energy(sims_nrpt[end], Q, m)/n)
        
        for i in 1:nb_sim
            push!(trajectoires_pt[i], findfirst(==(i), indices_pt))
            push!(trajectoires_nrpt[i], findfirst(==(i), indices_nrpt))
        end

        # --- AFFICHAGE ---
        if t % 100 == 0
            # Graphiques comparatifs temporels
            p_mag = plot([mag_hb, mag_pt, mag_nrpt], labels=["HB" "PT" "NRPT"], 
                         title="Magnétisation |M|", lw=1.5, color=[:blue :green :red])
            p_ene = plot([ene_hb, ene_pt, ene_nrpt], labels=["HB" "PT" "NRPT"], 
                         title="Énergie par spin E/N", lw=1.5, color=[:blue :green :red])
            
            # Comparaison des trajectoires (on suit la première configuration de chaque)
            p_tr_pt = plot(trajectoires_pt[1], title="Traj PT (Marche Aléatoire)", 
                           legend=false, color=:green, alpha=0.7)
            p_tr_nrpt = plot(trajectoires_nrpt[1], title="Traj NRPT (Flux Dirigé)", 
                             legend=false, color=:red, alpha=0.7)
            
            # Grilles finales (PT vs NRPT)
            p_h_pt = heatmap(reshape(sims_pt[end], L, L), c=:binary, title="PT Froid", ratio=1, axis=false, clims=(-1, 1))
            p_h_nrpt = heatmap(reshape(sims_nrpt[end], L, L), c=:binary, title="NRPT Froid", ratio=1, axis=false, clims=(-1, 1))
            
            # Mise en page du Dashboard
            l = @layout [grid(1,2); grid(1,2) ; grid(1,2)]
            display(plot(p_mag, p_ene, p_tr_pt, p_tr_nrpt, p_h_pt, p_h_nrpt, layout=l, size=(1000, 1000)))
        end
    end
end

# Test avec champ magnétique faible pour voir lequel trouve le minimum global le plus vite
L = 20
m_val = 0.02
m_field = fill(m_val, L*L)
run_full_comparison(L, 5000, 0.8, m_field)