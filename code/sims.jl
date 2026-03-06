using Plots
include("ising_structures.jl")
include("ising_models.jl")

function create_dashboard(hb::ChainState, pt::ChainState, nrpt::ChainState, L, n_sample)
    # Trajectoires (uniquement PT et NRPT, car HB n'a qu'une seule chaîne)
    p1 = plot(nrpt.trajectoires[1], title="Trajectoire NRPT (Chaîne 1)", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:blue, linewidth=1.5)
    p2 = plot(pt.trajectoires[1], title="Trajectoire PT (Chaîne 1)", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:red, linewidth=1.5)
    
    # Magnétisation (Comparaison des 3 algos sur la chaîne cible)
    p_mag = plot(hb.mag_hist[1], label="Heat Bath", color=:green, alpha=0.8, title="Magnétisation (Cible, β=$(hb.betas[1]))", xlabel="Scans", ylabel="|M|")
    plot!(p_mag, pt.mag_hist[end], label="PT Classique", color=:red, alpha=0.7)
    plot!(p_mag, nrpt.mag_hist[end], label="NRPT", color=:blue, alpha=0.7)
    
    # Énergie (Comparaison des 3 algos sur la chaîne cible)
    p_ene = plot(hb.ene_hist[1], label="Heat Bath", color=:green, alpha=0.8, title="Énergie (Cible, β=$(hb.betas[1]))", xlabel="Scans", ylabel="E")
    plot!(p_ene, pt.ene_hist[end], label="PT Classique", color=:red, alpha=0.7)
    plot!(p_ene, nrpt.ene_hist[end], label="NRPT", color=:blue, alpha=0.7)
    
    # Round Trips
    noms = ["NRPT", "PT Classique"]
    valeurs_rt = [nrpt.round_trips, pt.round_trips]
    p_rt = bar(noms, valeurs_rt, title="Nombre de Round Trips (Mélange)", ylabel="Count", legend=false, color=[:blue, :red], bar_width=0.4)
    
    # Mise en page (Layout)
    l = @layout [
        a b
        c d
        e
    ]
    
    dash = plot(p1, p2, p_mag, p_ene, p_rt, layout=l, size=(1000, 900), margin=5Plots.mm)
    return dash
end

function run_simulation(L=20, nb_it=4000, nb_tune=1000, beta_min=0.0, beta_max=0.8, m_val=0.02, nb_chains=30)
    Q = adjacency_2d(L, periodic=true)
    m_field = fill(m_val, L*L)
    
    betas = collect(range(beta_min, beta_max, length=nb_chains))
    
    hb = ChainState(L, [beta_max])
    pt = ChainState(L, copy(betas))
    nrpt = ChainState(L, copy(betas))

    maxRound = floor(Int, log2(nb_tune))
    n_burnin = (2^maxRound) - 1 
    n_sample = nb_it - n_burnin 
    
    if n_sample <= 0
        error("nb_it est trop petit par rapport à nb_tune. Augmentez nb_it.")
    end

    println("Lancement : $n_burnin scans de tuning/burn-in, suivis de $n_sample scans d'exploitation.")
    println("Bornes des températures : de $beta_min à $beta_max avec $nb_chains chaînes.")

    # Exécution de Heat Bath
    run_hb_complet!(hb, Q, m_field, n_burnin, n_sample)
    
    # Exécution de Parallel Tempering Classique
    # (Utilise la fonction algo_pt_complet! définie dans les messages précédents)
    algo_pt_complet!(pt, Q, m_field, n_burnin, n_sample)
    
    # Exécution de NRPT
    algo4_nrpt_complet!(nrpt, Q, m_field, nb_tune, n_sample)

    println("\nGénération du Dashboard en cours...")
    
    # Génération et affichage du Dashboard Final
    final_dash = create_dashboard(hb, pt, nrpt, L, n_sample)
    
    display(final_dash)
    savefig(final_dash, "dashboard_N$(nb_chains)_final.png")
    
    println("Simulation terminée et graphique sauvegardé sous 'dashboard_N$(nb_chains)_final.png'")
    
    return hb, pt, nrpt
end

run_simulation()