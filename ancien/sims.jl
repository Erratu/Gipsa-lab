using Plots
include("ising_structures.jl")
include("ising_models.jl")

using Plots

function create_dashboard(hb::ChainState, pt::ChainState, nrpt::ChainState, wolff::ChainState, wl::ChainState, L, n_sample)
    
    # Trajectoires
    p1 = plot(nrpt.trajectoires[1], title="Trajectoire NRPT", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:blue, linewidth=1.5)
    p2 = plot(pt.trajectoires[1], title="Trajectoire PT", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:red, linewidth=1.5)
    
    # Magnétisation (Superposition des 5)
    p_mag = plot(hb.mag_hist[1], label="Heat Bath", color=:green, alpha=0.5, title="Magnétisation (β=$(hb.betas[1]))", xlabel="Scans", ylabel="|M|")
    plot!(p_mag, pt.mag_hist[end], label="PT Classique", color=:red, alpha=0.5)
    plot!(p_mag, nrpt.mag_hist[end], label="NRPT", color=:blue, alpha=0.5)
    plot!(p_mag, wolff.mag_hist[1], label="Wolff", color=:purple, alpha=0.8, linewidth=2)
    plot!(p_mag, wl.mag_hist[1], label="Wang-Landau", color=:orange, alpha=0.5)
    
    # Énergie (Superposition des 5)
    p_ene = plot(hb.ene_hist[1], label="Heat Bath", color=:green, alpha=0.5, title="Énergie (β=$(hb.betas[1]))", xlabel="Scans", ylabel="E")
    plot!(p_ene, pt.ene_hist[end], label="PT Classique", color=:red, alpha=0.5)
    plot!(p_ene, nrpt.ene_hist[end], label="NRPT", color=:blue, alpha=0.5)
    plot!(p_ene, wolff.ene_hist[1], label="Wolff", color=:purple, alpha=0.8, linewidth=2)
    
    # Wang-Landau va balayer tout l'espace d'énergie, c'est normal !
    plot!(p_ene, wl.ene_hist[1], label="Wang-Landau", color=:orange, alpha=0.7)
    
    # Round Trips
    noms = ["NRPT", "PT Classique"]
    valeurs_rt = [nrpt.round_trips, pt.round_trips]
    p_rt = bar(noms, valeurs_rt, title="Round Trips", ylabel="Count", legend=false, color=[:blue, :red], bar_width=0.4)
    
    l = @layout [
        a b
        c d
        e
    ]
    
    dash = plot(p1, p2, p_mag, p_ene, p_rt, layout=l, size=(1200, 1000), margin=5Plots.mm)
    return dash
end

function run_simulation(L=20, nb_it=4000, nb_tune=1000, beta_min=0.0, beta_max=0.8, m_val=0.02, nb_chains=30)
    Q = adjacency_2d(L, periodic=true)
    m_field = fill(m_val, L*L)
    
    betas = collect(range(beta_min, beta_max, length=nb_chains))
    
    # Création des 5 états
    hb = ChainState(L, [beta_max])
    wolff = ChainState(L, [beta_max])
    wl = ChainState(L, [beta_max]) # beta_max ignoré par WL, mais utile pour la structure
    
    pt = ChainState(L, copy(betas))
    nrpt = ChainState(L, copy(betas))

    maxRound = floor(Int, log2(nb_tune))
    n_burnin = (2^maxRound) - 1 
    n_sample = nb_it - n_burnin 

    println("=========================================================")
    println("Lancement Global : $n_burnin scans de tuning, suivis de $n_sample scans d'exploitation.")
    println("Méthodes : HB, PT, NRPT, Wolff, Wang-Landau")
    println("=========================================================\n")

    # Exécutions séquentielles synchronisées
    run_hb_complet!(hb, Q, m_field, n_burnin, n_sample)
    run_wolff_complet!(wolff, Q, m_field, n_burnin, n_sample)
    run_wl_complet!(wl, Q, m_field, n_burnin, n_sample)
    
    algo_pt_complet!(pt, Q, m_field, n_burnin, n_sample)
    algo4_nrpt_complet!(nrpt, Q, m_field, nb_tune, n_sample)

    println("\nGénération du Dashboard Global...")
    
    final_dash = create_dashboard(hb, pt, nrpt, wolff, wl, L, n_sample)
    
    display(final_dash)
    savefig(final_dash, "dashboard_N$(nb_chains)_all_algos.png")
    
    return hb, pt, nrpt, wolff, wl
end

run_simulation()