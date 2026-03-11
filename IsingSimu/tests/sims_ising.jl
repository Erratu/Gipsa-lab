using IsingSimu
using Plots
using Statistics
gr()

function create_dashboard(pt, nrpt, wolff, wl, L, beta_cible)
    # Wolff 
    p_mag = plot(wolff.history.obs_hist[1], label="Wolff", color=:purple, alpha=0.4, title="Magnétisation (β=$beta_cible)", xlabel="Scans", ylabel="|M|")
    # PT et NRPT (en pointillés)
    plot!(p_mag, pt.history.obs_hist[end], label="PT", color=:red, alpha=0.8)
    plot!(p_mag, nrpt.history.obs_hist[end], label="NRPT", color=:blue, alpha=0.8, linestyle=:dash)
    
    # Energie
    p_ene = plot(wolff.history.ene_hist[1] ./ (L^2), label="Wolff", color=:purple, alpha=0.4, title="Énergie moy (β=$beta_cible)", xlabel="Scans", ylabel="E/N")
    plot!(p_ene, pt.history.ene_hist[end] ./ (L^2), label="PT", color=:red, alpha=0.8)
    plot!(p_ene, nrpt.history.ene_hist[end] ./ (L^2), label="NRPT", color=:blue, alpha=0.8, linestyle=:dash)
    
    # Wang-Landau
    p_wl = plot(wl.history.ene_hist[1] ./ (L^2), label="WL (Exploration)", color=:orange, title="Marche aléatoire Wang-Landau", xlabel="Scans", ylabel="E/N")
    
    # Trajectoires de températures
    p_tr_nrpt = plot(nrpt.history.trajectoires[1], title="Trajectoire NRPT (Chaîne 1)", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:blue, linewidth=1.5)
    p_tr_pt = plot(pt.history.trajectoires[1], title="Trajectoire PT (Chaîne 1)", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:red, linewidth=1.5)
    
    # Round Trips
    noms = ["NRPT", "PT Classique"]
    valeurs_rt = [nrpt.metrics.round_trips, pt.metrics.round_trips]
    p_rt = bar(noms, valeurs_rt, title="Comparaison Round Trips", ylabel="Count", legend=false, color=[:blue, :red], bar_width=0.4)
    
    # Mise en page sur 3 lignes et 2 colonnes
    l = @layout [
        a b
        c d
        e f
    ]
    
    dash = plot(p_tr_nrpt, p_tr_pt, p_mag, p_ene, p_rt, p_wl, layout=l, size=(1000, 900), margin=5Plots.mm)
    return dash
end

function run_benchmark(L=20, n_tune=1000, n_sample=3000, beta_min=0.0, beta_max=0.8, m_val=0.0, nb_chains=15)
    n_sites = L * L
    Q = adjacency_2d(L, periodic=true)
    m_field = fill(m_val, n_sites)
    
    betas = collect(range(beta_min, beta_max, length=nb_chains))
    initial_states_multi = [rand([-1, 1], n_sites) for _ in 1:nb_chains]
    initial_states_mono  = [rand([-1, 1], n_sites)] 
    
    e_func = X -> energy(X, Q, m_field)
    obs_func = X -> abs(sum(X)) / n_sites
    u_func_hb = (X, b) -> begin heat_bath_ising!(X, Q, m_field, b, n_sites); return X end
    
    sampler_pt = PTSampler(copy(betas))
    state_pt = PTState{Vector{Int}}(copy.(initial_states_multi))
    
    sampler_nrpt = PTSampler(copy(betas))
    state_nrpt = PTState{Vector{Int}}(copy.(initial_states_multi))
    
    sampler_wolff = PTSampler([beta_max])
    state_wolff = PTState{Vector{Int}}(copy.(initial_states_mono))
    
    sampler_wl = PTSampler([beta_max])
    state_wl = PTState{Vector{Int}}(copy.(initial_states_mono))

    maxRound = floor(Int, log2(n_tune))
    n_burnin_equivalent = (2^maxRound) - 1 

    @info "Lancement Global : $n_burnin_equivalent scans burn-in, suivis de $n_sample scans échantillonnage."
    @info "Méthodes : PT, NRPT, Wolff, Wang-Landau"

    algo_pt_complet!(sampler_pt, state_pt, e_func, u_func_hb, obs_func, n_burnin_equivalent, n_sample)
    algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func_hb, obs_func, n_tune, n_sample)
    algo_wolff_complet!(Q, sampler_wolff, state_wolff, e_func, obs_func, n_burnin_equivalent, n_sample)
    algo_wl_complet!(sampler_wl, state_wl, e_func, obs_func, n_burnin_equivalent, n_sample, n_sites=n_sites)

    @info "Génération du Dashboard Global"
    final_dash = create_dashboard(state_pt, state_nrpt, state_wolff, state_wl, L, beta_max)
    
    display(final_dash)
    mkpath("graphiques")
    savefig(final_dash, "graphiques/dashboard_N$(nb_chains)_all_algos.png")
end

run_benchmark()