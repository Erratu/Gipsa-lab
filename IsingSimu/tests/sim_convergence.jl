using Plots
using Statistics
using IsingSimu
gr()

function create_convergence_dashboard(pt, nrpt, wolff, wl, L, beta_cible)
    p_mag = plot(wolff.history.obs_hist[1], label="Wolff", color=:purple, alpha=0.6, title="Convergence Magnétisation", xlabel="Scans", ylabel="|M|")
    plot!(p_mag, pt.history.obs_hist[end], label="PT", color=:red, alpha=0.8)
    plot!(p_mag, nrpt.history.obs_hist[end], label="NRPT", color=:blue, alpha=0.8, linestyle=:dash)
    
    p_ene = plot(wolff.history.ene_hist[1] ./ (L^2), label="Wolff", color=:purple, alpha=0.6, title="Thermalisation Énergie", xlabel="Scans", ylabel="E/N")
    plot!(p_ene, pt.history.ene_hist[end] ./ (L^2), label="PT", color=:red, alpha=0.8)
    plot!(p_ene, nrpt.history.ene_hist[end] ./ (L^2), label="NRPT", color=:blue, alpha=0.8, linestyle=:dash)
    
    p_wl = plot(wl.history.ene_hist[1] ./ (L^2), label="WL", color=:orange, title="Exploration Wang-Landau", xlabel="Scans", ylabel="E/N")
    
    l = @layout [
        a b
        c
    ]
    
    dash = plot(p_mag, p_ene, p_wl, layout=l, size=(1000, 700), margin=5Plots.mm)
    return dash
end

function run_convergence(L=20, n_tune=1000, n_sample=2000, beta_min=0.0, beta_max=0.8, m_val=0.0, nb_chains=15)
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

    # Calcul exact du nombre de scans pour que les courbes fassent la même longueur
    actual_tune_scans = (2^floor(Int, log2(n_tune))) - 1
    total_scans = actual_tune_scans + n_sample

    @info "Lancement Convergence : $total_scans scans enregistrés dès la première itération."

    algo_pt_complet!(sampler_pt, state_pt, e_func, u_func_hb, obs_func, 0, total_scans)
    algo_wolff_complet!(Q, sampler_wolff, state_wolff, e_func, obs_func, 0, total_scans)
    algo_wl_complet!(sampler_wl, state_wl, e_func, obs_func, 0, total_scans, n_sites=n_sites)

    algo4_nrpt_complet!(sampler_nrpt, state_nrpt, e_func, u_func_hb, obs_func, n_tune, n_sample, record_tuning=true)

    final_dash = create_convergence_dashboard(state_pt, state_nrpt, state_wolff, state_wl, L, beta_max)
    
    display(final_dash)
    mkpath("graphiques")
    savefig(final_dash, "graphiques/convergence_algos.png")
end

run_convergence()