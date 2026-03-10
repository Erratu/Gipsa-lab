using Plots
gr()

include("ising_structures.jl")
include("ising_models.jl")

function run_simulation(L=20, nb_it=4000, nb_tune=1000, beta_min=0.0, beta_max=0.8, m_val=0.02, nb_chains=30)
    Q = adjacency_2d(L, periodic=true)
    m_field = fill(m_val, L*L)
    
    # On définit directement l'intervalle physique souhaité
    betas = collect(range(beta_min, beta_max, length=nb_chains))
    
    # La cible pour l'algorithme Heat Bath de référence est bien beta_max
    hb = ChainState(L, [beta_max])
    pt = ChainState(L, copy(betas))
    nrpt = ChainState(L, copy(betas))

    println("Lancement : $nb_tune itérations de tuning, suivies de $(nb_it - nb_tune) itérations d'exploitation.")
    println("Bornes des températures : de $beta_min à $beta_max avec $nb_chains chaînes.")

    for t in 1:nb_it
        run_hb!(hb, Q, m_field)
        run_pt!(pt, Q, m_field)
        
        if t <= nb_tune
            algo4_nrpt!(nrpt, Q, m_field, t)
        else
            algo1_deo!(nrpt, Q, m_field, t) 
        end

        push!(pt.rt_hist, pt.round_trips / t)
        push!(nrpt.rt_hist, nrpt.round_trips / t)

        if t % 100 == 0
            dash = create_dashboard(hb, pt, nrpt, L, t)
            display(dash)
        end
    end
    
    final_dash = create_dashboard(hb, pt, nrpt, L, nb_it)
    savefig(final_dash, "dashboard_N$(nb_chains)_final.png")
end

function create_dashboard(hb, pt, nrpt, L, t)
    p_mag = plot([hb.mag_hist[1], pt.mag_hist[end], nrpt.mag_hist[end]], 
                 labels=["HB" "PT" "NRPT"], title="Magnetisation", color=[:blue :green :red])
    
    p_ene = plot([hb.ene_hist[1], pt.ene_hist[end], nrpt.ene_hist[end]], 
                 labels=["HB" "PT" "NRPT"], title="Energie", color=[:blue :green :red])
    
    p_tr_pt = plot(pt.trajectoires[1], title="Traj PT", legend=false, color=:green)
    p_tr_nrpt = plot(nrpt.trajectoires[1], title="Traj NRPT", legend=false, color=:red)
    
    p_rt = plot(pt.rt_hist, label="PT", color=:green, title="Round Trip Rate")
    plot!(p_rt, nrpt.rt_hist, label="NRPT", color=:red)
    
    p_h_pt = heatmap(reshape(pt.sims[end], L, L), c=:binary, title="PT", legend=false, axis=false, clims=(-1,1))
    p_h_nrpt = heatmap(reshape(nrpt.sims[end], L, L), c=:binary, title="NRPT", legend=false, axis=false, clims=(-1,1))

    return plot(p_mag, p_ene, p_tr_pt, p_tr_nrpt, p_rt, p_h_pt, p_h_nrpt, layout=(3, 3), size=(1000, 800))
end

#run_simulation()


function run_and_compare_pt_nrpt(L, nb_sim, n_tune, n_sample, m_val, beta_max=0.8)
    Q = adjacency_2d(L, periodic=true)
    m_field = fill(m_val, L*L)
    betas_init = collect(range(0.0, beta_max, length=nb_sim))
    
    maxRound = floor(Int, log2(n_tune))
    n_burnin_equivalent = (2^maxRound) - 1
    
    println("PT et NRPT, Cible : beta = $beta_max")

    # RUN NRPT
    state_nrpt = ChainState(L, copy(betas_init))
    algo4_nrpt_complet!(state_nrpt, Q, m_field, n_tune, n_sample)
    
    # RUN PT
    state_pt = ChainState(L, copy(betas_init))
    algo_pt_complet!(state_pt, Q, m_field, n_burnin_equivalent, n_sample)
    
    println("RÉSULTATS DE LA COMPARAISON ($n_sample scans)")
    println("Round Trips (NRPT) : ", state_nrpt.round_trips)
    println("Round Trips (PT)   : ", state_pt.round_trips)
    ratio = state_pt.round_trips > 0 ? round(state_nrpt.round_trips / state_pt.round_trips, digits=2) : "Infini"
    println("Ratio d'efficacité (NRPT/PT) : ", ratio, "x")

    # GRAPHIQUES
    p1 = plot(state_nrpt.trajectoires[1], title="Trajectoire NRPT (Chaîne 1)", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:blue, linewidth=1.5)
    p2 = plot(state_pt.trajectoires[1], title="Trajectoire PT Classique (Chaîne 1)", xlabel="Scans", ylabel="Indice Temp", legend=false, color=:red, linewidth=1.5)

    # Titres mis à jour avec beta_max
    p_mag = plot(state_nrpt.mag_hist[end], label="NRPT", color=:blue, alpha=0.7, title="Magnétisation (Cible, β=$beta_max)", xlabel="Scans", ylabel="|M|")
    plot!(p_mag, state_pt.mag_hist[end], label="PT Classique", color=:red, alpha=0.7)

    p_ene = plot(state_nrpt.ene_hist[end], label="NRPT", color=:blue, alpha=0.7, title="Énergie (Cible, β=$beta_max)", xlabel="Scans", ylabel="E")
    plot!(p_ene, state_pt.ene_hist[end], label="PT Classique", color=:red, alpha=0.7)

    noms = ["NRPT", "PT Classique"]
    valeurs_rt = [state_nrpt.round_trips, state_pt.round_trips]
    p_rt = bar(noms, valeurs_rt, title="Nombre total de Round Trips", ylabel="Round Trips", legend=false, color=[:blue, :red], bar_width=0.5)

    l = @layout [
        a b
        c d
        e
    ]
    
    figure_finale = plot(p1, p2, p_mag, p_ene, p_rt, layout=l, size=(1000, 900), margin=5Plots.mm)
    savefig(figure_finale, joinpath(@__DIR__, "comparaison_PT_NRPT.png"))
    display(figure_finale)
    
    return figure_finale, state_nrpt, state_pt
end

fig, final_nrpt, final_pt = run_and_compare_pt_nrpt(20, 30, 1000, 4000, 0.02, 0.8)