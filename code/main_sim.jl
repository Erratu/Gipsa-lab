using Plots
gr()

include("ising_structures.jl")
include("ising_models.jl")

# nb_it est maintenant le nombre TOTAL d'itérations (ex: 5000)
# nb_tune est la portion allouée au tuning (ex: les 1000 premières)
function run_simulation(L=20, nb_it=4000, nb_tune=1000, m_val=0.02, nb_chains=30)
    Q = adjacency_2d(L, periodic=true)
    m_field = fill(m_val, L*L)
    
    betas = collect(range(0.0, 1.0, length=nb_chains))
    
    hb = ChainState(L, [1.0])
    pt = ChainState(L, copy(betas))
    nrpt = ChainState(L, copy(betas))

    println("Lancement : $nb_tune itérations de tuning, suivies de $(nb_it - nb_tune) itérations d'exploitation.")

    for t in 1:nb_it
        run_hb!(hb, Q, m_field)
        run_pt!(pt, Q, m_field)
        
        # NRPT adapte ses températures seulement pendant la phase de tuning
        if t <= nb_tune
            algo4_nrpt!(nrpt, Q, m_field, t)
        else
            algo1_deo!(nrpt, Q, m_field, t) 
        end

        # On enregistre le round trip rate en continu depuis t=1
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
    
    p_h_pt = heatmap(reshape(pt.sims[end], L, L), c=:binary, title="PT", legend=false, axis=false, clims=(-1, 1))
    p_h_nrpt = heatmap(reshape(nrpt.sims[end], L, L), c=:binary, title="NRPT", legend=false, axis=false, clims=(-1, 1))

    return plot(p_mag, p_ene, p_tr_pt, p_tr_nrpt, p_rt, p_h_pt, p_h_nrpt, layout=(3, 3), size=(1000, 800))
end

run_simulation()