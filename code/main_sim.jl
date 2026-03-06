using Plots
gr()

include("ising_structures.jl")
include("ising_models.jl")

function run_simulation(L=20, nb_it=4000, beta_max=0.8, m_val=0.02, nb_chains=10)
    Q = adjacency_2d(L, periodic=true)
    m_field = fill(m_val, L*L)
    
    betas = collect(range(0.1, beta_max, length=nb_chains))
    
    hb = ChainState(L, [beta_max])
    pt = ChainState(L, copy(betas))
    nrpt = ChainState(L, copy(betas))

    for t in 1:nb_it
        run_hb!(hb, Q, m_field)
        run_pt!(pt, Q, m_field)
        algo4_nrpt!(nrpt, Q, m_field, t)

        if t % 100 == 0
            dash = create_dashboard(hb, pt, nrpt, L)
            display(dash)
        end
    end
    
    final_dash = create_dashboard(hb, pt, nrpt, L)
    savefig(final_dash, "dashboard_N$(nb_chains)_final.png")
end

function create_dashboard(hb, pt, nrpt, L)
    p_mag = plot([hb.mag_hist[1], pt.mag_hist[end], nrpt.mag_hist[end]], 
                 labels=["HB" "PT" "NRPT"], title="Magnetisation", color=[:blue :green :red])
    
    p_ene = plot([hb.ene_hist[1], pt.ene_hist[end], nrpt.ene_hist[end]], 
                 labels=["HB" "PT" "NRPT"], title="Energie", color=[:blue :green :red])
    
    p_tr_pt = plot(pt.trajectoires[1], title="Traj PT", legend=false, color=:green)
    p_tr_nrpt = plot(nrpt.trajectoires[1], title="Traj NRPT", legend=false, color=:red)
    
    acc_pt = pt.swaps_accepted ./ max.(1, pt.swaps_attempted)
    acc_nrpt = nrpt.swaps_accepted ./ max.(1, nrpt.swaps_attempted)
    
    p_acc = plot(acc_pt, label="PT", marker=:circle, color=:green, title="Taux Acceptation", ylims=(0, 1.1))
    plot!(p_acc, acc_nrpt, label="NRPT", marker=:square, color=:red)

    p_h_pt = heatmap(reshape(pt.sims[end], L, L), c=:binary, title="PT Froid", ratio=1, axis=false, clims=(-1, 1))
    p_h_nrpt = heatmap(reshape(nrpt.sims[end], L, L), c=:binary, title="NRPT Froid", ratio=1, axis=false, clims=(-1, 1))
    
    l = @layout [grid(1,2) ; grid(1,2) ; grid(1,2)]
    return plot(p_mag, p_ene, p_tr_pt, p_tr_nrpt, p_acc, p_h_pt, p_h_nrpt, layout=l, size=(1000, 1000))
end

run_simulation(30, 10000, 0.8, 0.02, 30)