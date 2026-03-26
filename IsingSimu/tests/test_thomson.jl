import GLMakie as GL 
using Printf 
using Random
using IsingSimu

function plot_all_results(results_dict, histories_dict, n)
    fig = GL.Figure(size = (1600, 900))
    
    # énergie
    ax_energy = GL.Axis(fig[1, 1:length(results_dict)], 
        title="Convergence de l'Énergie pour n=$n", 
        xlabel="Itérations", 
        ylabel="Énergie de Thomson")
    
    for (name, hist) in histories_dict
        if !isempty(hist)
            GL.lines!(ax_energy, hist, label=name, linewidth=2)
        end
    end
    GL.axislegend(ax_energy)
    
    r_sphere = 0.98
    u_vals = range(0, 2π, length=40)
    v_vals = range(0, π, length=40)
    x_sphere = [r_sphere * cos(u) * sin(v) for u in u_vals, v in v_vals]
    y_sphere = [r_sphere * sin(u) * sin(v) for u in u_vals, v in v_vals]
    z_sphere = [r_sphere * cos(v) for u in u_vals, v in v_vals]

    opaque_gray = fill(GL.RGBf(0.8, 0.8, 0.8), size(z_sphere))

    # Visualisation 3D de la config finale pour chaque méthode
    col = 1
    for (name, X_final) in results_dict
        ax_3d = GL.Axis3(fig[2, col], title=name, aspect = :data, azimuth=pi/4, elevation=pi/6)
        
        # Sphère
        GL.surface!(ax_3d, x_sphere, y_sphere, z_sphere, color= opaque_gray, transparency=false)
        
        # Points
        GL.scatter!(ax_3d, X_final[1,:], X_final[2,:], X_final[3,:], 
            markersize = 15, color = :red, strokewidth = 1, strokecolor = :black)
        
        col += 1
    end
    
    screen = GL.Screen()
    GL.display(screen, fig)
    
    return fig, screen
end

function main()
    n = 38 
    @info "Résolution du problème de Thomson sur la sphère pour n = $n" 
    
    Random.seed!(1234)
    X0 = random_configuration(n)
    
    results = Dict{String, Matrix{Float64}}()
    histories = Dict{String, Vector{Float64}}()
    times = Dict{String, Float64}()
    
    # PGD
    t_pgd = @elapsed begin X_pgd, hist_pgd = solve_pgd(X0, lr=0.005, max_iter=2000) end
    results["PGD"] = X_pgd
    histories["PGD"] = hist_pgd
    times["PGD"] = t_pgd
    @printf("PGD : %.6f\n", thomson_energy(X_pgd))
    
    # PGD Momentum
    t_mom = @elapsed begin X_mom, hist_mom = solve_pgd_momentum(X0, lr=0.005, max_iter=2000) end
    results["Momentum"] = X_mom
    histories["Momentum"] = hist_mom
    times["Momentum"] = t_mom
    @printf("Momentum : %.6f\n", thomson_energy(X_mom))
    
    # L-BFGS pénalisé
    t_pen = @elapsed begin X_pen, hist_pen = solve_lbfgs_penalized(X0, lambda=200.0, max_iter=3000) end
    results["L-BFGS Pénalisé"] = X_pen
    histories["L-BFGS Pénalisé"] = hist_pen
    times["L-BFGS Pénalisé"] = t_pen
    @printf("L-BFGS Pénalisé : %.6f\n", thomson_energy(X_pen))
    
    # JuMP avec Ipopt
    t_jump = @elapsed begin X_jump, _ = solve_jump_ipopt(X0) end
    results["JuMP/Ipopt"] = X_jump
    times["JuMP/Ipopt"] = t_jump
    @printf("JuMP Ipopt : %.6f\n", thomson_energy(X_jump))
    
    # Parallel Tempering
    println("\nLancement du Parallel Tempering")
    t_pt = @elapsed begin X_pt, _ = solve_pt(n) end
    results["Parallel Tempering"] = X_pt
    times["Parallel Tempering"] = t_pt
    @printf("PT (avec quench) : %.6f\n", thomson_energy(X_pt))

    # Manifold L-BFGS
    println("Lancement Manifold L-BFGS")
    t_man = @elapsed begin X_man, hist_man = solve_manifold_lbfgs(X0, max_iter=2000) end
    results["Manifold L-BFGS"] = X_man
    histories["Manifold L-BFGS"] = hist_man
    times["Manifold L-BFGS"] = t_man

    min_global_theorique = get_min_thomson_energy(n)
    println(@sprintf("| %-15s | %-12s | %-12s | %-10s |", "Méthode", "Énergie", "Δ au Global", "Temps (s)"))

    order = ["PGD", "Momentum", "L-BFGS Pénalisé", "JuMP/Ipopt", "Parallel Tempering", "Manifold L-BFGS"]
    for name in order
        E = thomson_energy(results[name])
        delta = E - min_global_theorique
        t = times[name]
        println(@sprintf("| %-20s | %12.6f | %12.6f | %10.3f |", name, E, delta, t))
    end

    fig, screen = plot_all_results(results, histories, n)
    
    GL.save("graphiques/thomson_comparaison_n$(n).png", fig)
    wait(screen)
end

main()