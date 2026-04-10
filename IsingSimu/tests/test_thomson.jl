import GLMakie as GL 
using Printf 
using Random
using LinearAlgebra
using IsingSimu

function align_smart_symmetry!(X::Matrix{Float64})
    n = size(X, 2)
    
    # Point ayant l'énergie locale max
    energies_locales = zeros(n)
    for i in 1:n
        for j in 1:n
            if i != j
                energies_locales[i] += 1.0 / norm(X[:, i] .- X[:, j])
            end
        end
    end
    
    idx_p1 = argmax(energies_locales)
    p1 = X[:, idx_p1]
    
    # Rotation pour mettre p1 au Nord
    z_axis = [0.0, 0.0, 1.0]
    v = cross(p1, z_axis)
    c = dot(p1, z_axis)
    
    if c < -0.999999 
        X[1, :] .*= -1.0
        X[3, :] .*= -1.0
    elseif c < 0.999999 
        V_mat = [  0.0  -v[3]   v[2];
                  v[3]   0.0  -v[1];
                 -v[2]   v[1]   0.0 ]
        R1 = I + V_mat + (V_mat * V_mat) * (1.0 / (1.0 + c))
        X .= R1 * X
    end
    
    # Trouver le voisin le plus proche du nouveau pôle Nord (pour la 2ème rotation)
    distances_au_nord = [norm(X[:, i] .- [0.0, 0.0, 1.0]) for i in 1:n]
    distances_au_nord[idx_p1] = Inf # Ignorer le point lui-même
    idx_p2 = argmin(distances_au_nord)
    p2 = X[:, idx_p2]
    
    # Rotation azimutale pour p2
    angle_phi = atan(p2[2], p2[1])
    cos_p, sin_p = cos(-angle_phi), sin(-angle_phi)
    R2 = [cos_p -sin_p 0.0;
          sin_p  cos_p 0.0;
           0.0    0.0  1.0]
           
    X .= R2 * X
    
    # Reprojection finale propre
    for i in 1:n
        @views nrm = norm(X[:, i])
        @views X[:, i] ./= nrm
    end
    
    return X
end

function plot_all_results(results_dict, histories_dict, n)
    fig = GL.Figure(size = (1600, 900))
    
    # Énergie
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

    col = 1
    for (name, X_final) in results_dict
        
        # On aligne la matrice X_final
        X_aligned = copy(X_final)
        align_smart_symmetry!(X_aligned)

        ax_3d = GL.Axis3(fig[2, col], title=name, aspect = :data, azimuth=pi/4, elevation=pi/6)
        
        GL.surface!(ax_3d, x_sphere, y_sphere, z_sphere, color= opaque_gray, transparency=false)
        
        GL.scatter!(ax_3d, X_aligned[1,:], X_aligned[2,:], X_aligned[3,:], 
            markersize = 15, color = :red, strokewidth = 1, strokecolor = :black)
        
        # Colorer le point 1 (au pôle Nord) en bleu pour vérifier l'alignement
        GL.scatter!(ax_3d, [X_aligned[1,1]], [X_aligned[2,1]], [X_aligned[3,1]], 
            markersize = 20, color = :blue)

        col += 1
    end
    
    screen = GL.Screen()
    GL.display(screen, fig)
    
    return fig, screen
end

function main()
    n = 188 
    @info "Résolution du problème de Thomson sur la sphère pour n = $n" 
    
    Random.seed!(1234)
    X0 = random_configuration(n)
    
    results = Dict{String, Matrix{Float64}}()
    histories = Dict{String, Vector{Float64}}()
    times = Dict{String, Float64}()
    
    # PGD
    t_pgd = @elapsed begin X_pgd, hist_pgd = solve_pgd_fast(X0, lr=0.005, max_iter=2000) end
    results["PGD"] = X_pgd
    histories["PGD"] = hist_pgd
    times["PGD"] = t_pgd
    @printf("PGD : %.6f\n", thomson_energy_fast(X_pgd))
    
    # PGD Momentum
    t_mom = @elapsed begin X_mom, hist_mom = solve_pgd_momentum_fast(X0, lr=0.005, max_iter=2000) end
    results["Momentum"] = X_mom
    histories["Momentum"] = hist_mom
    times["Momentum"] = t_mom
    @printf("Momentum : %.6f\n", thomson_energy_fast(X_mom))
    
    # L-BFGS pénalisé
    #t_pen = @elapsed begin X_pen, hist_pen = solve_lbfgs_penalized(X0, lambda=200.0, max_iter=3000) end
    #results["L-BFGS Pénalisé"] = X_pen
    #histories["L-BFGS Pénalisé"] = hist_pen
    #times["L-BFGS Pénalisé"] = t_pen
    #@printf("L-BFGS Pénalisé : %.6f\n", thomson_energy_fast(X_pen))
    
    # JuMP avec Ipopt
    t_jump = @elapsed begin X_jump, _ = solve_jump_ipopt(X0) end
    results["JuMP/Ipopt"] = X_jump
    times["JuMP/Ipopt"] = t_jump
    @printf("JuMP Ipopt : %.6f\n", thomson_energy_fast(X_jump))
    
    # Parallel Tempering
    println("\nLancement du Parallel Tempering")
    t_pt = @elapsed begin X_pt, _ = solve_pt_fast(n) end
    results["Parallel Tempering"] = X_pt
    times["Parallel Tempering"] = t_pt
    @printf("PT (avec quench) : %.6f\n", thomson_energy_fast(X_pt))

    # NRPT
    println("\nLancement du NRPT")
    t_nrpt = @elapsed begin X_nrpt, _ = solve_nrpt_fast(n) end
    results["NRPT"] = X_nrpt
    times["NRPT"] = t_nrpt
    @printf("NRPT (avec quench) : %.6f\n", thomson_energy_fast(X_nrpt))

    # Basin Hopping
    #println("\nLancement du Basin Hopping")
    #t_bh = @elapsed begin X_bh, _ = solve_basin_hopping_fast(n) end
    #results["Basin Hopping"] = X_bh
    #times["Basin Hopping"] = t_bh
    #@printf("BH (avec quench) : %.6f\n", thomson_energy_fast(X_bh))

    # Manifold L-BFGS
    println("Lancement Manifold L-BFGS")
    t_man = @elapsed begin X_man, hist_man = solve_manifold_lbfgs_fast(X0, max_iter=2000) end
    results["Manifold L-BFGS"] = X_man
    histories["Manifold L-BFGS"] = hist_man
    times["Manifold L-BFGS"] = t_man

    min_global_theorique = get_min_thomson_energy(n)
    println(@sprintf("| %-15s | %-12s | %-12s | %-10s |", "Méthode", "Énergie", "Δ au Global", "Temps (s)"))

    order = ["PGD", "Momentum", "JuMP/Ipopt", "Parallel Tempering", "NRPT", "Manifold L-BFGS"]
    for name in order
        E = thomson_energy_fast(results[name])
        delta = E - min_global_theorique
        t = times[name]
        println(@sprintf("| %-20s | %12.6f | %12.6f | %10.3f |", name, E, delta, t))
    end

    #fig, screen = plot_all_results(results, histories, n)
    #
    #GL.save("graphiques/thomson_comparaison_n$(n).png", fig)
    #wait(screen)
end

main()

# @profview solve_pt_fast(46)