using IsingSimu
using Printf 
using LinearAlgebra 
using Random 
import GLMakie as GL 
using Zygote

function thomson_energy(X::Matrix{Float64}) 
    n = size(X,2) 
    E = 0.0 
    for i in 1:n-1, j in i+1:n 
        E += 1 / norm(X[:,i] - X[:,j]) 
    end 
    return E 
end 

function random_point_sphere() 
    v = randn(3) 
    return v / norm(v) 
end 

function random_configuration(n) 
    return hcat([random_point_sphere() for _ in 1:n]...) 
end

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

function take_random_subset(Y::Matrix{Float64})
    n = size(Y, 2)
    
    # q(X) uniforme
    idx_to_remove = rand(1:n)
    indices_to_keep = [1:idx_to_remove-1; idx_to_remove+1:n]
    X = Y[:, indices_to_keep]
    removed_point = Y[:, idx_to_remove]
    
    return X, removed_point, idx_to_remove
end

function sgd_loo(lr=0.01,max_iter=2000)
    X = random_configuration(n+1)
    n = size(X, 2)
    history = Float64[]
    decay_rate = 0.999
    current_lr = lr
    
    for iter in 1:max_iter
        batch_idx = randperm(n)[1:n]
        
        grad = Zygote.gradient(thomson_energy, X)[1]
        
        X[:, batch_idx] .-= current_lr .* grad[:, batch_idx]

        for i in batch_idx
            X[:, i] ./= norm(X[:, i])
        end

        current_lr *= decay_rate
    end
    
    # regarder l'énergie de tous les subsets, et sélectionner ?

    return X, history
end

function main()
    n = 20 
    @info "Résolution du problème de Thomson sur la sphère pour n = $n" 
    
    Random.seed!(1234)
    
    results = Dict{String, Matrix{Float64}}()
    histories = Dict{String, Vector{Float64}}()
    times = Dict{String, Float64}()
    
    # leave-one-out
    t_pgd = @elapsed begin X_sgd_loo, hist_sgd_loo = sgd_loo() end
    results["LOO"] = X_sgd_loo
    histories["LOO"] = hist_sgd_loo
    times["LOO"] = t_pgd
    @printf("LOO : %.6f\n", thomson_energy(X_sgd_loo))

    min_global_theorique = 150.881568
    println(@sprintf("| %-15s | %-12s | %-12s | %-10s |", "Méthode", "Énergie", "Δ au Global", "Temps (s)"))

    order = ["LOO"]
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