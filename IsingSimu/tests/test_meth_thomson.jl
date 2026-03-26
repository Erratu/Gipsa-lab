using IsingSimu
using Printf 
using LinearAlgebra 
using Random 
import GLMakie as GL 
using Zygote
using Combinatorics
using JuMP
using Ipopt

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

function matrice_interactions(X)
    m = size(X, 2)
    H = zeros(m, m)
    for i in 1:m, j in (i+1):m
        dist = norm(X[:, i] - X[:, j])
        H[i, j] = H[j, i] = 1.0 / dist
    end
    return H
end

function evaluer_energie(H, indices)
    return 0.5 * sum(H[i, j] for i in indices, j in indices)
end

# Recherche Exhaustive (Exacte mais lente)
function methode_exhaustive(H, n)
    m = size(H, 1)
    if binomial(BigInt(m), BigInt(n)) > 100_000
        return Int[], Inf
    end
    
    E_min = Inf
    best_idx = Int[]
    
    for indices in combinations(1:m, n)
        E_curr = evaluer_energie(H, indices)
        if E_curr < E_min
            E_min = E_curr
            best_idx = indices
        end
    end
    return best_idx, E_min
end

# Glouton + Recherche Locale
function methode_heuristique(H, n)
    m = size(H, 1)
    
    # Init gloutonne
    S = [1]
    U = Set(2:m)
    while length(S) < n
        best_u, min_add_E = -1, Inf
        for u in U
            add_E = sum(H[u, s] for s in S)
            if add_E < min_add_E
                min_add_E, best_u = add_E, u
            end
        end
        push!(S, best_u)
        delete!(U, best_u)
    end
    
    U_arr = collect(U)
    
    # Swaps
    improved = true
    while improved
        improved = false
        best_delta, best_s_idx, best_u_idx = 0.0, 0, 0
        
        for (i, s) in enumerate(S), (j, u) in enumerate(U_arr)
            delta_E = sum(H[u, x] - H[s, x] for x in S if x != s)
            if delta_E < best_delta - 1e-7
                best_delta, best_s_idx, best_u_idx = delta_E, i, j
            end
        end
        
        if best_delta < -1e-7
            S[best_s_idx], U_arr[best_u_idx] = U_arr[best_u_idx], S[best_s_idx]
            improved = true
        end
    end
    
    return S, evaluer_energie(H, S)
end

# Relaxation Continue (JuMP + Ipopt avec Spectral Shift)
function methode_relaxation_jump(H, n)
    m = size(H, 1)
    
    eigvals_H = eigvals(Symmetric(H))
    lambda_shift = minimum(eigvals_H) < 0 ? abs(minimum(eigvals_H)) + 1e-3 : 0.0
    H_psd = H + lambda_shift * I
    
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    @variable(model, 0 <= epsilon[1:m] <= 1)
    @constraint(model, sum(epsilon) == n)
    @objective(model, Min, 0.5 * sum(epsilon[i] * H_psd[i, j] * epsilon[j] for i in 1:m, j in 1:m) - 0.5 * lambda_shift * sum(epsilon))
    
    optimize!(model)
    
    eps_opt = value.(epsilon)
    best_idx = sortperm(eps_opt, rev=true)[1:n]
    
    return best_idx, evaluer_energie(H, best_idx)
end

# Champ Moyen (Inspiré de Belief Propagation)
function methode_champ_moyen(H, n; T_init=10.0, T_final=0.01, decay=0.9)
    m = size(H, 1)
    v = fill(n / m, m) # Croyances initiales (probabilité uniforme)
    
    T = T_init
    while T > T_final
        # Calcul du champ local agissant sur chaque point
        h_field = H * v 
        
        # Trouver le potentiel mu garantissant sum(v) = n
        mu_min, mu_max = minimum(h_field) - 5*T, maximum(h_field) + 5*T
        mu = 0.0
        for _ in 1:40
            mu = (mu_min + mu_max) / 2
            sum_v = sum(1.0 / (1.0 + exp((h_field[i] - mu) / T)) for i in 1:m)
            sum_v > n ? (mu_max = mu) : (mu_min = mu)
        end
        
        # Mise à jour des croyances (avec un peu d'inertie)
        v_new = [1.0 / (1.0 + exp((h_field[i] - mu) / T)) for i in 1:m]
        v = 0.5 * v .+ 0.5 * v_new 
        
        T *= decay
    end
    
    best_idx = sortperm(v, rev=true)[1:n]
    return best_idx, evaluer_energie(H, best_idx)
end

function comparer_methodes(m, n)
    println("Comparaison pour m=$m, n=$n")
    
    X = randn(3, m)
    X = X ./ sqrt.(sum(X.^2, dims=1))
    H = matrice_interactions(X)
    
    # Recherche exhaustive (uniquement si raisonnable)
    if binomial(BigInt(m), BigInt(n)) <= 100_000
        t_ex = @elapsed idx, E = methode_exhaustive(H, n)
        println("Exhaustive : E = $(round(E, digits=5)) | Temps = $(round(t_ex, digits=4)) s")
    else
        println("Exhaustive : Ignorée (trop de combinaisons)")
    end
    
    # Heuristique
    t_heu = @elapsed idx, E = methode_heuristique(H, n)
    println("Heuristique: E = $(round(E, digits=5)) | Temps = $(round(t_heu, digits=4)) s")
    
    # Relaxation JuMP
    t_jump = @elapsed idx, E = methode_relaxation_jump(H, n)
    println("Relaxation : E = $(round(E, digits=5)) | Temps = $(round(t_jump, digits=4)) s")
    
    # Champ Moyen (Ising) -> Belief Propagation pour un graphe complet
    t_cm = @elapsed idx, E = methode_champ_moyen(H, n)
    println("Champ Moyen: E = $(round(E, digits=5)) | Temps = $(round(t_cm, digits=4)) s")
    println("-"^40)
end

comparer_methodes(21, 20) 
#comparer_methodes(50, 10) 
#comparer_methodes(148, 100) 

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