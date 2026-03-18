using LinearAlgebra 
using Random 
import GLMakie as GL 
using JuMP
import Ipopt
using Zygote
using Printf 
using Optim
using IsingSimu 

function thomson_energy(X::Matrix{Float64}) 
    n = size(X,2) 
    E = 0.0 
    for i in 1:n-1, j in i+1:n 
        E += 1 / norm(X[:,i] - X[:,j]) 
    end 
    return E 
end 

function project_sphere!(X::Matrix{Float64})
    for i in 1:size(X, 2)
        X[:, i] ./= norm(X[:, i])
    end
    return X
end

function random_point_sphere() 
    v = randn(3) 
    return v / norm(v) 
end 

function random_configuration(n) 
    return hcat([random_point_sphere() for _ in 1:n]...) 
end

function delta_energy(X::Matrix{Float64}, i::Int, new_point::Vector{Float64}) 
    n = size(X,2) 
    old = X[:,i] 
    ΔE = 0.0 
    for j in 1:n 
        if j == i 
            continue 
        end 
        ΔE += 1 / norm(new_point - X[:,j]) 
        ΔE -= 1 / norm(old - X[:,j]) 
    end 
    return ΔE 
end 

function thomson_update!(X::Matrix{Float64}, beta::Float64; step_size=0.08) 
    n = size(X,2) 
    for _ in 1:n 
        i = rand(1:n) 
        old = X[:,i] # [cite: 3]
        
        proposal = old + step_size * randn(3) # [cite: 3]
        proposal /= norm(proposal) # [cite: 3]
        
        ΔE = delta_energy(X, i, proposal) # [cite: 3]
        
        if ΔE <= 0 || rand() < exp(-beta * ΔE) 
            X[:,i] = proposal 
        end 
    end 
    return X 
end 

# PGD
function solve_pgd(X0::Matrix{Float64}; lr=0.01, max_iter=2000)
    X = copy(X0)
    history = Float64[]
    for _ in 1:max_iter
        push!(history, thomson_energy(X))
        G = Zygote.gradient(thomson_energy, X)[1]
        X .-= lr .* G
        project_sphere!(X)
    end
    return X, history
end

# PGD avec Momentum
function solve_pgd_momentum(X0::Matrix{Float64}; lr=0.01, beta=0.9, max_iter=2000)
    X = copy(X0)
    V = zeros(size(X))
    history = Float64[]
    for _ in 1:max_iter
        push!(history, thomson_energy(X))
        G = Zygote.gradient(thomson_energy, X)[1]
        V .= beta .* V .+ (1 - beta) .* G
        X .-= lr .* V
        project_sphere!(X)
    end
    return X, history
end

# L-BFGS pénalisé : peut-être changer le lambda au fil des itérations ?
function solve_lbfgs_penalized(X0::Matrix{Float64}; lambda=100.0, max_iter=3000)
    n = size(X0, 2)
    
    # fonction objectif pour Optim (prend un vecteur 1D)
    function f(x_vec)
        X = reshape(x_vec, 3, n)
        E = thomson_energy(X)
        penalty = sum(lambda * (norm(X[:, i])^2 - 1.0)^2 for i in 1:n)
        return E + penalty
    end
    
    # gradient
    function g!(G_vec, x_vec)
        G_vec .= Zygote.gradient(f, x_vec)[1]
        return G_vec
    end
    
    # L-BFGS
    options = Optim.Options(iterations=max_iter, store_trace=true, extended_trace=true)
    x0_vec = vec(copy(X0))
    res = optimize(f, g!, x0_vec, LBFGS(), options)
    
    # config finale
    X_opt = reshape(Optim.minimizer(res), 3, n)
    project_sphere!(X_opt) # projection finale
    
    history = Float64[]
    for t in Optim.trace(res)
        x_iter = reshape(t.metadata["x"], 3, n)
        push!(history, thomson_energy(x_iter))
    end
    
    return X_opt, history
end

# JuMP avec Ipopt : point intérieur
function solve_jump_ipopt(X0::Matrix{Float64})
    n = size(X0, 2)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    @variable(model, x[1:3, 1:n])
    for i in 1:3, j in 1:n
        set_start_value(x[i,j], X0[i,j])
    end
    
    @constraint(model, [j=1:n], x[1,j]^2 + x[2,j]^2 + x[3,j]^2 == 1.0)
    
    @objective(model, Min, sum(
        1.0 / sqrt((x[1,i] - x[1,j])^2 + (x[2,i] - x[2,j])^2 + (x[3,i] - x[3,j])^2) 
        for i in 1:n-1, j in i+1:n
    ))
    
    optimize!(model)
    return value.(x), Float64[] # Pas d'historique
end

# PT
function solve_pt(n::Int; n_burnin=4000, n_sample=10000)
    n_replicas = 16 
    betas = collect(range(0.05, 15.0, length=n_replicas)) 
    sampler = PTSampler(betas) 
    
    initial_states = [random_configuration(n) for _ in 1:n_replicas] 
    state = PTState{Matrix{Float64}}(initial_states) 
    
    e_func = X -> thomson_energy(X) 
    u_func = (X, b) -> thomson_update!(X, b, step_size=0.08) 
    obs_func = X -> thomson_energy(X) 
    
    algo_pt_complet!(sampler, state, e_func, u_func, obs_func, n_burnin, n_sample) 
    
    best_X = state.sims[end] 
    return best_X, Float64[] 
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
    
    u_vals = range(0, 2π, length=40)
    v_vals = range(0, π, length=40)
    x_sphere = [cos(u) * sin(v) for u in u_vals, v in v_vals]
    y_sphere = [sin(u) * sin(v) for u in u_vals, v in v_vals]
    z_sphere = [cos(v) for u in u_vals, v in v_vals]

    color_matrix = fill((:lightblue, 0.2), size(z_sphere))

    # Visualisation 3D de la config finale pour chaque méthode
    col = 1
    for (name, X_final) in results_dict
        ax_3d = GL.Axis3(fig[2, col], title=name, aspect = :data)
        
        # Sphère
        GL.surface!(ax_3d, x_sphere, y_sphere, z_sphere, color=color_matrix, transparency=true)
        
        # Points
        GL.scatter!(ax_3d, X_final[1,:], X_final[2,:], X_final[3,:], 
            markersize = 12, color = :red, strokewidth = 1, strokecolor = :black)
        
        col += 1
    end
    
    GL.display(fig)
    return fig
end

function main()
    n = 20 
    @info "Résolution du problème de Thomson sur la sphère pour n = $n" 
    
    Random.seed!(1234)
    X0 = random_configuration(n)
    
    results = Dict{String, Matrix{Float64}}()
    histories = Dict{String, Vector{Float64}}()
    
    # PGD
    X_pgd, hist_pgd = solve_pgd(X0, lr=0.005, max_iter=2000)
    results["PGD"] = X_pgd
    histories["PGD"] = hist_pgd
    @printf("PGD : %.6f\n", thomson_energy(X_pgd))
    
    # PGD Momentum
    X_mom, hist_mom = solve_pgd_momentum(X0, lr=0.005, max_iter=2000)
    results["Momentum"] = X_mom
    histories["Momentum"] = hist_mom
    @printf("Momentum : %.6f\n", thomson_energy(X_mom))
    
    # L-BFGS pénalisé
    X_pen, hist_pen = solve_lbfgs_penalized(X0, lambda=200.0, max_iter=3000)
    results["L-BFGS Pénalisé"] = X_pen
    histories["L-BFGS Pénalisé"] = hist_pen
    @printf("L-BFGS Pénalisé : %.6f\n", thomson_energy(X_pen))
    
    # JuMP avec Ipopt
    X_jump, _ = solve_jump_ipopt(X0)
    results["JuMP/Ipopt"] = X_jump
    @printf("JuMP / Ipopt : %.6f\n", thomson_energy(X_jump))
    
    # Parallel Tempering
    println("\nLancement du Parallel Tempering...")
    X_pt, _ = solve_pt(n)
    results["Parallel Tempering"] = X_pt
    @printf("PT (MCMC) : %.6f\n", thomson_energy(X_pt))

    fig = plot_all_results(results, histories, n)
    
    GL.save("thomson_comparaison_n$(n).png", fig)
end

main()