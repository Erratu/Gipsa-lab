using LinearAlgebra
using Random
using Printf
import GLMakie as GL
using IsingSimu

using Optim
using ADTypes: AutoForwardDiff

function random_point_sphere()
    v = randn(3)
    return v / norm(v)
end

function thomson_energy(X::Matrix{Float64})
    n = size(X,2)
    E = 0.0
    for i in 1:n-1, j in i+1:n
        E += 1 / norm(X[:,i] - X[:,j])
    end
    return E
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
        old = X[:,i]

        # proposal gaussienne + projection sphère
        proposal = old + step_size * randn(3)
        proposal /= norm(proposal)

        ΔE = delta_energy(X, i, proposal)

        if ΔE <= 0 || rand() < exp(-beta * ΔE)
            X[:,i] = proposal
        end
    end
    
    return X
end

function random_configuration(n)
    return hcat([random_point_sphere() for _ in 1:n]...)
end

function plot_configuration(X)
    # Utilisation explicite de l'espace de nom GL
    fig = GL.Figure(resolution = (800, 800))
    ax = GL.Axis3(fig[1,1], title = "Problème de Thomson (n=$(size(X,2)))")

    GL.scatter!(ax,
        X[1,:],
        X[2,:],
        X[3,:],
        markersize = 15,
        color = :blue,
        strokewidth = 1,
        strokecolor = :black
    )

    GL.display(fig)
    return fig
end

function quench_lbfgs_cart(X::Matrix{Float64})
    @info "Quenching avec descente de gradient (L-BFGS)"
    n = size(X, 2)
    
    x_init = vec(X) 
    
    function objective(x_flat)
        X_mat = reshape(x_flat, 3, n)
        E = 0.0
        
        for i in 1:n-1
            for j in i+1:n
                vi = X_mat[:, i] / norm(X_mat[:, i])
                vj = X_mat[:, j] / norm(X_mat[:, j])
                E += 1.0 / norm(vi - vj)
            end
        end
        return E
    end
    
    result = optimize(objective, x_init, LBFGS(); autodiff = AutoForwardDiff())
    
    x_opt_flat = Optim.minimizer(result)
    
    X_opt = reshape(x_opt_flat, 3, n)
    for i in 1:n
        X_opt[:, i] /= norm(X_opt[:, i])
    end
    
    @info "L-BFGS terminé."
    @info "Détails de convergence : $(Optim.converged(result)) en $(Optim.iterations(result)) itérations."
    
    return X_opt
end

function quench_to_zero(X::Matrix{Float64}; max_steps=5000, initial_step=0.01, final_step=1e-7)
    @info "Quenching avec diminution du pas"
    X_opt = copy(X)
    n = size(X_opt, 2)
    
    decay_rate = (final_step / initial_step)^(1.0 / max_steps)
    current_step = initial_step
    
    for _ in 1:max_steps
        for _ in 1:n
            i = rand(1:n)
            old = X_opt[:,i]
            
            proposal = old + current_step * randn(3)
            proposal /= norm(proposal)
            
            if delta_energy(X_opt, i, proposal) < 0
                X_opt[:,i] = proposal
            end
        end
        current_step *= decay_rate 
    end
    return X_opt
end

function quench_lbfgs_sph(X_init::Matrix{Float64})
    @info "Quenching avec L-BFGS (Coordonnées Sphériques)"
    n = size(X_init, 2)
    
    angles_init = zeros(2 * n)
    for i in 1:n
        x, y, z = X_init[1, i], X_init[2, i], X_init[3, i]
        angles_init[2*i - 1] = acos(clamp(z, -1.0, 1.0)) # theta
        angles_init[2*i]     = atan(y, x)                # phi
    end
    
    function objective(angles)
        E = 0.0
        for i in 1:n-1
            theta_i = angles[2*i - 1]
            phi_i   = angles[2*i]
            # Les coordonnées (x,y,z) sont implicitement sur la sphère de rayon 1
            xi = sin(theta_i) * cos(phi_i)
            yi = sin(theta_i) * sin(phi_i)
            zi = cos(theta_i)
            
            for j in i+1:n
                theta_j = angles[2*j - 1]
                phi_j   = angles[2*j]
                xj = sin(theta_j) * cos(phi_j)
                yj = sin(theta_j) * sin(phi_j)
                zj = cos(theta_j)
                
                # Distance euclidienne
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                dist = sqrt(dx^2 + dy^2 + dz^2)
                E += 1.0 / dist
            end
        end
        return E
    end
    
    result = optimize(objective, angles_init, LBFGS(); autodiff = AutoForwardDiff())
    
    angles_opt = Optim.minimizer(result)
    
    X_opt = zeros(3, n)
    for i in 1:n
        theta = angles_opt[2*i - 1]
        phi   = angles_opt[2*i]
        X_opt[1, i] = sin(theta) * cos(phi)
        X_opt[2, i] = sin(theta) * sin(phi)
        X_opt[3, i] = cos(theta)
    end
    
    @info "L-BFGS terminé."
    @info "Détails : $(Optim.converged(result)) en $(Optim.iterations(result)) itérations."
    
    return X_opt
end

function main()
    n = 20  # nombre de charges
    @info "Résolution du problème de Thomson pour n = $n avec Parallel Tempering"

    # Configuration du Sampler
    n_replicas = 16
    beta_min = 0.05
    beta_max = 15.0
    betas = collect(range(beta_min, beta_max, length=n_replicas))
    
    sampler = PTSampler(betas)

    # Initialisation de l'État
    Random.seed!(1234)
    initial_states = [random_configuration(n) for _ in 1:n_replicas]
    
    state = PTState{Matrix{Float64}}(initial_states)

    # Paramétrage des fonctions du modèle
    e_func = X -> thomson_energy(X)
    u_func = (X, b) -> thomson_update!(X, b, step_size=0.08)
    obs_func = X -> thomson_energy(X) 

    # Simulation
    n_burnin = 4_000
    n_sample = 10_000
    n_tune = 5_000
    
    algo_pt_complet!(sampler, state, e_func, u_func, obs_func, n_burnin, n_sample) #150.88156954
    #algo4_nrpt_complet!(sampler, state, e_func, u_func, obs_func, n_tune, n_sample) #150.88156833

    best_X = quench_lbfgs_sph(state.global_best_X)
    best_E = thomson_energy(best_X)
    print(size(best_X, 2))
    print(all(x -> isapprox(norm(x), 1.0), eachcol(best_X)))
    @info @sprintf("Énergie finale après quenching lbfgs: %.8f", best_E)

    # Visualisation et Sauvegarde
    fig = plot_configuration(best_X)
    
    filename = "graphiques/thomson_n$(n)_result.png"
    GL.save(filename, fig)
    @info "Graphique sauvegardé sous : $filename"
end

main()