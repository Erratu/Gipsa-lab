using LinearAlgebra
using Random
using Printf
import GLMakie as GL
using IsingSimu

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
    
    algo_pt_complet!(sampler, state, e_func, u_func, obs_func, n_burnin, n_sample)

    # Résultat final
    best_X = state.sims[end]
    best_E = thomson_energy(best_X)

    @info @sprintf("Énergie finale (chaîne la plus froide) : %.8f", best_E)

    # Visualisation et Sauvegarde
    fig = plot_configuration(best_X)
    
    filename = "thomson_n$(n)_result.png"
    GL.save(filename, fig)
    @info "Graphique sauvegardé sous : $filename"
end

main()