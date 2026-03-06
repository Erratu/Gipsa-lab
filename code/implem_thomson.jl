using LinearAlgebra
using Random
using Printf
using GLMakie

function random_point_sphere()
    v = randn(3)
    return v / norm(v)
end

function thomson_energy(X)
    n = size(X,2)
    E = 0.0
    for i in 1:n-1, j in i+1:n
        E += 1 / norm(X[:,i] - X[:,j])
    end
    return E
end

function delta_energy(X, i, new_point)
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

function metropolis_step!(X, beta, step_size)

    n = size(X,2)
    i = rand(1:n)

    old = X[:,i]

    # proposal gaussienne + projection sphère
    proposal = old + step_size * randn(3)
    proposal /= norm(proposal)

    ΔE = delta_energy(X, i, proposal)

    if ΔE <= 0 || rand() < exp(-beta * ΔE)
        X[:,i] = proposal
        return ΔE
    end

    return 0.0
end

function random_configuration(n)
    return hcat([random_point_sphere() for _ in 1:n]...)
end

function parallel_tempering_thomson(n;
        n_replicas = 16,
        beta_min = 0.05,
        beta_max = 15.0,
        step_size = 0.08,
        n_steps = 120_000,
        swap_every = 10,
        rng_seed=1234)

    Random.seed!(rng_seed)

    betas = collect(range(beta_min, beta_max, length=n_replicas))

    # initialisation
    replicas = [random_configuration(n) for _ in 1:n_replicas]
    energies = [thomson_energy(X) for X in replicas]

    best_E = minimum(energies)
    best_X = deepcopy(replicas[argmin(energies)])

    for step in 1:n_steps

        # Evolution locale
        for k in 1:n_replicas
            metropolis_step!(replicas[k], betas[k], step_size)
            energies[k] = thomson_energy(replicas[k])

            if energies[k] < best_E
                best_E = energies[k]
                best_X = deepcopy(replicas[k])
            end
        end

        # Swaps
        if step % swap_every == 0
            for k in 1:n_replicas-1

                Δ = (betas[k] - betas[k+1]) *
                    (energies[k+1] - energies[k])

                if rand() < exp(Δ)
                    replicas[k], replicas[k+1] =
                        replicas[k+1], replicas[k]

                    energies[k], energies[k+1] =
                        energies[k+1], energies[k]
                end
            end
        end

        # -------- Logging --------
        if step % 2000 == 0
            @printf("Step %-7d  Best energy = %.8f\n", step, best_E)
        end
    end

    return best_X, best_E
end

function plot_configuration(X)

    fig = Figure()
    ax = Axis3(fig[1,1])

    GLMakie.scatter!(ax,
        X[1,:],
        X[2,:],
        X[3,:],
        markersize = 12
    )

    fig
end

function main()
    n = 20   # nombre de charges

    println("Résolution du problème de Thomson pour n = $n")
    
    X, E = parallel_tempering_thomson(n)

    println("\n====================================")
    println("Énergie finale : ", E)
    println("====================================")

    # Capturer la figure
    plot_configuration(X)
end

main()
