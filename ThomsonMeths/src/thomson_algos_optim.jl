import Manifolds as Manif

# PGD (avec points fixés) avec gradient manuel
function solve_pgd_fast(X0::Matrix{Float64}; lr::Float64=0.01, max_iter::Int=2000)
    X::Matrix{Float64} = copy(X0)
    G::Matrix{Float64} = similar(X)
    history::Vector{Float64} = Float64[]
    sizehint!(history, max_iter)
    
    for _ in 1:max_iter
        E::Float64 = thomson_energy_and_grad!(G, X)
        push!(history, E)
        
        # Mise à jour sans allocation avec @.
        @. X -= lr * G
        
        project_sphere!(X)
    end
    return X, history
end

# PGD avec Momentum (avec points fixés) optimisé
function solve_pgd_momentum_fast(X0::Matrix{Float64}; lr::Float64=0.01, beta::Float64=0.9, max_iter::Int=2000)
    X::Matrix{Float64} = copy(X0)
    V::Matrix{Float64} = zeros(size(X))
    G::Matrix{Float64} = similar(X)
    history::Vector{Float64} = Float64[]
    sizehint!(history, max_iter)
    for _ in 1:max_iter
        E::Float64 = thomson_energy_and_grad!(G, X)
        push!(history, E)
        V .= beta .* V .+ (1 - beta) .* G
        X .-= lr .* V
        project_sphere!(X)
    end
    return X, history
end

# PT avec quench optimisé
function solve_pt_fast(n::Int; n_burnin::Int=500, n_sample::Int=2000, n_restarts::Int=5)
    best_E = Inf
    best_overall_X = zeros(3, n)

    for _ in 1:n_restarts
        n_replicas::Int = 32 
        betas::Vector{Float64} = collect(range(0.01, 20.0, length=n_replicas)) 
        sampler::PTSampler = PTSampler(betas) 
        
        initial_states::Vector{Matrix{Float64}} = [random_configuration(n) for _ in 1:n_replicas] 
        state::PTState{Matrix{Float64}} = PTState{Matrix{Float64}}(initial_states) 
        
        e_func = (X::Matrix{Float64}) -> thomson_energy_fast(X) 
        u_func = (X::Matrix{Float64}, b::Float64) -> thomson_update_fast!(X, b, step_size=0.15) 
        obs_func = (X::Matrix{Float64}) -> thomson_energy_fast(X) 
        
        algo_pt_complet!(sampler, state, e_func, u_func, obs_func, n_burnin, n_sample) 

        best_X::Matrix{Float64}, _ = solve_manifold_lbfgs_fast(state.global_best_X)
        E_val = thomson_energy_fast(best_X)
        if E_val < best_E
            best_E = E_val
            best_overall_X = copy(best_X)
        end
    end

    return best_overall_X, Float64[] 
end

# NRPT avec quench optimisé
function solve_nrpt_fast(n::Int; n_burnin::Int=1000, n_sample::Int=3000, n_restarts::Int=5)
    best_E = Inf
    best_overall_X = zeros(3, n)

    for _ in 1:n_restarts
        n_replicas::Int = 26 
        betas::Vector{Float64} = collect(range(0.01, 20.0, length=n_replicas)) 
        sampler::PTSampler = PTSampler(betas) 
        
        initial_states::Vector{Matrix{Float64}} = [random_configuration(n) for _ in 1:n_replicas] 
        state::PTState{Matrix{Float64}} = PTState{Matrix{Float64}}(initial_states) 
        
        e_func = (X::Matrix{Float64}) -> thomson_energy_fast(X) 
        u_func = (X::Matrix{Float64}, b::Float64) -> thomson_update_fast!(X, b, step_size=0.15) 
        obs_func = (X::Matrix{Float64}) -> thomson_energy_fast(X) 
        
        algo4_nrpt_complet!(sampler, state, e_func, u_func, obs_func, n_burnin, n_sample) 

        best_X::Matrix{Float64}, _ = solve_manifold_lbfgs_fast(state.global_best_X)
        E_val = thomson_energy_fast(best_X)
        if E_val < best_E
            best_E = E_val
            best_overall_X = copy(best_X)
        end
    end

    return best_overall_X, Float64[] 
end

function solve_basin_hopping_fast(n::Int; n_steps::Int=500, temperature::Float64=0.05, amplitude_kick::Float64=0.3)
    @info "Starting optimized Basin-Hopping (N=$n, Steps=$n_steps)"
    
    # Initialisation et premier quench profond
    X_current::Matrix{Float64} = random_configuration(n)
    X_current, _ = solve_manifold_lbfgs_fast(X_current, max_iter=2000)
    E_current::Float64 = thomson_energy_fast(X_current)
    
    X_best::Matrix{Float64} = copy(X_current)
    E_best::Float64 = E_current
    
    X_trial::Matrix{Float64} = similar(X_current)
    history::Vector{Float64} = Float64[]
    sizehint!(history, n_steps)
    
    acceptances::Int = 0
    
    for step::Int in 1:n_steps
        # Kick ciblé
        X_trial .= X_current
        
        # On ne secoue que 1 à 3 particules au hasard
        num_to_kick::Int = rand(1:3) 
        for _ in 1:num_to_kick
            idx::Int = rand(1:n)
            # Perturbation en place
            X_trial[1, idx] += amplitude_kick * randn()
            X_trial[2, idx] += amplitude_kick * randn()
            X_trial[3, idx] += amplitude_kick * randn()
            
            # Reprojection rapide de la particule modifiée
            @inbounds begin
                nrm::Float64 = sqrt(X_trial[1, idx]^2 + X_trial[2, idx]^2 + X_trial[3, idx]^2)
                X_trial[1, idx] /= nrm
                X_trial[2, idx] /= nrm
                X_trial[3, idx] /= nrm
            end
        end
        
        # Quench lâche (Exploration rapide)
        X_trial, _ = solve_manifold_lbfgs_fast(X_trial, max_iter=40)
        E_trial::Float64 = thomson_energy_fast(X_trial)
        
        # Acceptation (Metropolis)
        delta_E::Float64 = E_trial - E_current
        
        if delta_E < 0 || rand() < exp(-delta_E / temperature)
            X_current .= X_trial
            E_current = E_trial
            acceptances += 1
            
            # Quench
            if E_current < E_best + 0.05 # Si on est très proche du record
                X_deep::Matrix{Float64}, _ = solve_manifold_lbfgs_fast(X_current, max_iter=1000)
                E_deep::Float64 = thomson_energy_fast(X_deep)
                
                if E_deep < E_best
                    E_best = E_deep
                    X_best .= X_deep
                    @info "Step $step: New record ! E = $(round(E_best, digits=6))"
                end
                # On met à jour l'état courant avec la géométrie parfaite
                X_current .= X_deep 
                E_current = E_deep
            end
        end
        
        push!(history, E_current)
    end
    
    taux::Float64 = acceptances / n_steps
    @info "Basin-Hopping finished. Acceptance rate : $(round(taux * 100, digits=1))%"
    
    return X_best, history
end

function solve_manifold_lbfgs_fast(X0::Matrix{Float64}; max_iter::Int=2000)
    n::Int = size(X0, 2)
    M = Manif.PowerManifold(Manif.Sphere(2), Manif.ArrayPowerRepresentation(), n)

    G_buffer::Matrix{Float64} = similar(X0)
    # fonction objectif
    f = (M, X::Matrix{Float64}) -> thomson_energy_fast(X)

    function grad_f(M, X::Matrix{Float64})
        thomson_energy_and_grad!(G_buffer, X)
        return Manif.project(M, X, G_buffer) 
    end

    state = quasi_Newton(M, f, grad_f, copy(X0);
        memory_size = 15,
        stopping_criterion = StopAfterIteration(max_iter) | StopWhenGradientNormLess(1e-6),
        record = [:Cost],
        return_state = true
    )

    X_opt::Matrix{Float64} = get_solver_result(state)
    history::Vector{Float64} = get_record(state)
    
    return X_opt, history
end