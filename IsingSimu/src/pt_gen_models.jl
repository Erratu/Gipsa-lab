function algo_pt_complet!(sampler::PTSampler, state::PTState{T}, energy_func::Function, local_update!::Function, obs_func::Function, n_burnin::Int, n_sample::Int) where T
    total_scans = n_burnin + n_sample
    @info "PT (Burn-in: $n_burnin, Sample: $n_sample)"
    
    for t in 1:total_scans
        for s in 1:sampler.nb_sim
            state.sims[s] = local_update!(state.sims[s], sampler.betas[s])
        end

        for _ in 1:sampler.nb_sim
            i = rand(1:sampler.nb_sim-1)
            E1 = energy_func(state.sims[i])
            E2 = energy_func(state.sims[i+1])
            alpha = min(1.0, exp((sampler.betas[i] - sampler.betas[i+1]) * (E1 - E2)))

            if rand() < alpha
                state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
            end
        end

        if t == n_burnin
            state.metrics.round_trips = 0
            empty_history!(state.history)
        end

        if t > n_burnin
            for i in 1:sampler.nb_sim
                idx = state.indices[i]
                if i == sampler.nb_sim
                    state.metrics.is_going_down[idx] = true
                elseif i == 1 && state.metrics.is_going_down[idx]
                    state.metrics.is_going_down[idx] = false
                    state.metrics.round_trips += 1
                end
            end

            for s in 1:sampler.nb_sim
                push!(state.history.obs_hist[s], obs_func(state.sims[s]))
                push!(state.history.ene_hist[s], energy_func(state.sims[s]))
                push!(state.history.trajectoires[s], findfirst(==(s), state.indices))
            end
        end
    end
    @info "PT classique terminé : Round trips : $(state.metrics.round_trips)"
end

function algo1_deo_complet!(sampler::PTSampler, state::PTState{T}, energy_func::Function, local_update!::Function, obs_func::Function, n_scan::Int; record_history::Bool=false) where T
    fill!(state.metrics.swap_rejections, 0.0)

    for t in 1:n_scan
        for s in 1:sampler.nb_sim
            state.sims[s] = local_update!(state.sims[s], sampler.betas[s])
        end

        offset = (t % 2 == 0) ? 2 : 1
        
        for i in 1:(sampler.nb_sim - 1)
            E1 = energy_func(state.sims[i])
            E2 = energy_func(state.sims[i+1])
            alpha = min(1.0, exp((sampler.betas[i] - sampler.betas[i+1]) * (E1 - E2)))

            state.metrics.swap_rejections[i] += (1.0 - alpha)

            if i % 2 == offset % 2
                if rand() < alpha
                    state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                    state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
                end
            end
        end
        
        for i in 1:sampler.nb_sim
            idx = state.indices[i]
            if i == sampler.nb_sim
                state.metrics.is_going_down[idx] = true
            elseif i == 1 && state.metrics.is_going_down[idx]
                state.metrics.is_going_down[idx] = false
                state.metrics.round_trips += 1
            end
        end

        if record_history
            for s in 1:sampler.nb_sim
                push!(state.history.obs_hist[s], obs_func(state.sims[s]))
                push!(state.history.ene_hist[s], energy_func(state.sims[s]))
                push!(state.history.trajectoires[s], findfirst(==(s), state.indices))
            end
        end
    end

    state.metrics.swap_rejections ./= n_scan
end

function algo2_updateschedule(Lambda_func, current_betas::Vector{Float64})
    N = length(current_betas)
    beta_min = current_betas[1]
    beta_max = current_betas[end]
    
    Lambda_total = Lambda_func(beta_max) 
    
    if Lambda_total <= 1e-8
        return copy(current_betas)
    end
    
    new_betas = zeros(Float64, N)
    for k in 0:(N-1)
        target = (k / (N - 1)) * Lambda_total
        new_betas[k+1] = find_beta_inverse(Lambda_func, target, beta_min, beta_max)
    end
    
    new_betas[1] = beta_min
    new_betas[end] = beta_max
    return new_betas
end

function find_beta_inverse(Lambda_func, target::Float64, beta_min::Float64, beta_max::Float64; tol=1e-8)
    low, high = beta_min, beta_max
    if target <= 0.0 return beta_min end
    if target >= Lambda_func(beta_max) return beta_max end
    
    while (high - low) > tol
        mid = (low + high) / 2
        if Lambda_func(mid) < target low = mid else high = mid end
    end
    return (low + high) / 2
end

function algo3_communicationbarrier(rejections::Vector{Float64}, betas::Vector{Float64})
    nb_sim = length(betas)
    Lambda_hat_points = zeros(Float64, nb_sim)
    Lambda_hat_points[1] = 0.0
    for i in 2:nb_sim
        Lambda_hat_points[i] = Lambda_hat_points[i-1] + rejections[i-1]
    end
    return interpolate(betas, Lambda_hat_points, FritschCarlsonMonotonicInterpolation())
end

function algo4_nrpt_complet!(sampler::PTSampler, state::PTState{T}, energy_func, local_update!, obs_func, n_tune::Int, n_sample::Int; record_tuning::Bool=false) where T
    maxRound = floor(Int, log2(n_tune))
    n_scans_per_round = 1 
    
    @info "NRPT (Tuning: $n_tune, Sample: $n_sample)"
    
    for r in 1:maxRound
        algo1_deo_complet!(sampler, state, energy_func, local_update!, obs_func, n_scans_per_round, record_history=record_tuning)
        
        Lambda_func = algo3_communicationbarrier(state.metrics.swap_rejections, sampler.betas)
        sampler.betas = algo2_updateschedule(Lambda_func, sampler.betas)
        
        n_scans_per_round *= 2
        Lambda_total = Lambda_func(sampler.betas[end])
        @info "Round $r/$maxRound | Scans: $(n_scans_per_round/2) | Lambda total: $(round(Lambda_total, digits=2))"
    end
    
    Lambda_final = algo3_communicationbarrier(state.metrics.swap_rejections, sampler.betas)(sampler.betas[end])
    N_optimal = floor(Int, 2 * Lambda_final)
    
    @info "Lambda final estimé : $(round(Lambda_final, digits=2))"
    @info "Nombre de chaînes optimal recommandé (N*) : $N_optimal"
    
    # On n'efface l'historique que si on ne veut pas voir le tuning
    if !record_tuning
        state.metrics.round_trips = 0
        empty_history!(state.history)
    end
    
    @info "Echantillonnage (n_sample = $n_sample)"
    algo1_deo_complet!(sampler, state, energy_func, local_update!, obs_func, n_sample, record_history=true)
    
    @info "NRPT terminé. Round trips réalisés : $(state.metrics.round_trips)"
end