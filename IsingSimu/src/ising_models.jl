function wolff_update!(Q_in, X::Vector{Int}, beta::Float64)
    Q = sparse(Q_in)
    n_sites = size(Q, 1)
    p_add = 1.0 - exp(-2.0 * beta)
    seed = rand(1:n_sites)
    stack = [seed]
    spin_val = X[seed]
    X[seed] *= -1 
    
    while !isempty(stack)
        current = pop!(stack)
        for v_idx in Q.colptr[current]:(Q.colptr[current+1]-1)
            voisin = Q.rowval[v_idx]
            if X[voisin] == spin_val && rand() < p_add
                X[voisin] *= -1
                push!(stack, voisin)
            end
        end
    end
    return X
end

function wl_flip(n_sites, X::Vector{Int})
    site = rand(1:n_sites)
    X[site] *= -1
    return (X_ref) -> (X_ref[site] *= -1) 
end

function algo_wolff_complet!(Q, sampler::PTSampler, state::PTState{T}, energy_func, obs_func, n_burnin::Int, n_sample::Int) where T
    total_scans = n_burnin + n_sample
    @info "Wolff (Burn-in: $n_burnin, Sample: $n_sample)"
    
    for t in 1:total_scans
        state.sims[1] = wolff_update!(Q, state.sims[1], sampler.betas[1])
        
        if t == n_burnin
            empty_history!(state.history)
        end
        
        if t > n_burnin
            push!(state.history.obs_hist[1], obs_func(state.sims[1]))
            push!(state.history.ene_hist[1], energy_func(state.sims[1]))
        end
    end
    @info "Wolff terminé"
end

function algo_wl_complet!(sampler::PTSampler, state::PTState{T}, energy_func, obs_func, n_burnin::Int, n_sample::Int; n_sites::Int=1) where T
    total_scans = n_burnin + n_sample
    
    ln_g = Dict{Float64, Float64}() 
    H = Dict{Float64, Int}()        
    log_f = 1.0                     
    
    E_current = energy_func(state.sims[1])
    ln_g[E_current] = 0.0
    H[E_current] = 1
    
    @info "Wang-Landau (Scans: $total_scans)"
    
    for t in 1:total_scans
        for _ in 1:n_sites
            revert_func = wl_flip(n_sites, state.sims[1]) 
            E_new = energy_func(state.sims[1])
            
            g_old = get(ln_g, E_current, 0.0)
            g_new = get(ln_g, E_new, 0.0)
            
            if rand() < exp(g_old - g_new)
                E_current = E_new
            else
                revert_func(state.sims[1]) 
            end
            
            ln_g[E_current] = get(ln_g, E_current, 0.0) + log_f
            H[E_current] = get(H, E_current, 0) + 1
        end
        
        if t % 1000 == 0 && length(H) > 5
            h_vals = values(H)
            if minimum(h_vals) > 0.8 * (sum(h_vals) / length(H))
                log_f /= 2.0
                empty!(H)
            end
        end
        
        if t == n_burnin
            empty_history!(state.history)
        end
        
        if t > n_burnin
            push!(state.history.obs_hist[1], obs_func(state.sims[1]))
            push!(state.history.ene_hist[1], E_current)
        end
    end
    @info "Wang-Landau terminé"
end