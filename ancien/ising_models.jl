include("ising_model_2.jl")
using Interpolations

function run_hb!(state::ChainState, Q, m_field)
    n = state.L^2
    heat_bath_ising!(state.sims[1], Q, m_field, state.betas[1], n)
    push!(state.mag_hist[1], abs(sum(state.sims[1])/n))
    push!(state.ene_hist[1], energy(state.sims[1], Q, m_field)/n)
end

function run_hb_complet!(state::ChainState, Q, m_field, n_burnin, n_sample)
    n_sites = state.L^2
    total_scans = n_burnin + n_sample
    
    println("=== DEBUT HEAT BATH (Burn-in: $n_burnin, Sample: $n_sample) ===")
    
    for t in 1:total_scans
        # Exploration locale uniquement
        heat_bath_ising!(state.sims[1], Q, m_field, state.betas[1], n_sites)
        
        # Remise à zéro à la fin du burn-in
        if t == n_burnin
            empty!(state.mag_hist[1])
            empty!(state.ene_hist[1])
        end
        
        # phase d'échantillonnage
        if t > n_burnin
            push!(state.mag_hist[1], abs(sum(state.sims[1])/n_sites))
            push!(state.ene_hist[1], energy(state.sims[1], Q, m_field)/n_sites)
        end
    end
    println("Heat Bath terminé !")
end

function run_pt!(state::ChainState, Q, m_field)
    n = state.L^2
    for s in 1:state.nb_sim
        heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n)
    end
    
    for _ in 1:state.nb_sim
        i = rand(1:state.nb_sim-1)
        E1 = energy(state.sims[i], Q, m_field)
        E2 = energy(state.sims[i+1], Q, m_field)
        
        if rand() < exp((state.betas[i] - state.betas[i+1]) * (E1 - E2))
            state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
            state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
        end
    end
    
    for i in 1:state.nb_sim
        idx = state.indices[i]
        if i == state.nb_sim
            state.is_going_down[idx] = true
        elseif i == 1 && state.is_going_down[idx]
            state.is_going_down[idx] = false
            state.round_trips += 1
        end
    end
    
    for s in 1:state.nb_sim
        push!(state.mag_hist[s], abs(sum(state.sims[s])/n))
        push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n)
        push!(state.trajectoires[s], findfirst(==(s), state.indices))
    end
end

function algo_pt_complet!(state::ChainState, Q, m_field, n_burnin, n_sample)
    n = state.L^2
    total_scans = n_burnin + n_sample
    
    println("=== DEBUT PT CLASSIQUE (Burn-in: $n_burnin, Sample: $n_sample) ===")
    
    for t in 1:total_scans
        # Exploration locale
        for s in 1:state.nb_sim
            heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n)
        end
        
        # swaps
        for _ in 1:state.nb_sim
            i = rand(1:state.nb_sim-1)
            E1 = energy(state.sims[i], Q, m_field)
            E2 = energy(state.sims[i+1], Q, m_field)
            
            # Probabilité d'acceptation du swap
            alpha = min(1.0, exp((state.betas[i] - state.betas[i+1]) * (E1 - E2)))
            
            if rand() < alpha
                state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
            end
        end
        
        # Remise à zéro
        if t == n_burnin
            state.round_trips = 0
            
            # On vide complètement les tableaux
            empty!(state.mag_hist)
            empty!(state.ene_hist)
            empty!(state.trajectoires)
            
            # On les recrée vides pour le sampling
            for s in 1:state.nb_sim
                push!(state.mag_hist, Float64[])
                push!(state.ene_hist, Float64[])
                push!(state.trajectoires, Int[])
            end
        end
        
        # Enregistrement
        if t > n_burnin
            # Mise à jour des Round Trips
            for i in 1:state.nb_sim
                idx = state.indices[i]
                if i == state.nb_sim
                    state.is_going_down[idx] = true
                elseif i == 1 && state.is_going_down[idx]
                    state.is_going_down[idx] = false
                    state.round_trips += 1
                end
            end
            
            # Sauvegarde des historiques
            for s in 1:state.nb_sim
                push!(state.mag_hist[s], abs(sum(state.sims[s])/n))
                push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n)
                push!(state.trajectoires[s], findfirst(==(s), state.indices))
            end
        end
    end
    
    println("PT classique terminé ! Round trips : $(state.round_trips)")
end

function algo1_deo!(state::ChainState, Q, m_field, t)
    n = state.L^2
    for s in 1:state.nb_sim
        heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n)
    end

    offset = (t % 2 == 0) ? 2 : 1
    
    for i in 1:(state.nb_sim - 1)
        E1 = energy(state.sims[i], Q, m_field)
        E2 = energy(state.sims[i+1], Q, m_field)
        alpha = min(1.0, exp((state.betas[i] - state.betas[i+1]) * (E1 - E2)))
        
        state.swap_rejections[i] += (1.0 - alpha)
        
        if i % 2 == offset % 2
            if rand() < alpha
                state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
            end
        end
    end

    for i in 1:state.nb_sim
        idx = state.indices[i]
        if i == state.nb_sim
            state.is_going_down[idx] = true
        elseif i == 1 && state.is_going_down[idx]
            state.is_going_down[idx] = false
            state.round_trips += 1
        end
    end

    for s in 1:state.nb_sim
        push!(state.mag_hist[s], abs(sum(state.sims[s])/n))
        push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n)
        push!(state.trajectoires[s], findfirst(==(s), state.indices))
    end
end

function algo1_deo_complet!(state::ChainState, Q, m_field, n_scan)
    n_sites = state.L^2
    fill!(state.swap_rejections, 0.0)

    for t in 1:n_scan
        for s in 1:state.nb_sim
            heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n_sites)
        end

        offset = (t % 2 == 0) ? 2 : 1
        
        for i in 1:(state.nb_sim - 1)
            E1 = energy(state.sims[i], Q, m_field)
            E2 = energy(state.sims[i+1], Q, m_field)
            delta_beta = state.betas[i+1] - state.betas[i]
            alpha = min(1.0, exp(delta_beta * (E1 - E2)))

            state.swap_rejections[i] += (1.0 - alpha)

            if i % 2 == offset % 2
                if rand() < alpha
                    state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                    state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
                end
            end
        end
        
        for i in 1:state.nb_sim
            idx = state.indices[i]
            if i == state.nb_sim
                state.is_going_down[idx] = true
            elseif i == 1 && state.is_going_down[idx]
                state.is_going_down[idx] = false
                state.round_trips += 1
            end
        end

        for s in 1:state.nb_sim
            push!(state.mag_hist[s], abs(sum(state.sims[s])/n_sites))
            push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n_sites)
            push!(state.trajectoires[s], findfirst(==(s), state.indices))
        end
    end

    state.swap_rejections ./= n_scan
    
    return state.swap_rejections
end

function algo2_updateschedule(Lambda_func, N, beta_max)
    Lambda_total = Lambda_func(beta_max) 
    new_betas = zeros(Float64, N)
    
    for k in 0:(N-1)
        target = (k / (N - 1)) * Lambda_total
        new_betas[k+1] = find_beta_inverse(Lambda_func, target, beta_max)
    end
    
    new_betas[1] = 0.0
    new_betas[end] = beta_max
    return new_betas
end

function find_beta_inverse(Lambda_func, target, beta_max; tol=1e-8)
    low, high = 0.0, beta_max
    if target <= 0.0 return 0.0 end
    if target >= Lambda_func(beta_max) return beta_max end
    
    while (high - low) > tol
        mid = (low + high) / 2
        if Lambda_func(mid) < target low = mid else high = mid end
    end
    return (low + high) / 2
end

function algo3_communicationbarrier(state::ChainState)
    nb_sim = state.nb_sim
    betas = state.betas
    
    # Calcul des points S = (beta_i, Lambda_hat_i)
    Lambda_hat_points = zeros(Float64, nb_sim)
    Lambda_hat_points[1] = 0.0
    for i in 2:nb_sim
        Lambda_hat_points[i] = Lambda_hat_points[i-1] + state.swap_rejections[i-1]
    end
    
    # Interpolation (Fritsch-Carlson)
    itp = interpolate(betas, Lambda_hat_points, FritschCarlsonMonotonicInterpolation())
    
    return itp
end

function algo4_nrpt!(state::ChainState, Q, m_field, t)
    algo1_deo!(state, Q, m_field, t)
    
    freq = 100
    if t % freq == 0
        state.swap_rejections ./= freq
        lambda_func = algo3_communicationbarrier(state)
        state.betas = algo2_updateschedule(lambda_func, state.nb_sim, state.betas[end])
        fill!(state.swap_rejections, 0.0)
    end
end

function algo4_nrpt_complet!(state::ChainState, Q, m_field, n_tune, n_sample)
    maxRound = floor(Int, log2(n_tune))
    n_scans_per_round = 1 
    
    println("n_tune = $n_tune")
    
    for r in 1:maxRound
        algo1_deo_complet!(state, Q, m_field, n_scans_per_round)
        
        Lambda_func = algo3_communicationbarrier(state)
        
        state.betas = algo2_updateschedule(Lambda_func, state.nb_sim, state.betas[end])
        
        n_scans_per_round *= 2
        
        Lambda_total = Lambda_func(state.betas[end])
        println("Round $r/$maxRound | Scans: $(n_scans_per_round/2) | Lambda total: $(round(Lambda_total, digits=2))")
    end
    
    Lambda_final = algo3_communicationbarrier(state)(state.betas[end])
    N_optimal = floor(Int, 2 * Lambda_final)
    
    println("Lambda final estimé : $(round(Lambda_final, digits=2))")
    println("Nombre de chaînes optimal recommandé (N*) : $N_optimal")
    println("Nombre de chaînes utilisé actuellement : $(state.nb_sim)")
    
    state.round_trips = 0
    empty!(state.rt_hist)
    
    println("Echantillonnage (n_sample = $n_sample)")
    algo1_deo_complet!(state, Q, m_field, n_sample)
    
    println("Echantillonnage terminé avec succès !")
    println("Round trips réalisés : $(state.round_trips)")
end

function algo_pt_complet!(state::ChainState, Q, m_field, n_burnin, n_sample)
    n_sites = state.L^2
    total_scans = n_burnin + n_sample

    println("=== DEBUT PT CLASSIQUE (Burn-in: $n_burnin, Sample: $n_sample) ===")
    
    for t in 1:total_scans
        # Exploration locale
        for s in 1:state.nb_sim
            heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n_sites)
        end

        # Swaps
        for _ in 1:state.nb_sim
            i = rand(1:state.nb_sim-1)
            E1 = energy(state.sims[i], Q, m_field)
            E2 = energy(state.sims[i+1], Q, m_field)
            delta_beta = state.betas[i+1] - state.betas[i]
            alpha = min(1.0, exp(delta_beta * (E1 - E2)))

            if rand() < alpha
                state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
            end
        end

        # Remise à zéro
        if t == n_burnin
            state.round_trips = 0
            empty!(state.mag_hist)
            empty!(state.ene_hist)
            empty!(state.trajectoires)
            for s in 1:state.nb_sim
                push!(state.mag_hist, Float64[])
                push!(state.ene_hist, Float64[])
                push!(state.trajectoires, Int[])
            end
        end

        # sample
        if t > n_burnin
            for i in 1:state.nb_sim
                idx = state.indices[i]
                if i == state.nb_sim
                    state.is_going_down[idx] = true
                elseif i == 1 && state.is_going_down[idx]
                    state.is_going_down[idx] = false
                    state.round_trips += 1
                end
            end

            for s in 1:state.nb_sim
                push!(state.mag_hist[s], abs(sum(state.sims[s])/n_sites))
                push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n_sites)
                push!(state.trajectoires[s], findfirst(==(s), state.indices))
            end
        end
    end
    println("PT classique terminé : Round trips : $(state.round_trips)")
end

function run_wolff_complet!(state::ChainState, Q, m_field, n_burnin, n_sample)
    n_sites = state.L^2
    total_scans = n_burnin + n_sample
    beta = state.betas[1]
    
    # Probabilité d'ajouter un voisin au cluster (pour J=1)
    p_add = 1.0 - exp(-2.0 * beta) 
    
    println("WOLFF (Burn-in: $n_burnin, Sample: $n_sample)")
    
    for t in 1:total_scans
        # 1 scan = 1 itération de cluster
        seed = rand(1:n_sites)
        cluster = [seed]
        stack = [seed]
        spin_val = state.sims[1][seed]
        
        # On flip le seed
        state.sims[1][seed] *= -1 
        
        # Construction du cluster
        while !isempty(stack)
            current = pop!(stack)
            
            # Récupération des voisins via la matrice creuse Q
            voisins = Q.colptr[current]:(Q.colptr[current+1]-1)
            for v_idx in voisins
                voisin = Q.rowval[v_idx]
                # Si le voisin a le même spin initial, on tente de l'ajouter
                if state.sims[1][voisin] == spin_val
                    if rand() < p_add
                        state.sims[1][voisin] *= -1
                        push!(cluster, voisin)
                        push!(stack, voisin)
                    end
                end
            end
        end
        
        # Gestion des historiques
        if t == n_burnin
            empty!(state.mag_hist[1])
            empty!(state.ene_hist[1])
        end
        
        if t > n_burnin
            push!(state.mag_hist[1], abs(sum(state.sims[1])/n_sites))
            push!(state.ene_hist[1], energy(state.sims[1], Q, m_field)/n_sites)
        end
    end
    println("Wolff terminé")
end

function run_wl_complet!(state::ChainState, Q, m_field, n_burnin, n_sample)
    n_sites = state.L^2
    total_scans = n_burnin + n_sample
    
    ln_g = Dict{Float64, Float64}() # Log de la densité d'états
    H = Dict{Float64, Int}()        # Histogramme des énergies visitées
    log_f = 1.0                     # Facteur de modification initial (ln(e^1) = 1)
    
    E_current = energy(state.sims[1], Q, m_field)
    ln_g[E_current] = 0.0
    H[E_current] = 1
    
    println("WANG-LANDAU (Scans totaux: $total_scans)")
    
    for t in 1:total_scans
        # 1 scan = N tentatives de flip
        for _ in 1:n_sites
            site = rand(1:n_sites)
            state.sims[1][site] *= -1
            E_new = energy(state.sims[1], Q, m_field)
            
            g_old = get(ln_g, E_current, 0.0)
            g_new = get(ln_g, E_new, 0.0)
            
            # Acceptation basée sur la densité d'états (et non la température)
            if rand() < exp(g_old - g_new)
                E_current = E_new
            else
                state.sims[1][site] *= -1 # Rejet
            end
            
            ln_g[E_current] = get(ln_g, E_current, 0.0) + log_f
            H[E_current] = get(H, E_current, 0) + 1
        end
        
        # Vérification simplifiée de l'aplatissement de l'histogramme
        if t % 1000 == 0 && length(H) > 5
            h_vals = values(H)
            if minimum(h_vals) > 0.8 * (sum(h_vals)/length(H))
                log_f /= 2.0
                empty!(H) # Reset de l'histogramme
            end
        end
        
        if t == n_burnin
            empty!(state.mag_hist[1])
            empty!(state.ene_hist[1])
        end
        
        if t > n_burnin
            push!(state.mag_hist[1], abs(sum(state.sims[1])/n_sites))
            push!(state.ene_hist[1], E_current/n_sites)
        end
    end
    println("Wang-Landau terminé")
end