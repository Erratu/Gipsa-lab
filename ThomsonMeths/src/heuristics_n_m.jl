# Recherche Exhaustive (Exacte mais lente)
function methode_exhaustive(H, n)
    m = size(H, 1)
    if binomial(BigInt(m), BigInt(n)) > 1_000_000
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
    selected = [1]
    unselected = Set(2:m)
    while length(selected) < n
        best_u, min_add_E = -1, Inf
        for u in unselected
            add_E = sum(H[u, s] for s in selected)
            if add_E < min_add_E
                min_add_E, best_u = add_E, u
            end
        end
        push!(selected, best_u)
        delete!(unselected, best_u)
    end
    
    unselected_arr = collect(unselected)
    
    # Swaps
    improved = true
    while improved
        improved = false
        best_delta, best_s_idx, best_u_idx = 0.0, 0, 0
        
        for (i, s) in enumerate(selected), (j, u) in enumerate(unselected_arr)
            delta_E = sum(H[u, x] - H[s, x] for x in selected if x != s)
            if delta_E < best_delta - 1e-7
                best_delta, best_s_idx, best_u_idx = delta_E, i, j
            end
        end
        
        if best_delta < -1e-7
            selected[best_s_idx], unselected_arr[best_u_idx] = unselected_arr[best_u_idx], selected[best_s_idx]
            improved = true
        end
    end
    
    return selected, evaluer_energie(H, selected)
end

# Relaxation Continue (JuMP + Ipopt avec Spectral Shift) : spectral shift force tout à n/m
function methode_relaxation_jump(H, n) # mauvais résultat
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

# BP
function methode_belief_propagation(H, n; max_iter=500, damping=0.9)
    m = size(H, 1)
    
    # Changement de variable
    J = -H ./ 4.0
    for i in 1:m 
        J[i, i] = 0.0 
    end 

    h_base = [-sum(H[i, j] for j in 1:m if j != i) / 4.0 for i in 1:m]
    magnetization_target = 2n - m 
    
    best_idx = Int[]

    # On initialise les messages une seule fois pour gagner du temps
    u = zeros(m, m)
    u_new = zeros(m, m)

    for step in 1:n
        mu = 0.0 
        
        # BP min-sum lui-même
        for iter in 1:max_iter
            # Optimisation : On pré-calcule la somme de tous les messages entrants
            total_u_in = [sum(u[k, i] for k in 1:m if k != i) for i in 1:m]
            
            m_est = sum(clamp(h_base[i] + mu + total_u_in[i], -1.0, 1.0) for i in 1:m)
            mu += 0.1 * (magnetization_target - m_est) 
            
            for i in 1:m
                for j in 1:m
                    if i == j 
                        continue 
                    end
                    
                    # cavity trick : le champ total moins le message j
                    cavity_field_i = h_base[i] + mu + total_u_in[i] - u[j, i]
                    
                    u_new[i, j] = sign(J[i, j]) * sign(cavity_field_i) * min(abs(J[i, j]), abs(cavity_field_i))
                end
            end
            
            diff = norm(u_new - u)
            u = damping * u .+ (1 - damping) * u_new
            
            if diff < 1e-4
                break
            end
        end
        
        # Croyances finales
        beliefs = zeros(m)
        for i in 1:m
            if i in best_idx
                beliefs[i] = -Inf
            else
                beliefs[i] = h_base[i] + mu + sum(u[k, i] for k in 1:m if k != i)
            end
        end
        
        chosen = argmax(beliefs)
        push!(best_idx, chosen)
        
        h_base[chosen] = 1e6
    end
    
    best_idx = best_idx[1:n] # mieux que sortperm, on fixe xi donc les probas suivantes le prennent en compte
    E_min = 0.5 * sum(H[i, j] for i in best_idx, j in best_idx)
    
    return best_idx, E_min
end

# BP avec post-processing pour corriger l'erreur de décimation
function methode_belief_propagation_post_process(H, n; max_iter=500, damping=0.9)
    m = size(H, 1)
    
    # Changement de variable
    J = -H ./ 4.0
    for i in 1:m 
        J[i, i] = 0.0 
    end 

    h_base = [-sum(H[i, j] for j in 1:m if j != i) / 4.0 for i in 1:m]
    magnetization_target = 2n - m 
    
    best_idx = Int[]

    # On initialise les messages une seule fois pour gagner du temps
    u = zeros(m, m)
    u_new = zeros(m, m)

    for step in 1:n
        mu = 0.0 
        
        # BP min-sum lui-même
        for iter in 1:max_iter
            # Optimisation : On pré-calcule la somme de tous les messages entrants
            total_u_in = [sum(u[k, i] for k in 1:m if k != i) for i in 1:m]
            
            m_est = sum(clamp(h_base[i] + mu + total_u_in[i], -1.0, 1.0) for i in 1:m)
            mu += 0.1 * (magnetization_target - m_est) 
            
            for i in 1:m
                for j in 1:m
                    if i == j 
                        continue 
                    end
                    
                    # cavity trick : le champ total moins le message j
                    cavity_field_i = h_base[i] + mu + total_u_in[i] - u[j, i]
                    
                    u_new[i, j] = sign(J[i, j]) * sign(cavity_field_i) * min(abs(J[i, j]), abs(cavity_field_i))
                end
            end
            
            diff = norm(u_new - u)
            u = damping * u .+ (1 - damping) * u_new
            
            if diff < 1e-4
                break
            end
        end
        
        # Croyances finales
        beliefs = zeros(m)
        for i in 1:m
            if i in best_idx
                beliefs[i] = -Inf
            else
                beliefs[i] = h_base[i] + mu + sum(u[k, i] for k in 1:m if k != i)
            end
        end
        
        chosen = argmax(beliefs)
        push!(best_idx, chosen)
        
        h_base[chosen] = 1e6
    end
    
    best_idx = best_idx[1:n] # mieux que sortperm, on fixe xi donc les probas suivantes le prennent en compte

    # post-processing par recherche locale (swaps)
    unselected = setdiff(1:m, best_idx)
    improved = true
    while improved
        improved = false
        best_delta, best_s_idx, best_u_idx = 0.0, 0, 0

        for (i, s) in enumerate(best_idx), (j, u) in enumerate(unselected)
            # delta = (Énergie si on ajoute 'u') - (Énergie si on enlève 's')
            delta_E = sum(H[u, x] - H[s, x] for x in best_idx if x != s)
            if delta_E < best_delta - 1e-7
                best_delta, best_s_idx, best_u_idx = delta_E, i, j
            end
        end

        if best_delta < -1e-7
            # On corrige l'erreur de décimation de BP
            best_idx[best_s_idx], unselected[best_u_idx] = unselected[best_u_idx], best_idx[best_s_idx]
            improved = true
        end
    end

    E_min = 0.5 * sum(H[i, j] for i in best_idx, j in best_idx)
    return best_idx, E_min
end

# Reregarder #TODO
# parfois presque le min, parfois très mauvais
function methode_gumbel_softmax_it(H, n; epochs=1500, lr=0.1, tau=0.5, lambda_reg=5.0)
    m = size(H, 1)
    logits = randn(m) # Initialisation des préférences
    
    # Optimiseur Adam pour les logits
    opt_m = zeros(m)
    opt_v = zeros(m)
    
    for epoch in 1:epochs
        # Générer le bruit Gumbel (Astuce de reparamétrisation)
        U = rand(m)
        G = -log.(-log.(U .+ 1e-10) .+ 1e-10)
        
        # Le bloc Zygote, plus rapide avec l'algèbre linéaire
        grad = Zygote.gradient(logits) do L
            epsilons = 1.0 ./ (1.0 .+ exp.(-(L .+ G) ./ tau))
            
            # dot(a, b) calcule a^T * b de manière nativement optimisée
            E = 0.5 * dot(epsilons, H * epsilons)
            
            penalty = lambda_reg * (sum(epsilons) - n)^2
            
            return E + penalty
        end[1]
        
        # Mise à jour Adam (Vectorisée avec .+= et .*=)
        opt_m .= 0.9 .* opt_m .+ 0.1 .* grad
        opt_v .= 0.999 .* opt_v .+ 0.001 .* (grad.^2)
        logits .-= lr .* opt_m ./ (sqrt.(opt_v) .+ 1e-8)
    end
    
    best_idx = sortperm(logits, rev=true)[1:n]
    return best_idx, evaluer_energie(H, best_idx)
end

function methode_gumbel_softmax(H, n; n_starts=10)
    best_E = Inf
    best_idx = Int[]
    
    # On lance l'algorithme n_starts fois
    for i in 1:n_starts
        # On peut réduire les epochs à 1000 pour que le multistart soit encore plus rapide
        idx, E = methode_gumbel_softmax_it(H, n)
        
        if E < best_E
            best_E = E
            best_idx = idx
        end
    end
    
    return best_idx, best_E
end