function take_random_subset_dpp_opt(Y::Matrix{Float64}, m::Int; gamma=1.0, iter=10000)
    n = size(Y, 2)
    if m >= n return collect(1:n) end
    
    # 1. Précalcul de la matrice noyau L (Une seule fois !)
    L = zeros(n, n)
    for i in 1:n
        for j in (i+1):n
            # Distances au carré pour le DPP
            d2 = sum(abs2, view(Y, :, i) .- view(Y, :, j))
            L[i, j] = L[j, i] = exp(-gamma * d2)
        end
        # Stabilité numérique sur la diagonale
        L[i, i] = 1.0 + 1e-8 
    end

    # 2. Initialisation (Séparation des actifs et exclus)
    S = collect(1:m)          # Les points gardés
    Sc = collect((m+1):n)     # Les points exclus
    
    # 3. Pré-allocation du Buffer (Le secret de la vitesse : aucune allocation dans la boucle)
    buffer = zeros(m, m)
    
    # Fonction interne ultra-rapide pour évaluer le "volume" (déterminant)
    function log_det_S!(current_S)
        @inbounds for i in 1:m, j in 1:m
            buffer[i, j] = L[current_S[i], current_S[j]]
        end
        # Cholesky est O(m^3) mais environ 3x plus rapide que det(), et logdet évite les underflows
        C = cholesky!(Symmetric(buffer); check=false) 
        return issuccess(C) ? logdet(C) : -Inf
    end

    # Évaluation initiale
    current_logdet = log_det_S!(S)

    # 4. Chaîne de Markov (Metropolis-Hastings)
    for _ in 1:iter
        # Tirage d'un échange (Swap)
        idx_in = rand(1:m)           # Un point à retirer de S
        idx_out = rand(1:(n-m))      # Un point à injecter depuis Sc
        
        # On fait l'échange
        S[idx_in], Sc[idx_out] = Sc[idx_out], S[idx_in]
        
        # Nouvelle évaluation
        new_logdet = log_det_S!(S)
        
        # Ratio d'acceptation de Metropolis (en log pour la stabilité)
        log_alpha = new_logdet - current_logdet
        
        if log_alpha >= 0.0 || rand() < exp(log_alpha)
            # Accepté ! On garde l'échange
            current_logdet = new_logdet
        else
            # Refusé : Rollback (on annule l'échange, O(1))
            S[idx_in], Sc[idx_out] = Sc[idx_out], S[idx_in]
        end
    end

    return S
end

function take_random_subset_boltzmann(Y::Matrix{Float64}, m::Int; beta=2.0)
    n = size(Y, 2)
    k = n - m # Nombre de points à enlever
    
    if k <= 0 
        return Int[] 
    end
    
    active_indices = collect(1:n)
    
    for step in 1:k
        current_n = length(active_indices)
        energies = zeros(current_n)
        
        @inbounds for i in 1:current_n
            E_i = 0.0
            idx_i = active_indices[i]
            for j in 1:current_n
                if i != j
                    idx_j = active_indices[j]
                    E_i += 1.0 / norm(Y[:, idx_i] - Y[:, idx_j])
                end
            end
            energies[i] = E_i
        end

        weights = exp.(beta .* (energies .- maximum(energies)))
        weights ./= sum(weights)
        
        idx_to_remove_local = sample(1:current_n, Weights(weights)) 
        
        deleteat!(active_indices, idx_to_remove_local)
    end
    
    return active_indices
end

function take_random_subset_bp(Y::Matrix{Float64}, m::Int; beta=2.0, tau=0.5, max_iter=100, damping=0.8)
    n = size(Y, 2)
    if m <= 0 return Int[] end
    if m >= n return collect(1:n) end

    # 1. Matrice d'interactions
    H = zeros(n, n)
    for i in 1:n, j in (i+1):n
        H[i, j] = H[j, i] = 1.0 / norm(Y[:, i] - Y[:, j])
    end

    U = zeros(n, n) # U[i, j] = message envoyé par i à j (en log-odds)
    U_new = zeros(n, n)
    mu = 0.0 # Potentiel chimique global pour cibler m points
    sum_U = zeros(n)
    p = zeros(n)
    
    for iter in 1:max_iter
        # Somme des messages entrants pour chaque point (son "champ" actuel)
        sum!(sum_U', U)
        
        # Ajustement de mu pour cibler m points (les probabilités marginales doivent sommer à m)
        @. p = 1.0 / (1.0 + exp(-(mu + sum_U)))
        mu += 0.2 * (m - sum(p)) 

        # Calcul des nouveaux messages
        @inbounds for i in 1:n, j in 1:n
            if i == j continue end
            
            # Champ cavité : ce que 'i' ressent de tout le monde SAUF de 'j'
            cavity_field = mu + sum_U[i] - U[j, i]
            
            # Formule exacte Sum-Product (avec prévention d'overflow numérique)
            if cavity_field > 50.0 
                # i est certain d'être gardé : il envoie sa répulsion maximale à j
                U_new[i, j] = -beta * H[i, j]
            elseif cavity_field < -50.0
                # i est certain d'être exclu : il n'envoie aucune répulsion à j
                U_new[i, j] = 0.0
            else
                # Régime probabiliste standard
                exp_cav = exp(cavity_field)
                U_new[i, j] = log((exp(-beta * H[i, j]) * exp_cav + 1.0) / (exp_cav + 1.0))
            end
        end

        if norm(U_new - U) < 1e-4 break end
        U = damping * U .+ (1 - damping) * U_new
    end

    # 3. Calcul des croyances finales (en log-odds)
    beliefs = zeros(n)
    for i in 1:n
        beliefs[i] = mu + sum(U[:, i])
    end

    # 4. Décimation séquentielle avec soulagement
    active = collect(1:n)
    current_beliefs = copy(beliefs)
    
    for _ in 1:(n - m)
        # On exclut séquentiellement (pires croyances = plus fortes probabilités d'exclusion)
        min_b = minimum(current_beliefs[active])
        w_remove = exp.(-(current_beliefs[active] .- min_b) ./ tau)
        w_remove ./= sum(w_remove)
        
        idx_local = sample(1:length(active), Weights(w_remove))
        idx_global = active[idx_local]
        deleteat!(active, idx_local)
        
        # le point idx_global disparaît, on retire sa répulsion aux autres
        for voisin in active
            current_beliefs[voisin] += beta * H[idx_global, voisin]
        end
    end

    return active
end

function take_random_subset_bp_opt(Y::Matrix{Float64}, m::Int; tau=0.5, max_iter=100, damping=0.9)
    n = size(Y, 2)
    dim = size(Y, 1)
    
    if m <= 0 return Int[] end
    if m >= n return collect(1:n) end

    # 1. Initialisation optimisée de H et J
    H = zeros(n, n)
    J = zeros(n, n)
    h_base = zeros(n)
    
    @inbounds for i in 1:n
        for j in (i+1):n
            # Calcul manuel de la distance pour éviter les allocations de Y[:, i]
            dist_sq = 0.0
            for k in 1:dim
                dist_sq += (Y[k, i] - Y[k, j])^2
            end
            val = 1.0 / sqrt(dist_sq)
            
            H[i, j] = H[j, i] = val
            J[i, j] = J[j, i] = -val / 4.0
        end
    end

    # Calcul de h_base pré-alloué
    @inbounds for i in 1:n
        s = 0.0
        for j in 1:n
            if j != i
                s += H[i, j]
            end
        end
        h_base[i] = -s / 4.0
    end
    
    magnetization_target = 2.0 * m - n
    
    # 2. Pré-allocation des buffers de messages (Zéro allocation dans la boucle MCMC)
    u = zeros(n, n)
    u_new = zeros(n, n)
    total_u_in = zeros(n)
    mu = 0.0

    # 3. Exécution de BP (Min-Sum)
    @inbounds for iter in 1:max_iter
        
        # A. Mise à jour de total_u_in
        for i in 1:n
            s = 0.0
            for k in 1:n
                if k != i
                    s += u[k, i]
                end
            end
            total_u_in[i] = s
        end
        
        # B. Estimation de la magnétisation et ajustement de mu
        m_est = 0.0
        for i in 1:n
            m_est += clamp(h_base[i] + mu + total_u_in[i], -1.0, 1.0)
        end
        mu += 0.1 * (magnetization_target - m_est)

        # C. Calcul de u_new et mise à jour de l'erreur (diff) en une seule passe
        diff_sq = 0.0
        for i in 1:n
            cavity_base = h_base[i] + mu + total_u_in[i]
            for j in 1:n
                if i == j continue end
                
                cavity_field_i = cavity_base - u[j, i]
                J_ij = J[i, j]
                
                # Cœur du calcul Min-Sum
                new_val = sign(J_ij) * sign(cavity_field_i) * min(abs(J_ij), abs(cavity_field_i))
                
                # Damping et calcul de différence "in-place"
                damped_val = damping * u[i, j] + (1.0 - damping) * new_val
                diff_sq += (damped_val - u[i, j])^2
                
                # Stockage direct (on réutilise u pour éviter de copier après)
                u_new[i, j] = damped_val 
            end
        end

        # Mettre à jour u
        for i in 1:n, j in 1:n
            u[i, j] = u_new[i, j]
        end
        
        if sqrt(diff_sq) < 1e-3
            break
        end
    end

    # 4. Calcul des croyances
    beliefs = zeros(n)
    @inbounds for i in 1:n
        s = 0.0
        for k in 1:n
            if k != i
                s += u[k, i]
            end
        end
        beliefs[i] = h_base[i] + mu + s
    end

    # 5. Softmax optimisé
    max_b = maximum(beliefs)
    weights = zeros(n)
    sum_w = 0.0
    @inbounds for i in 1:n
        w = exp((beliefs[i] - max_b) / tau)
        weights[i] = w
        sum_w += w
    end
    weights ./= sum_w

    # 6. Échantillonnage
    return sample(1:n, Weights(weights), m; replace=false)
end