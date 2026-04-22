# Ici m est le nombre de points à retirer, je me suis trompée
function take_random_subset_dpp(Y::Matrix{Float64}, m::Int; iter=1000, gamma=2.0)
    n = size(Y, 2)
    if m <= 0 return Int[] end
    if m >= n return collect(1:n) end

    # 1. Construction optimisée de L
    # Utilisation de abs2 et @views pour éviter les allocations de tableaux intermédiaires
    L = zeros(n, n)
    for i in 1:n
        for j in i:n # On ne calcule que la moitié (symétrie)
            dist_sq = sum(abs2, @views Y[:, i] .- Y[:, j])
            val = exp(-gamma * dist_sq)
            L[i, j] = val
            L[j, i] = val
        end
    end

    current_subset = randperm(n)[1:m]
    L_sub = zeros(m, m) # Buffer pré-alloué
    
    # Fonction pour mettre à jour le buffer sans créer de nouvelle matrice
    function update_L_sub!(sub_idx)
        for i in 1:m, j in 1:m
            @inbounds L_sub[i, j] = L[sub_idx[i], sub_idx[j]]
        end
    end

    update_L_sub!(current_subset)
    # L'utilisation de Symmetric() permet à Julia d'utiliser la factorisation 
    # de Cholesky (O(m^3/3)), beaucoup plus rapide et stable que la LU par défaut.
    current_det = det(Symmetric(L_sub))
    
    while current_det < 1e-12
        current_subset = randperm(n)[1:m]
        update_L_sub!(current_subset)
        current_det = det(Symmetric(L_sub))
    end

    # 2. Boucle MCMC sans aucune allocation mémoire
    for _ in 1:iter
        idx_to_replace = rand(1:m)
        old_u = current_subset[idx_to_replace]
        
        v = rand(1:n)
        while v in current_subset
            v = rand(1:n)
        end
        
        # Mutation en place (on évite le copy())
        current_subset[idx_to_replace] = v
        update_L_sub!(current_subset)
        
        proposed_det = det(Symmetric(L_sub))
        
        ratio = proposed_det / current_det
        
        if rand() < ratio
            # Accepté
            current_det = proposed_det
        else
            # Rejeté : on restaure l'ancien état (rollback en place)
            current_subset[idx_to_replace] = old_u
        end
    end
    
    # Maj final de L_sub pour s'assurer que l'état correspond bien à current_subset
    # update_L_sub!(current_subset) # Décommenter si L_sub doit être renvoyé plus tard
    
    return current_subset
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

        weights = exp.(beta .* (energies .- maximum(energies))) # caractériser la loi selon laquelle j'échantillone, la loi de ce qui reste après les k enlevés, la loi ce qu'on peut dire dessus #TODO
        weights ./= sum(weights)
        
        idx_to_remove_local = sample(1:current_n, Weights(weights)) 
        
        deleteat!(active_indices, idx_to_remove_local)
    end
    
    return active_indices
end

function take_random_subset_bp(Y::Matrix{Float64}, m::Int; tau=0.5, max_iter=100, damping=0.9)
    n = size(Y, 2)
    if m <= 0 return Int[] end
    if m >= n return collect(1:n) end

    # 1. Calcul de la matrice d'interactions H du sous-ensemble actuel
    H = zeros(n, n)
    for i in 1:n
        for j in (i+1):n
            dist = norm(Y[:, i] - Y[:, j])
            H[i, j] = H[j, i] = 1.0 / dist
        end
    end

    # 2. Initialisation des paramètres de Belief Propagation
    J = -H ./ 4.0
    for i in 1:n J[i, i] = 0.0 end
    
    h_base = [-sum(H[i, j] for j in 1:n if j != i) / 4.0 for i in 1:n]
    magnetization_target = 2m - n
    
    u = zeros(n, n)
    u_new = zeros(n, n)
    mu = 0.0

    # 3. Exécution de BP (Min-Sum approché) pour évaluer les points
    for iter in 1:max_iter
        total_u_in = [sum(u[k, i] for k in 1:n if k != i) for i in 1:n]
        m_est = sum(clamp(h_base[i] + mu + total_u_in[i], -1.0, 1.0) for i in 1:n)
        mu += 0.1 * (magnetization_target - m_est)

        for i in 1:n, j in 1:n
            if i == j continue end
            cavity_field_i = h_base[i] + mu + total_u_in[i] - u[j, i]
            u_new[i, j] = sign(J[i, j]) * sign(cavity_field_i) * min(abs(J[i, j]), abs(cavity_field_i))
        end

        diff = norm(u_new - u)
        u = damping * u .+ (1 - damping) * u_new
        
        if diff < 1e-3
            break
        end
    end

    # 4. Calcul des croyances (Beliefs)
    # Plus la croyance est élevée, plus le point a sa place dans la configuration optimale
    beliefs = zeros(n)
    for i in 1:n
        beliefs[i] = h_base[i] + mu + sum(u[k, i] for k in 1:n if k != i)
    end

    # 5. Conversion en probabilités via Softmax
    # On soustrait le maximum pour éviter les débordements numériques (NaN) de l'exponentielle
    max_b = maximum(beliefs)
    weights = exp.((beliefs .- max_b) ./ tau)
    weights ./= sum(weights)

    # 6. Tirage aléatoire sans remise basé sur les poids calculés
    return sample(1:n, Weights(weights), m; replace=false) # Regarder ce que ça fait exactement #TODO
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