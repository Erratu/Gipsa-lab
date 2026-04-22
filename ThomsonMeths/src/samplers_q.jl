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

        weights = exp.(beta .* (energies .- maximum(energies))) 
        weights ./= sum(weights)
        
        idx_to_remove_local = sample(1:current_n, Weights(weights))
        
        deleteat!(active_indices, idx_to_remove_local)
    end
    
    return active_indices
end