function evaluer_energie(H, indices)
    return 0.5 * sum(H[i, j] for i in indices, j in indices)
end

function matrice_interactions(X)
    m = size(X, 2)
    H = zeros(m, m)
    for i in 1:m, j in (i+1):m
        dist = norm(X[:, i] - X[:, j])
        H[i, j] = H[j, i] = 1.0 / dist
    end
    return H
end

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



function sgd_loo_optim(n::Int, m::Int; lr=0.02, max_iter=2000, beta1=0.9, beta2=0.999)
    X = random_configuration(m)
    history = Float64[]
    
    # Initialisation des moments d'Adam
    M_v = zeros(3, m) 
    V_v = zeros(3, m) 
    epsilon_adam = 1e-8
    
    for iter in 1:max_iter 
        batch_idx = randperm(m)[1:n]
        
        # Gradient euclidien classique avec Zygote
        grad = Zygote.gradient(X -> thomson_energy(X[:, batch_idx]), X)[1]
        
        # Projection sur l'espace tangent (Riemannien)
        grad_tangent = zeros(3, m)
        for i in batch_idx
            g = grad[:, i]
            x = X[:, i]
            # On retire la composante normale à la sphère
            grad_tangent[:, i] = g - dot(g, x) * x 
        end
        
        # Mise à jour Adam (uniquement sur les points du batch)
        for i in batch_idx
            M_v[:, i] = beta1 * M_v[:, i] + (1 - beta1) * grad_tangent[:, i]
            V_v[:, i] = beta2 * V_v[:, i] + (1 - beta2) * (grad_tangent[:, i].^2)
            
            # Correction du biais
            m_hat = M_v[:, i] / (1 - beta1^iter)
            v_hat = V_v[:, i] / (1 - beta2^iter)
            
            # Application du pas
            X[:, i] .-= lr .* m_hat ./ (sqrt.(v_hat) .+ epsilon_adam)
            
            # Rétraction exacte sur la sphère
            X[:, i] ./= norm(X[:, i])
        end

        if iter % 10 == 0
            push!(history, thomson_energy(X))
        end
    end

    println("Résultat Adam (Riemannien) pour m=$m : E = $(round(thomson_energy(X), digits=6))")
    
    return X, history
end

function sgd_loo(m::Int, n::Int; lr=0.01, max_iter=2000, strategy=:boltzmann)
    X = random_configuration(n)
    history = Float64[]
    current_lr = lr
    decay_rate = 0.999
    
    for iter in 1:max_iter 
        # Choix de la distribution q
        if strategy == :uniform
            batch_idx = randperm(n)[1:m]
        elseif strategy == :boltzmann
            batch_idx = take_random_subset_boltzmann(X, m, beta=2.0)
        elseif strategy == :dpp
            batch_idx = take_random_subset_dpp(X, m, iter=10) # iter réduit pour la vitesse
        end
        
        grad = Zygote.gradient(thomson_energy, X[:, batch_idx])[1] 
        X[:, batch_idx] .-= current_lr .* grad

        for i in batch_idx
            X[:, i] ./= norm(X[:, i])
        end

        current_lr *= decay_rate
        if iter % 10 == 0
            push!(history, thomson_energy(X[:, batch_idx])) # Enregistrer l'énergie du sous-ensemble
        end
    end
    return X, history
end

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

function methode_exhaustive_n_proche_m(H, n)
    m = size(H, 1)
    k = m - n  # Nombre de points à exclure
    
    # Si k est trop grand, on risque l'explosion combinatoire
    if binomial(BigInt(m), BigInt(k)) > 1_000_000
        @warn "Trop de combinaisons, passage à l'heuristique"
        return methode_heuristique(H, n)
    end

    E_total_systeme = 0.5 * sum(H) # Énergie de tous les points m
    E_min = Inf
    best_excluded = Int[]

    # On cherche les k points qui, une fois retirés, baissent le plus l'énergie
    for excluded in combinations(1:m, k)
        # Énergie interne des points exclus
        E_internal_excl = 0.5 * sum(H[i, j] for i in excluded, j in excluded)
        
        # Interactions entre les points exclus et TOUT le reste du système
        E_interaction_excl = sum(H[i, j] for i in excluded, j in 1:m) - 2.0 * E_internal_excl
        
        # Énergie du sous-ensemble restant (n points)
        E_restant = E_total_systeme - (E_internal_excl + E_interaction_excl)
        
        if E_restant < E_min
            E_min = E_restant
            best_excluded = excluded
        end
    end
    
    indices_final = setdiff(1:m, best_excluded)
    return indices_final, E_min
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

function sgd_diffusion_measure(n_target::Int; n_large::Int=1000, gamma::Float64=0.005, max_iter::Int=5000, burn_in::Int=2000)
    X = random_configuration(n_large)
    
    # Historique de la phase stationnaire
    stationary_samples = []

    for iter in 1:max_iter 
        # Bruit inhérent au SGD (batch) agit comme la diffusion
        batch_idx = randperm(n_large)[1:Int(n_large/2)] 
        
        grad = Zygote.gradient(thomson_energy, X[:, batch_idx])[1] # energie des points restants, et pas tout X
        # plutôt que de l'enlever directement, mettre un poids dans [0,1], pour qu'il ne gâche pas trop l'énergie globale, qu'on pourrait faire tendre vers 0 ?
        X[:, batch_idx] .-= gamma .* grad

        for i in 1:n_large
            X[:, i] ./= norm(X[:, i])
        end

        if iter > burn_in && iter % 10 == 0
            # On collecte la mesure de probabilité
            push!(stationary_samples, copy(X))
        end
    end

    @info "SGD a terminé son orbite stationnaire. Extraction des amas (BLASSO)"
    
    # Connexion avec BLASSO : Trouver les "Spikes" de la mesure
    # On rassemble toutes les particules vues pendant la phase stationnaire
    all_points = hcat(stationary_samples...)
    
    # On utilise un algorithme de clustering (K-Means) pour trouver les centres de densité (les n_target points optimaux purs)
    clustering = kmeans(all_points, n_target, maxiter=200)
    X_optimal = clustering.centers
    
    # Projection finale stricte sur la sphère
    for i in 1:n_target
        X_optimal[:, i] ./= norm(X_optimal[:, i])
    end

    return X_optimal, all_points
end

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

# La "température" beta inverse sera modélisée par le nombre de points supplémentaires k (n = m + k)
# beta faible = haute température = grand k.
# On fixe m (le nombre cible)

function setup_pt_thomson(m::Int, k_max::Int)
    nb_sim = k_max + 1
    # betas de 1.0 (k=0, cible) à 0.0 (k=k_max, exploration)
    betas = collect(range(1.0, 0.0, length=nb_sim)) 
    
    # L'état initial : des configurations de taille croissante
    initial_sims = [random_configuration(m + k) for k in 0:k_max]
    
    state = PTState{Matrix{Float64}}(initial_sims)
    sampler = PTSampler(betas)
    
    # Fonction d'énergie : on évalue toujours la meilleure sous-configuration de taille m
    function energy_func(X)
        current_n = size(X, 2)
        if current_n == m
            return thomson_energy(X)
        else
            # Pour un pool plus grand, on extrait la meilleure sous-configuration (Heuristique rapide)
            H = matrice_interactions(X)
            idx, E = methode_heuristique(H, m)
            return E
        end
    end
    
    # Update local : quelques pas de SGD_loo sur le pool n
    function local_update!(X, beta)
        current_n = size(X, 2)
        # 50 itérations de SGD pour relaxer la configuration
        X_new, _ = sgd_loo(m, current_n, max_iter=50, strategy=:boltzmann)
        return X_new
    end
    
    # Observable (par exemple l'énergie)
    obs_func(X) = energy_func(X)
    
    return sampler, state, energy_func, local_update!, obs_func
end

# Utilisation :
# sampler, state, E_func, upd_func, obs_func = setup_pt_thomson(46, 5)
# algo_pt_complet!(sampler, state, E_func, upd_func, obs_func, 50, 200)