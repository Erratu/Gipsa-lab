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