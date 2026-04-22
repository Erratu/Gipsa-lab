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