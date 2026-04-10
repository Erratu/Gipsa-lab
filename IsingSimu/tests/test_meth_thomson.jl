using IsingSimu
using Printf 
using LinearAlgebra 
using Random 
import GLMakie as GL 
using Zygote
using Combinatorics
using JuMP
using Ipopt
using Plots
using StatsBase

function matrice_interactions(X)
    m = size(X, 2)
    H = zeros(m, m)
    for i in 1:m, j in (i+1):m
        dist = norm(X[:, i] - X[:, j])
        H[i, j] = H[j, i] = 1.0 / dist
    end
    return H
end

function evaluer_energie(H, indices)
    return 0.5 * sum(H[i, j] for i in indices, j in indices)
end

function take_random_subset_boltzmann(Y::Matrix{Float64}; beta=2.0)
    n_cols = size(Y, 2)
    energies = zeros(n_cols)
    
    for i in 1:n_cols
        E_i = 0.0
        for j in 1:n_cols
            if i != j
                E_i += 1 / norm(Y[:, i] - Y[:, j])
            end
        end
        energies[i] = E_i
    end

    # beta positif : on favorise l'exclusion des points à haute énergie
    # beta négatif : on favorise l'exclusion des points à basse énergie
    weights = exp.(beta .* (energies .- maximum(energies))) 
    weights ./= sum(weights)
    
    idx_to_remove = sample(1:n_cols, Weights(weights))
    indices_to_keep = setdiff(1:n_cols, idx_to_remove)
    
    return indices_to_keep
end

# Recherche Exhaustive (Exacte mais lente)
function methode_exhaustive(H, n)
    m = size(H, 1)
    if binomial(BigInt(m), BigInt(n)) > 100_000
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

# Relaxation Continue (JuMP + Ipopt avec Spectral Shift)
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
function methode_belief_propagation(H, n; max_iter=500, beta=1.0, damping=0.9)
    m = size(H, 1)
    
    # Changement de variable
    J = -H ./ 4.0
    for i in 1:m 
        J[i, i] = 0.0 
    end 

    h_base = [-sum(H[i, j] for j in 1:m if j != i) / 4.0 for i in 1:m]
    magnetization_target = 2n - m 
    
    # Initialisation des messages
    u = zeros(m, m)
    u_new = zeros(m, m)
    mu = 0.0 
    
    # BP lui-même
    for iter in 1:max_iter
        m_est = sum(tanh.(beta * (h_base[i] + mu + sum(u[k, i] for k in 1:m if k != i))) for i in 1:m)
        mu += 0.1 * (magnetization_target - m_est) 
        
        for i in 1:m
            for j in 1:m
                if i == j 
                    continue 
                end
                
                cavity_field_i = h_base[i] + mu
                for k in 1:m
                    if k != i && k != j
                        cavity_field_i += u[k, i]
                    end
                end
                
                tanh_J = tanh(beta * J[i, j])
                tanh_cavity = tanh(beta * cavity_field_i)
                val = clamp(tanh_J * tanh_cavity, -0.9999, 0.9999)
                u_new[i, j] = atanh(val) / beta
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
        total_field = h_base[i] + mu + sum(u[k, i] for k in 1:m if k != i)
        beliefs[i] = total_field 
    end
    
    best_idx = sortperm(beliefs, rev=true)[1:n]
    E_min = 0.5 * sum(H[i, j] for i in best_idx, j in best_idx)
    
    return best_idx, E_min
end

# Champ Moyen (Inspiré de Belief Propagation)
function methode_champ_moyen(H, n; T_init=10.0, T_final=0.01, decay=0.9)
    m = size(H, 1)
    v = fill(n / m, m) # Croyances initiales (probabilité uniforme)
    
    T = T_init
    while T > T_final
        # Calcul du champ local agissant sur chaque point
        h_field = H * v 
        
        # Trouver le potentiel mu garantissant sum(v) = n
        mu_min, mu_max = minimum(h_field) - 5*T, maximum(h_field) + 5*T
        mu = 0.0
        for _ in 1:40
            mu = (mu_min + mu_max) / 2
            sum_v = sum(1.0 / (1.0 + exp((h_field[i] - mu) / T)) for i in 1:m)
            sum_v > n ? (mu_max = mu) : (mu_min = mu)
        end
        
        # Mise à jour des croyances (avec un peu d'inertie)
        v_new = [1.0 / (1.0 + exp((h_field[i] - mu) / T)) for i in 1:m]
        v = 0.5 * v .+ 0.5 * v_new 
        
        T *= decay
    end
    
    best_idx = sortperm(v, rev=true)[1:n]
    return best_idx, evaluer_energie(H, best_idx)
end

function sgd_loo_optim(n::Int; lr=0.02, max_iter=2000, beta1=0.9, beta2=0.999)
    m = n + 1
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

function sgd_loo(n::Int; lr=0.01,max_iter=2000)
    m = n+1
    X = random_configuration(m)
    history = Float64[]
    decay_rate = 0.999
    current_lr = lr
    
    for iter in 1:max_iter 
        #batch_idx = randperm(m)[1:n]

        # Appel de la fonction de sélection non-uniforme q(X)
        batch_idx = take_random_subset_boltzmann(X)
        
        grad = Zygote.gradient(thomson_energy, X[:, batch_idx])[1] # energie des points restants, et pas tout X
        # plutôt que de l'enlever directement, mettre un poids dans [0,1], pour qu'il ne gâche pas trop l'énergie globale, qu'on pourrait faire tendre vers 0 ?
        X[:, batch_idx] .-= current_lr .* grad

        for i in batch_idx
            X[:, i] ./= norm(X[:, i])
        end

        current_lr *= decay_rate
        
        if iter % 10 == 0
            push!(history, thomson_energy(X))
        end
    end

    println("Résultat sgd_loo pour m=$m : E = $(round(thomson_energy(X), digits=6))")
    
    return X, history
end

function methode_gumbel_softmax(H, n; epochs=1500, lr=0.1, tau=0.5, lambda_reg=5.0)
    m = size(H, 1)
    logits = randn(m) # Initialisation des préférences
    
    # Optimiseur Adam pour les logits
    opt_m = zeros(m)
    opt_v = zeros(m)
    
    for epoch in 1:epochs
        # Générer le bruit Gumbel hors de Zygote (astuce de reparamétrisation)
        U = rand(m)
        G = -log.(-log.(U .+ 1e-10) .+ 1e-10)
        
        grad = Zygote.gradient(logits) do L
            # Relaxation continue de la sélection (Gumbel-Sigmoid)
            epsilons = 1.0 ./ (1.0 .+ exp.(-(L .+ G) ./ tau))
            
            # Énergie de la sous-configuration
            E = 0.5 * sum(epsilons[i] * H[i, j] * epsilons[j] for i in 1:m, j in 1:m)
            
            # Pénalité dure : on veut exactement n points
            penalty = lambda_reg * (sum(epsilons) - n)^2
            
            return E + penalty
        end[1]
        
        # Mise à jour Adam
        opt_m = 0.9 * opt_m + 0.1 * grad
        opt_v = 0.999 * opt_v + 0.001 * (grad.^2)
        logits .-= lr .* opt_m ./ (sqrt.(opt_v) .+ 1e-8)
    end
    
    # On tire les meilleures probabilités finales
    best_idx = sortperm(logits, rev=true)[1:n]
    return best_idx, evaluer_energie(H, best_idx)
end

function comparaison(n)
    m = n+1
    t_sgd = @elapsed X, history = sgd_loo(n)
    println("Config m trouvée. Temps = $(round(t_sgd, digits=5)) s")
    
    println("\nSélection du meilleur sous-ensemble (n=$n)")
    H = matrice_interactions(X)
    
    # Recherche Exhaustive
    t_ex = @elapsed idx_ex, E_ex = methode_exhaustive(H, n)
    println("Exhaustive : E = $(round(E_ex, digits=6)) | Point exclu : $(setdiff(1:m, idx_ex)[1]) | Temps = $(round(t_ex, digits=5)) s")
    
    # Glouton + Swaps
    t_heu = @elapsed idx_heu, E_heu = methode_heuristique(H, n)
    println("Heuristique: E = $(round(E_heu, digits=6)) | Point exclu : $(setdiff(1:m, idx_heu)[1]) | Temps = $(round(t_heu, digits=5)) s")
    
    # Relaxation Continue (JuMP/Ipopt)
    t_jump = @elapsed idx_jump, E_jump = methode_relaxation_jump(H, n)
    println("Relaxation : E = $(round(E_jump, digits=6)) | Point exclu : $(setdiff(1:m, idx_jump)[1]) | Temps = $(round(t_jump, digits=5)) s")
    
    # Champ Moyen (BP approché)
    t_cm = @elapsed idx_cm, E_cm = methode_champ_moyen(H, n)
    println("Champ Moyen: E = $(round(E_cm, digits=6)) | Point exclu : $(setdiff(1:m, idx_cm)[1]) | Temps = $(round(t_cm, digits=5)) s")
    
    # Belief Propagation
    t_bp = @elapsed idx_bp, E_bp = methode_belief_propagation(H, n)
    println("Belief Propagation: E = $(round(E_bp, digits=6)) | Point exclu : $(setdiff(1:m, idx_bp)[1]) | Temps = $(round(t_bp, digits=5)) s")
    
    # Gumbel-Softmax
    t_gs = @elapsed idx_gs, E_gs = methode_gumbel_softmax(H, n)
    println("Gumbel-Softmax: E = $(round(E_gs, digits=6)) | Point exclu : $(setdiff(1:m, idx_gs)[1]) | Temps = $(round(t_gs, digits=5)) s")

    return X, history
end

n=45
X_final, hist = comparaison(n)

p1 = plot(hist, 
    title="Convergence de l'énergie (SGD)", 
    xlabel="Itérations (x500)", 
    ylabel="Énergie de Thomson", 
    linewidth=2, 
    legend=false,
    color=:blue
)

x_coords = X_final[1, :]
y_coords = X_final[2, :]
z_coords = X_final[3, :]

H_final = matrice_interactions(X_final)
idx_gardes, _ = methode_heuristique(H_final, 20)
couleurs = [i in idx_gardes ? :green : :red for i in 1:21]

p2 = scatter(x_coords, y_coords, z_coords, 
    title="Configuration finale (n=$n sur m=$(n+1))",
    markersize=5,
    color=couleurs,
    legend=false,
    camera=(45, 30)
)
p = plot(p1, p2, layout=(1, 2), size=(900, 400))
display(p)

function take_random_subset(Y::Matrix{Float64})
    n = size(Y, 2)
    
    # q(X) uniforme
    idx_to_remove = rand(1:n)
    indices_to_keep = [1:idx_to_remove-1; idx_to_remove+1:n]
    X = Y[:, indices_to_keep]
    removed_point = Y[:, idx_to_remove]
    
    return X, removed_point, idx_to_remove
end
