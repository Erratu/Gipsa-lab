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

function thomson_energy(X::Matrix{Float64}) 
    n = size(X,2) 
    E = 0.0 
    for i in 1:n-1, j in i+1:n 
        E += 1 / norm(X[:,i] - X[:,j]) 
    end 
    return E 
end 

function random_point_sphere() 
    v = randn(3) 
    return v / norm(v) 
end 

function random_configuration(n) 
    return hcat([random_point_sphere() for _ in 1:n]...) 
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

function evaluer_energie(H, indices)
    return 0.5 * sum(H[i, j] for i in indices, j in indices)
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
    S = [1]
    U = Set(2:m)
    while length(S) < n
        best_u, min_add_E = -1, Inf
        for u in U
            add_E = sum(H[u, s] for s in S)
            if add_E < min_add_E
                min_add_E, best_u = add_E, u
            end
        end
        push!(S, best_u)
        delete!(U, best_u)
    end
    
    U_arr = collect(U)
    
    # Swaps
    improved = true
    while improved
        improved = false
        best_delta, best_s_idx, best_u_idx = 0.0, 0, 0
        
        for (i, s) in enumerate(S), (j, u) in enumerate(U_arr)
            delta_E = sum(H[u, x] - H[s, x] for x in S if x != s)
            if delta_E < best_delta - 1e-7
                best_delta, best_s_idx, best_u_idx = delta_E, i, j
            end
        end
        
        if best_delta < -1e-7
            S[best_s_idx], U_arr[best_u_idx] = U_arr[best_u_idx], S[best_s_idx]
            improved = true
        end
    end
    
    return S, evaluer_energie(H, S)
end

# Relaxation Continue (JuMP + Ipopt avec Spectral Shift)
function methode_relaxation_jump(H, n)
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

function sgd_loo(n::Int; lr=0.01,max_iter=2000)
    X = random_configuration(n+1)
    n = size(X, 2)
    history = Float64[]
    decay_rate = 0.999
    current_lr = lr
    
    for iter in 1:max_iter
        batch_idx = randperm(n)[1:n]
        
        grad = Zygote.gradient(thomson_energy, X)[1]
        
        X[:, batch_idx] .-= current_lr .* grad[:, batch_idx]

        for i in batch_idx
            X[:, i] ./= norm(X[:, i])
        end

        current_lr *= decay_rate
    end

    return X, history
end

function comparaison(n)
    m = n+1
    X, history = sgd_loo(n)
    println("Config m trouvée.")
    
    println("\n--- 2. Sélection du meilleur sous-ensemble (n=$n) ---")
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
    
    return X, history
end

n=38
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
    title="Configuration finale (n=20 sur m=21)",
    markersize=5,
    color=couleurs,
    legend=false,
    camera=(45, 30)
)
plot(p1, p2, layout=(1, 2), size=(900, 400))

function take_random_subset(Y::Matrix{Float64})
    n = size(Y, 2)
    
    # q(X) uniforme
    idx_to_remove = rand(1:n)
    indices_to_keep = [1:idx_to_remove-1; idx_to_remove+1:n]
    X = Y[:, indices_to_keep]
    removed_point = Y[:, idx_to_remove]
    
    return X, removed_point, idx_to_remove
end

