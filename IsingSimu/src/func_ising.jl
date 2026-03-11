"""
Génère la matrice d'adjacence Q pour une chaîne de n spins.
periodic = true  => Boucle (Anneau)
periodic = false => Chaîne ouverte
"""
function adjacency_1d(n; periodic=false)
    Q = zeros(Int, n, n)
    for i in 1:n-1
        Q[i, i+1] = Q[i+1, i] = 1
    end
    if periodic && n > 2
        Q[1, n] = Q[n, 1] = 1
    end
    return Q
end 

""" Génère Q pour une grille L x L (Ising 2D) """
function adjacency_2d(L; periodic=false)
    n = L*L
    Q = zeros(Int, n, n)
    grid = reshape(1:n, L, L)
    for r in 1:L, c in 1:L
        i = grid[r, c]
        if c < L 
            Q[i, grid[r, c+1]] = Q[grid[r, c+1], i] = 1
        elseif periodic 
            Q[i, grid[r, 1]] = Q[grid[r, 1], i] = 1 
        end
        if r < L 
            Q[i, grid[r+1, c]] = Q[grid[r+1, c], i] = 1
        elseif periodic 
            Q[i, grid[1, c]] = Q[grid[1, c], i] = 1 
        end
    end
    return Q
end

function energy(X, Q, m)
    return -0.5 * dot(X, Q, X) - dot(m, X)
end

function metropolis_ising!(X, Q, m, beta, iterations; fixed_indices=Int[])
    n = length(X)
    energies = Float64[]
    current_E = energy(X, Q, m)

    for _ in 1:iterations
        i = rand(1:n)
        if i in fixed_indices
            push!(energies, current_E)
            continue
        end
        
        # ΔE = E(spin inversé) - E(spin actuel)
        # Pour Ising : ΔE = 2 * s_i * (∑ Q_ij * s_j + m_i)
        h_local = dot(view(Q, i, :), X) + m[i]
        delta_E = 2 * X[i] * h_local
        
        # Critère d'acceptation de Metropolis
        if delta_E <= 0 || rand() < exp(-beta * delta_E)
            X[i] *= -1
            current_E += delta_E
        end
        push!(energies, current_E)
    end
    return X, energies
end

function heat_bath_ising!(X, Q, m, beta, iterations; fixed_indices=Int[])
    n = length(X)
    energies = Float64[]
    current_E = energy(X, Q, m)

    for _ in 1:iterations
        i = rand(1:n)
        
        # SI l'indice est fixé (condition aux limites), on ne fait rien
        if i in fixed_indices
            push!(energies, current_E)
            continue
        end

        # Calcul correct du Delta E pour le spin i
        # ΔE = E(s_i=1) - E(s_i=-1) = -2 * (∑ Q_ij s_j + m_i)
        h_local = dot(view(Q, i, :), X) + m[i]
        p = 1 / (1 + exp(-2 * beta * h_local)) 
        
        old_val = X[i]
        X[i] = rand() < p ? 1 : -1
        current_E += -h_local * (X[i] - old_val)
        push!(energies, current_E)
    end
    return X, energies
end
