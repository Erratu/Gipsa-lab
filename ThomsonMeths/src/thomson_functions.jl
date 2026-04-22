function thomson_energy_with_potential(X::Matrix{Float64}; A=50.0, sigma=0.5)
    E_base = thomson_energy(X)
    n = size(X, 2)
    P_south = [0.0, 0.0, -1.0]
    E_pot = 0.0
    
    # Ajout d'une pénalité si les points s'approchent du pôle sud
    for i in 2:n
        dist2 = norm(X[:, i] - P_south)^2
        E_pot += A * exp(-dist2 / (2 * sigma^2))
    end
    return E_base + E_pot
end

function thomson_energy(X::Matrix{Float64}) 
    n = size(X,2) 
    E = 0.0 
    for i in 1:n-1, j in i+1:n 
        E += 1 / norm(X[:,i] - X[:,j]) 
    end 
    return E 
end 

function project_sphere!(X::Matrix{Float64})
    for i in 1:size(X, 2)
        @views nrm = norm(X[:, i])
        @views X[:, i] ./= nrm
    end
    return X
end

function project_sphere_3D!(X::Matrix{Float64})
    for i in 1:size(X, 2)
        @inbounds begin
            nrm = sqrt(X[1, i]^2 + X[2, i]^2 + X[3, i]^2)
            X[1, i] /= nrm
            X[2, i] /= nrm
            X[3, i] /= nrm
        end
    end
    return X
end

function random_point_sphere() 
    v = randn(3) 
    return v / norm(v) 
end 

function random_configuration(n) 
    return hcat([random_point_sphere() for _ in 1:n]...) 
end

function delta_energy(X::Matrix{Float64}, i::Int, new_point::Vector{Float64}) 
    n = size(X,2) 
    old = X[:,i] 
    ΔE = 0.0 
    for j in 1:n 
        if j == i 
            continue 
        end 
        ΔE += 1 / norm(new_point - X[:,j]) 
        ΔE -= 1 / norm(old - X[:,j]) 
    end 
    return ΔE 
end 

# Version finale optimisée (Aucune allocation)
function delta_energy_fast(X::Matrix{Float64}, i::Int, nx::Float64, ny::Float64, nz::Float64) 
    n = size(X, 2)
    ΔE = 0.0
    
    # On extrait les coordonnées de l'ancien point
    ox, oy, oz = X[1, i], X[2, i], X[3, i]
    
    @inbounds for j in 1:n
        if j == i
            continue
        end
        
        # Accès direct (pas de copie)
        xj, yj, zj = X[1, j], X[2, j], X[3, j]
        
        # Calculs scalaires (ultra-rapides)
        d_new_sq = (nx - xj)^2 + (ny - yj)^2 + (nz - zj)^2
        d_old_sq = (ox - xj)^2 + (oy - yj)^2 + (oz - zj)^2
        
        ΔE += 1.0 / sqrt(d_new_sq) - 1.0 / sqrt(d_old_sq)
    end
    
    return ΔE
end

function thomson_update_fast!(X::Matrix{Float64}, beta::Float64; step_size=0.08) 
    n = size(X, 2) 
    
    @inbounds for _ in 1:n 
        i = rand(1:n) 
        
        # Extraction des coordonnées (Pas de copie de colonne)
        ox, oy, oz = X[1, i], X[2, i], X[3, i]
        
        # Génération de la proposition (Scalaire : Zéro allocation)
        px = ox + step_size * randn()
        py = oy + step_size * randn()
        pz = oz + step_size * randn()
        
        # Normalisation
        p_norm = sqrt(px^2 + py^2 + pz^2)
        px /= p_norm
        py /= p_norm
        pz /= p_norm
        
        # Calcul du delta_E
        delta_E = delta_energy_fast(X, i, px, py, pz)
        
        # Critère de Metropolis
        if delta_E <= 0 || rand() < exp(-beta * delta_E) 
            X[1, i] = px
            X[2, i] = py
            X[3, i] = pz
        end 
    end 
    return X 
end

function thomson_update!(X::Matrix{Float64}, beta::Float64; step_size=0.08) 
    n = size(X,2) 
    for _ in 1:n 
        i = rand(1:n) # les parcourir en scan
        old = X[:,i]
        
        proposal = old + step_size * randn(3)
        proposal /= norm(proposal)
        
        ΔE = delta_energy(X, i, proposal)
        
        if ΔE <= 0 || rand() < exp(-beta * ΔE) 
            X[:,i] = proposal 
        end 
    end 
    return X 
end

function thomson_energy_and_grad!(G::AbstractMatrix{Float64}, X::AbstractMatrix{Float64})
    d, n = size(X)
    fill!(G, 0.0)
    E = 0.0
    
    @inbounds for i in 1:(n-1)
        for j in (i+1):n
            # Calcul de la distance au carré
            dist_sq = 0.0
            for k in 1:d
                dist_sq += (X[k, i] - X[k, j])^2
            end
            
            dist = sqrt(dist_sq)
            E += 1.0 / dist
            
            # 2. Calcul du gradient
            factor = 1.0 / (dist_sq * dist)
            for k in 1:d
                diff_k = X[k, i] - X[k, j]
                gk = factor * diff_k
                
                # Action et Réaction
                G[k, i] -= gk
                G[k, j] += gk
            end
        end
    end
    return E
end

# Version juste pour l'énergie (sans le gradient)
function thomson_energy_fast(X::AbstractMatrix{Float64})
    d, n = size(X)
    E = 0.0
    @inbounds for i in 1:(n-1)
        for j in (i+1):n
            dist_sq = 0.0
            for k in 1:d
                dist_sq += (X[k, i] - X[k, j])^2
            end
            E += 1.0 / sqrt(dist_sq)
        end
    end
    return E
end

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