function init_fixed_symmetry(n::Int)
    X = random_configuration(n)
    
    # Point 1 au pôle nord
    X[:, 1] .= [0.0, 0.0, 1.0]
    
    # Point 2 avec azimut 0 (y = 0) et x positif.
    z2 = clamp(X[3, 2], -1.0, 1.0)
    x2 = sqrt(1.0 - z2^2)
    X[:, 2] .= [x2, 0.0, z2]
    
    return X
end

function project_fixed_sphere!(X::Matrix{Float64})
    for i in 1:size(X, 2)
        X[:, i] ./= norm(X[:, i])
    end
    
    # contraintes de symétrie
    X[:, 1] .= [0.0, 0.0, 1.0]
    
    # Pour le point 2, on annule y et on re-normalise
    X[2, 2] = 0.0
    X[1, 2] = abs(X[1, 2]) # Forcer x > 0 pour l'azimut 0
    X[:, 2] ./= norm(X[:, 2])
    
    return X
end

function thomson_update_fixed!(X::Matrix{Float64}, beta::Float64; step_size=0.08) 
    n = size(X,2) 
    for _ in 1:n 
        i = rand(1:n) 
        
        # On ne bouge jamais le point 1 (Pôle Nord)
        if i == 1
            continue
        end
        
        old = X[:,i] 
        proposal = old + step_size * randn(3) 
        
        # Si c'est le point 2, on contraint le mouvement dans le plan y=0
        if i == 2
            proposal[2] = 0.0
            proposal[1] = abs(proposal[1]) 
        end
        
        proposal /= norm(proposal) 
        
        ΔE = delta_energy(X, i, proposal) 
        
        if ΔE <= 0 || rand() < exp(-beta * ΔE) 
            X[:,i] = proposal 
        end 
    end 
    return X 
end

function quench_lbfgs_sph_fixed(X_init::Matrix{Float64})
    @info "Quenching L-BFGS (Symétrie O(3) brisée : 2N-3 ddl)"
    n = size(X_init, 2)
    
    angles_init = zeros(2n - 3)
    
    angles_init[1] = acos(clamp(X_init[3, 2], -1.0, 1.0))
    
    idx = 2
    for i in 3:n
        x, y, z = X_init[1, i], X_init[2, i], X_init[3, i]
        angles_init[idx]   = acos(clamp(z, -1.0, 1.0)) # theta
        angles_init[idx+1] = atan(y, x)                # phi
        idx += 2
    end
    
    function objective(angles)
        # Détecte dynamiquement le type (Float64 ou ForwardDiff.Dual)
        T = eltype(angles) 
        
        # Point 1 (Fixe)
        p1_x, p1_y, p1_z = T(0.0), T(0.0), T(1.0)
        
        # Point 2 (Méridien fixe, y=0)
        theta_2 = angles[1]
        p2_x, p2_y, p2_z = sin(theta_2), T(0.0), cos(theta_2)
        
        pts_x = Vector{T}(undef, n)
        pts_y = Vector{T}(undef, n)
        pts_z = Vector{T}(undef, n)
        
        pts_x[1], pts_y[1], pts_z[1] = p1_x, p1_y, p1_z
        pts_x[2], pts_y[2], pts_z[2] = p2_x, p2_y, p2_z
        
        E = 1.0 / sqrt((p1_x - p2_x)^2 + (p1_y - p2_y)^2 + (p1_z - p2_z)^2)
        
        a_idx = 2
        for i in 3:n
            t_i, p_i = angles[a_idx], angles[a_idx+1]
            pts_x[i] = sin(t_i) * cos(p_i)
            pts_y[i] = sin(t_i) * sin(p_i)
            pts_z[i] = cos(t_i)
            a_idx += 2
            
            for j in 1:i-1
                dist = sqrt((pts_x[i] - pts_x[j])^2 + (pts_y[i] - pts_y[j])^2 + (pts_z[i] - pts_z[j])^2)
                E += 1.0 / dist
            end
        end
        return E
    end
    
    result = optimize(objective, angles_init, LBFGS(); autodiff = AutoForwardDiff())
    angles_opt = Optim.minimizer(result)
    
    # Reconstruire la matrice finale
    X_opt = zeros(3, n)
    X_opt[:, 1] .= [0.0, 0.0, 1.0]
    X_opt[:, 2] .= [sin(angles_opt[1]), 0.0, cos(angles_opt[1])]
    
    a_idx = 2
    for i in 3:n
        t_i, p_i = angles_opt[a_idx], angles_opt[a_idx+1]
        X_opt[1, i] = sin(t_i) * cos(p_i)
        X_opt[2, i] = sin(t_i) * sin(p_i)
        X_opt[3, i] = cos(t_i)
        a_idx += 2
    end
    
    return X_opt
end

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

function project_sphere_old!(X::Matrix{Float64})
    for i in 1:size(X, 2)
        X[:, i] ./= norm(X[:, i])
    end
    return X
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

function quench_lbfgs_sph(X_init::Matrix{Float64})
    @info "Quenching avec L-BFGS (Coordonnées Sphériques)"
    n = size(X_init, 2)
    
    angles_init = zeros(2 * n)
    for i in 1:n
        x, y, z = X_init[1, i], X_init[2, i], X_init[3, i]
        angles_init[2*i - 1] = acos(clamp(z, -1.0, 1.0)) # theta
        angles_init[2*i]     = atan(y, x)                # phi
    end
    
    function objective(angles)
        E = 0.0
        for i in 1:n-1
            theta_i = angles[2*i - 1]
            phi_i   = angles[2*i]
            # Les coordonnées (x,y,z) sont implicitement sur la sphère de rayon 1
            xi = sin(theta_i) * cos(phi_i)
            yi = sin(theta_i) * sin(phi_i)
            zi = cos(theta_i)
            
            for j in i+1:n
                theta_j = angles[2*j - 1]
                phi_j   = angles[2*j]
                xj = sin(theta_j) * cos(phi_j)
                yj = sin(theta_j) * sin(phi_j)
                zj = cos(theta_j)
                
                # Distance euclidienne
                dx = xi - xj
                dy = yi - yj
                dz = zi - zj
                dist = sqrt(dx^2 + dy^2 + dz^2)
                E += 1.0 / dist
            end
        end
        return E
    end
    
    result = optimize(objective, angles_init, LBFGS(); autodiff = AutoForwardDiff())
    
    angles_opt = Optim.minimizer(result)
    
    X_opt = zeros(3, n)
    for i in 1:n
        theta = angles_opt[2*i - 1]
        phi   = angles_opt[2*i]
        X_opt[1, i] = sin(theta) * cos(phi)
        X_opt[2, i] = sin(theta) * sin(phi)
        X_opt[3, i] = cos(theta)
    end
    
    @info "L-BFGS terminé."
    @info "Détails : $(Optim.converged(result)) en $(Optim.iterations(result)) itérations."
    
    return X_opt
end

function quench_to_zero(X::Matrix{Float64}; max_steps=5000, initial_step=0.01, final_step=1e-7)
    @info "Quenching avec diminution du pas"
    X_opt = copy(X)
    n = size(X_opt, 2)
    
    decay_rate = (final_step / initial_step)^(1.0 / max_steps)
    current_step = initial_step
    
    for _ in 1:max_steps
        for _ in 1:n
            i = rand(1:n)
            old = X_opt[:,i]
            
            proposal = old + current_step * randn(3)
            proposal /= norm(proposal)
            
            if delta_energy_fast(X_opt, i, proposal) < 0
                X_opt[:,i] = proposal
            end
        end
        current_step *= decay_rate 
    end
    return X_opt
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
