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
        X[:, i] ./= norm(X[:, i])
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

function thomson_update!(X::Matrix{Float64}, beta::Float64; step_size=0.08) 
    n = size(X,2) 
    for _ in 1:n 
        i = rand(1:n) 
        old = X[:,i] # [cite: 3]
        
        proposal = old + step_size * randn(3) # [cite: 3]
        proposal /= norm(proposal) # [cite: 3]
        
        ΔE = delta_energy(X, i, proposal) # [cite: 3]
        
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
            
            if delta_energy(X_opt, i, proposal) < 0
                X_opt[:,i] = proposal
            end
        end
        current_step *= decay_rate 
    end
    return X_opt
end