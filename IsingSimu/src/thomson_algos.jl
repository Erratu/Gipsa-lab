# PGD
function solve_pgd(X0::Matrix{Float64}; lr=0.01, max_iter=2000)
    X = copy(X0)
    history = Float64[]
    for _ in 1:max_iter
        push!(history, thomson_energy(X))
        G = Zygote.gradient(thomson_energy, X)[1]
        X .-= lr .* G
        project_sphere!(X)
    end
    return X, history
end

# PGD avec Momentum
function solve_pgd_momentum(X0::Matrix{Float64}; lr=0.01, beta=0.9, max_iter=2000)
    X = copy(X0)
    V = zeros(size(X))
    history = Float64[]
    for _ in 1:max_iter
        push!(history, thomson_energy(X))
        G = Zygote.gradient(thomson_energy, X)[1]
        V .= beta .* V .+ (1 - beta) .* G
        X .-= lr .* V
        project_sphere!(X)
    end
    return X, history
end

# L-BFGS pénalisé : peut-être changer le lambda au fil des itérations ?
function solve_lbfgs_penalized(X0::Matrix{Float64}; lambda=100.0, max_iter=3000)
    n = size(X0, 2)
    
    # fonction objectif pour Optim (prend un vecteur 1D)
    function f(x_vec)
        X = reshape(x_vec, 3, n)
        E = thomson_energy(X)
        penalty = sum(lambda * (norm(X[:, i])^2 - 1.0)^2 for i in 1:n)
        return E + penalty
    end
    
    # gradient
    function g!(G_vec, x_vec)
        G_vec .= Zygote.gradient(f, x_vec)[1]
        return G_vec
    end
    
    # L-BFGS
    options = Optim.Options(iterations=max_iter, store_trace=true, extended_trace=true)
    x0_vec = vec(copy(X0))
    res = optimize(f, g!, x0_vec, LBFGS(), options)
    
    # config finale
    X_opt = reshape(Optim.minimizer(res), 3, n)
    project_sphere!(X_opt) # projection finale
    
    history = Float64[]
    for t in Optim.trace(res)
        x_iter = reshape(t.metadata["x"], 3, n)
        push!(history, thomson_energy(x_iter))
    end
    
    return X_opt, history
end

# JuMP avec Ipopt : point intérieur
function solve_jump_ipopt(X0::Matrix{Float64})
    n = size(X0, 2)
    model = Model(Ipopt.Optimizer)
    set_silent(model)
    
    @variable(model, x[1:3, 1:n])
    for i in 1:3, j in 1:n
        set_start_value(x[i,j], X0[i,j])
    end
    
    @constraint(model, [j=1:n], x[1,j]^2 + x[2,j]^2 + x[3,j]^2 == 1.0)
    
    @objective(model, Min, sum(
        1.0 / sqrt((x[1,i] - x[1,j])^2 + (x[2,i] - x[2,j])^2 + (x[3,i] - x[3,j])^2) 
        for i in 1:n-1, j in i+1:n
    ))
    
    optimize!(model)
    return value.(x), Float64[] # Pas d'historique
end

# PT avec quench
function solve_pt(n::Int; n_burnin=4000, n_sample=10000)
    n_replicas = 16 
    betas = collect(range(0.05, 15.0, length=n_replicas)) 
    sampler = PTSampler(betas) 
    
    initial_states = [random_configuration(n) for _ in 1:n_replicas] 
    state = PTState{Matrix{Float64}}(initial_states) 
    
    e_func = X -> thomson_energy(X) 
    u_func = (X, b) -> thomson_update!(X, b, step_size=0.08) 
    obs_func = X -> thomson_energy(X) 
    
    algo_pt_complet!(sampler, state, e_func, u_func, obs_func, n_burnin, n_sample) 

    best_X = quench_lbfgs_sph(state.global_best_X)

    return best_X, Float64[] 
end

function solve_manifold_lbfgs(X0::Matrix{Float64}; max_iter=2000)
    n = size(X0, 2)
    M = Manif.PowerManifold(Manif.Sphere(2), Manif.ArrayPowerRepresentation(), n)

    # fonction objectif
    f(M, X) = thomson_energy(X)

    # gradient riemannien
    function grad_f(M, X)
        G_eucl = Zygote.gradient(thomson_energy, X)[1]
        return Manif.project(M, X, G_eucl) 
    end

    state = quasi_Newton(M, f, grad_f, copy(X0);
        memory_size = 15,
        stopping_criterion = StopAfterIteration(max_iter) | StopWhenGradientNormLess(1e-6),
        record = [:Cost],
        return_state = true
    )

    X_opt = get_solver_result(state)
    history = get_record(state)
    
    return X_opt, history
end
