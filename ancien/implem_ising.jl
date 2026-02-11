include("visualisation.jl")
using LinearAlgebra

# Heat bath
function heat_bath_ising(Q,X,m,n,nb_it,beta,Xobs)

    for t in 1:nb_it
        i = rand(1:n)
        #alpha = beta*(dot(view(Q, i, :), X) - Q[i,i]*X[i] + m[i])
        #p = exp(-alpha)/(exp(-alpha)+exp(alpha))

        delta_E = 2 * (dot(view(Q, i, :), X) - Q[i,i]*X[i] + m[i])
        p = 1 / (1 + exp(beta * delta_E))
        if rand() < p
            X[i] = 1
        else 
            X[i] = -1
        end

        if t % 10 == 0
            update_ising_view!(Xobs, X)
            sleep(0.1)
        end

    end

    return X
end

# Appel de Heat bath pour Ising 
beta = 0.5
n = 10
X0 = rand([-1,1],n)
Q = rand([0,1],n,n)
Q = Q .| Q'
diagind = LinearAlgebra.diagind(Q)
Q[diagind] .= 0
m = Array{Int8}(undef,n)

_, Xobs = init_replicas_grid(Q, [X0])
X = heat_bath_ising(Q,X0,m,n,200,beta,only(Xobs))

# Parallel tempering
function parallel_tempering_ising(Q,m,n,nb_it,nb_sim)
    sims = [rand([-1,1], n) for _ in 1:nb_sim]
    betas = 10 .^ range(-2, 1, length=nb_sim)
    E = zeros(Float64, nb_sim)
    _, Xobs = init_replicas_grid(Q, sims)
    for i in 1:nb_it
        for sim in 1:nb_sim
            sims[sim] = heat_bath_ising(Q,sims[sim],m,n,n, betas[sim], Xobs[sim])
            X = sims[sim]
            # calcul de l'énergie
            E[sim] = -0.5*dot(X, Q, X) - dot(m,X)
        end
        # échange entre les simulations de différentes températures
        for sim in 1:nb_sim-1
            if rand() < exp((betas[sim]-betas[sim+1])*(E[sim]-E[sim+1]))
                sims[sim], sims[sim+1] = sims[sim+1], sims[sim]
                E[sim], E[sim+1] = E[sim+1], E[sim]
            end
        end
    end
    return sims[nb_sim]
end

X = parallel_tempering_ising(Q,m,n,200,3)
