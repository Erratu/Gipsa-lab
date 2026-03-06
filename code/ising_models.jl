include("ising_model_2.jl")

function run_hb!(state::ChainState, Q, m_field)
    n = state.L^2
    heat_bath_ising!(state.sims[1], Q, m_field, state.betas[1], n)
    push!(state.mag_hist[1], abs(sum(state.sims[1])/n))
    push!(state.ene_hist[1], energy(state.sims[1], Q, m_field)/n)
end

function run_pt!(state::ChainState, Q, m_field)
    n = state.L^2
    for s in 1:state.nb_sim
        heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n)
    end
    
    for _ in 1:state.nb_sim
        i = rand(1:state.nb_sim-1)
        E1 = energy(state.sims[i], Q, m_field)
        E2 = energy(state.sims[i+1], Q, m_field)
        
        if rand() < exp((state.betas[i] - state.betas[i+1]) * (E1 - E2))
            state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
            state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
        end
    end
    
    for i in 1:state.nb_sim
        idx = state.indices[i]
        if i == state.nb_sim
            state.is_going_down[idx] = true
        elseif i == 1 && state.is_going_down[idx]
            state.is_going_down[idx] = false
            state.round_trips += 1
        end
    end
    
    for s in 1:state.nb_sim
        push!(state.mag_hist[s], abs(sum(state.sims[s])/n))
        push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n)
        push!(state.trajectoires[s], findfirst(==(s), state.indices))
    end
end

function algo1_deo!(state::ChainState, Q, m_field, t)
    n = state.L^2
    for s in 1:state.nb_sim
        heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n)
    end

    offset = (t % 2 == 0) ? 2 : 1
    
    for i in 1:(state.nb_sim - 1)
        E1 = energy(state.sims[i], Q, m_field)
        E2 = energy(state.sims[i+1], Q, m_field)
        alpha = min(1.0, exp((state.betas[i] - state.betas[i+1]) * (E1 - E2)))
        
        state.swap_rejections[i] += (1.0 - alpha)
        
        if i % 2 == offset % 2
            if rand() < alpha
                state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
            end
        end
    end

    for i in 1:state.nb_sim
        idx = state.indices[i]
        if i == state.nb_sim
            state.is_going_down[idx] = true
        elseif i == 1 && state.is_going_down[idx]
            state.is_going_down[idx] = false
            state.round_trips += 1
        end
    end

    for s in 1:state.nb_sim
        push!(state.mag_hist[s], abs(sum(state.sims[s])/n))
        push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n)
        push!(state.trajectoires[s], findfirst(==(s), state.indices))
    end
end

function algo2_updateschedule(Lambda_hat, betas, N)
    Lambda_max = Lambda_hat[end]
    new_betas = zeros(Float64, N)
    new_betas[1] = betas[1]
    new_betas[end] = betas[end]
    
    for k in 2:(N-1)
        target = (k - 1) / (N - 1) * Lambda_max
        idx = findfirst(x -> x >= target, Lambda_hat)
        if idx == 1
            new_betas[k] = betas[1]
        elseif idx === nothing
            new_betas[k] = betas[end]
        else
            w = (target - Lambda_hat[idx-1]) / (Lambda_hat[idx] - Lambda_hat[idx-1])
            new_betas[k] = betas[idx-1] + w * (betas[idx] - betas[idx-1])
        end
    end
    return new_betas
end

function algo3_communicationbarrier(state::ChainState)
    nb_sim = state.nb_sim
    Lambda_hat = zeros(Float64, nb_sim)
    Lambda_hat[1] = 0.0
    for i in 2:nb_sim
        Lambda_hat[i] = Lambda_hat[i-1] + state.swap_rejections[i-1]
    end
    return Lambda_hat
end

function algo4_nrpt!(state::ChainState, Q, m_field, t)
    algo1_deo!(state, Q, m_field, t)
    
    freq = 100
    if t % freq == 0
        state.swap_rejections ./= freq
        Lambda_hat = algo3_communicationbarrier(state)
        state.betas = algo2_updateschedule(Lambda_hat, state.betas, state.nb_sim)
        fill!(state.swap_rejections, 0.0)
    end
end