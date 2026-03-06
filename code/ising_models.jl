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
        
        state.swaps_attempted[i] += 1
        
        if rand() < exp((state.betas[i] - state.betas[i+1]) * (E1 - E2))
            state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
            state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
            state.swaps_accepted[i] += 1
        end
    end
    for s in 1:state.nb_sim
        push!(state.mag_hist[s], abs(sum(state.sims[s])/n))
        push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n)
        push!(state.trajectoires[s], findfirst(==(s), state.indices))
    end
end

function algo1_exploration!(state::ChainState, Q, m_field)
    n = state.L^2
    for s in 1:state.nb_sim
        heat_bath_ising!(state.sims[s], Q, m_field, state.betas[s], n)
    end
end

function algo2_communication!(state::ChainState, Q, m_field, t)
    offset = (t % 2 == 0) ? 1 : 2
    for i in 1:(state.nb_sim - 1)
        E1 = energy(state.sims[i], Q, m_field)
        E2 = energy(state.sims[i+1], Q, m_field)
        alpha = min(1.0, exp((state.betas[i] - state.betas[i+1]) * (E1 - E2)))
        
        state.swap_rejections[i] += (1.0 - alpha)
        
        if i % 2 == offset % 2
            state.swaps_attempted[i] += 1
            if rand() < alpha
                state.sims[i], state.sims[i+1] = state.sims[i+1], state.sims[i]
                state.indices[i], state.indices[i+1] = state.indices[i+1], state.indices[i]
                state.swaps_accepted[i] += 1
            end
        end
    end
end

function algo3_adaptation!(state::ChainState, t)
    freq = 100
    if t % freq == 0
        rates = state.swap_rejections ./ freq
        target_rej = sum(rates) / (state.nb_sim - 1)
        
        new_betas = copy(state.betas)
        gamma = 1.0 / sqrt(t)
        
        for i in 2:(state.nb_sim - 1)
            diff = rates[i-1] - target_rej
            proposed_beta = state.betas[i] + gamma * diff * (state.betas[i+1] - state.betas[i-1])
            
            new_betas[i] = max(state.betas[i-1] + 1e-5, min(state.betas[i+1] - 1e-5, proposed_beta))
        end
        
        state.betas .= new_betas
        fill!(state.swap_rejections, 0.0)
    end
end

function algo4_nrpt!(state::ChainState, Q, m_field, t)
    algo1_exploration!(state, Q, m_field)
    algo2_communication!(state, Q, m_field, t)
    algo3_adaptation!(state, t)
    
    n = state.L^2
    for s in 1:state.nb_sim
        push!(state.mag_hist[s], abs(sum(state.sims[s])/n))
        push!(state.ene_hist[s], energy(state.sims[s], Q, m_field)/n)
        push!(state.trajectoires[s], findfirst(==(s), state.indices))
    end
end