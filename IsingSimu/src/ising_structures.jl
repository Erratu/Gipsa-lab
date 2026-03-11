mutable struct PTSampler
    nb_sim::Int
    betas::Vector{Float64}

    function PTSampler(betas::Vector{Float64})
        new(length(betas), copy(betas))
    end
end

mutable struct PTHistory
    trajectoires::Vector{Vector{Int}}
    obs_hist::Vector{Vector{Float64}}
    ene_hist::Vector{Vector{Float64}}
    rt_hist::Vector{Float64}

    function PTHistory(nb_sim::Int)
        new(
            [Int[] for _ in 1:nb_sim],
            [Float64[] for _ in 1:nb_sim],
            [Float64[] for _ in 1:nb_sim],
            Float64[]
        )
    end
end

function empty_history!(hist::PTHistory)
    for i in 1:length(hist.obs_hist)
        empty!(hist.trajectoires[i])
        empty!(hist.obs_hist[i])
        empty!(hist.ene_hist[i])
    end
    empty!(hist.rt_hist)
end

mutable struct PTMetrics
    round_trips::Int
    is_going_down::Vector{Bool}
    swap_rejections::Vector{Float64}

    function PTMetrics(nb_sim::Int)
        new(0, fill(false, nb_sim), zeros(Float64, nb_sim - 1))
    end
end

mutable struct PTState{T}
    sims::Vector{T}              
    indices::Vector{Int}         
    metrics::PTMetrics
    history::PTHistory  

    function PTState{T}(initial_states::Vector{T}) where T
        nb_sim = length(initial_states)
        new{T}(
            copy(initial_states), 
            collect(1:nb_sim), 
            PTMetrics(nb_sim), 
            PTHistory(nb_sim)
        )
    end
end