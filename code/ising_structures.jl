using Statistics

mutable struct ChainState
    L::Int
    nb_sim::Int
    betas::Vector{Float64}
    sims::Vector{Vector{Int}}
    indices::Vector{Int}
    trajectoires::Vector{Vector{Int}}
    mag_hist::Vector{Vector{Float64}}
    ene_hist::Vector{Vector{Float64}}
    swap_rejections::Vector{Float64}
    round_trips::Int
    is_going_down::Vector{Bool}
    rt_hist::Vector{Float64}

    function ChainState(L, betas)
        n = L * L
        nb_sim = length(betas)
        sims = [rand([-1, 1], n) for _ in 1:nb_sim]
        indices = collect(1:nb_sim)
        trajectoires = [Int[] for _ in 1:nb_sim]
        mag_hist = [Float64[] for _ in 1:nb_sim]
        ene_hist = [Float64[] for _ in 1:nb_sim]
        swap_rejections = zeros(Float64, nb_sim - 1)
        round_trips = 0
        is_going_down = fill(false, nb_sim)
        rt_hist = Float64[]
        new(L, nb_sim, betas, sims, indices, trajectoires, mag_hist, ene_hist, swap_rejections, round_trips, is_going_down, rt_hist)
    end
end