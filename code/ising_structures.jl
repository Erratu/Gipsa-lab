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
    swaps_accepted::Vector{Int}
    swaps_attempted::Vector{Int}

    function ChainState(L, betas)
        n = L * L
        nb_sim = length(betas)
        sims = [rand([-1, 1], n) for _ in 1:nb_sim]
        indices = collect(1:nb_sim)
        trajectoires = [Int[] for _ in 1:nb_sim]
        mag_hist = [Float64[] for _ in 1:nb_sim]
        ene_hist = [Float64[] for _ in 1:nb_sim]
        swap_rejections = zeros(Float64, nb_sim - 1)
        swaps_accepted = zeros(Int, nb_sim - 1)
        swaps_attempted = zeros(Int, nb_sim - 1)
        new(L, nb_sim, betas, sims, indices, trajectoires, mag_hist, ene_hist, swap_rejections, swaps_accepted, swaps_attempted)
    end
end