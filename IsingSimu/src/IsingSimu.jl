module IsingSimu

using LinearAlgebra, Statistics, Interpolations, SparseArrays

include("func_ising.jl")
include("ising_structures.jl")
include("pt_gen_models.jl")
include("ising_models.jl")

export adjacency_1d, adjacency_2d, energy, heat_bath_ising!, algo_wolff_complet!, algo_wl_complet!, PTSampler, PTHistory, empty_history!, PTMetrics, PTState, algo_pt_complet!,  algo4_nrpt_complet!

end