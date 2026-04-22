module IsingSimu

using LinearAlgebra, Statistics, Interpolations, SparseArrays # Ising


include("func_ising.jl")
include("ising_models.jl")


export adjacency_1d, adjacency_2d, energy, heat_bath_ising!, 
algo_wolff_complet!, algo_wl_complet!

end