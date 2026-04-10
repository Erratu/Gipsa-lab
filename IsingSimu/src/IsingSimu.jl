module IsingSimu

using LinearAlgebra, Statistics, Interpolations, SparseArrays # Ising
using Optim, JuMP, Ipopt, Zygote, Manopt, Random, ADTypes # Thomson

include("func_ising.jl")
include("ising_structures.jl")
include("pt_gen_models.jl")
include("ising_models.jl")

include("min_e_thomson.jl")
include("thomson_functions.jl")
include("thomson_algos.jl")
include("thomson_algos_optim.jl")

export adjacency_1d, adjacency_2d, energy, heat_bath_ising!, 
algo_wolff_complet!, algo_wl_complet!, algo_pt_complet!,  algo4_nrpt_complet!, 
PTMetrics, PTState, PTSampler, PTHistory, empty_history!,
solve_pgd, solve_pgd_momentum, solve_lbfgs_penalized, solve_jump_ipopt, solve_pt, solve_manifold_lbfgs,
solve_pgd_fast, solve_pgd_momentum_fast, solve_pt_fast, solve_manifold_lbfgs_fast, solve_nrpt_fast, solve_basin_hopping_fast,
random_configuration, get_min_thomson_energy, thomson_energy, thomson_energy_and_grad!, thomson_energy_fast, thomson_energy_with_potential

end