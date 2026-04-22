module ThomsonMeths

using LinearAlgebra, Statistics, Interpolations, SparseArrays # Ising
using Optim, JuMP, Ipopt, Zygote, Manopt, Random, ADTypes # Thomson
using Printf, Combinatorics, Plots, StatsBase, Clustering # New meth

include("min_e_thomson.jl")
include("thomson_functions.jl")
include("thomson_algos_optim.jl")

include("heuristics_n_m.jl")
include("samplers_q.jl")
include("optimizers_n.jl")
include("tempering_setup.jl")

export solve_pgd, solve_pgd_momentum, solve_lbfgs_penalized, solve_jump_ipopt, solve_pt, solve_manifold_lbfgs,
solve_pgd_fast, solve_pgd_momentum_fast, solve_pt_fast, solve_manifold_lbfgs_fast, solve_nrpt_fast, solve_basin_hopping_fast,

random_configuration, get_min_thomson_energy, thomson_energy, thomson_energy_and_grad!, thomson_energy_fast, thomson_energy_with_potential,

evaluer_energie, matrice_interactions, sgd_loo_optim, sgd_loo, 
take_random_subset_bp, take_random_subset_boltzmann, take_random_subset_dpp,
methode_exhaustive, methode_heuristique, methode_relaxation_jump, 
methode_belief_propagation, methode_belief_propagation_post_process, 
methode_gumbel_softmax, sgd_diffusion_measure, setup_pt_thomson

end