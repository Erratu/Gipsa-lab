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

export adjacency_1d, adjacency_2d, energy, heat_bath_ising!, 
algo_wolff_complet!, algo_wl_complet!, algo_pt_complet!,  algo4_nrpt_complet!, 
PTMetrics, PTState, PTSampler, PTHistory, empty_history!,
solve_pgd, solve_pgd_momentum, solve_lbfgs_penalized, solve_jump_ipopt, solve_pt, solve_manifold_lbfgs, solve_jump_ipopt_fixed,
random_configuration, get_min_thomson_energy, thomson_energy,
init_fixed_symmetry, project_fixed_sphere!, thomson_update_fixed!, quench_lbfgs_sph_fixed, thomson_energy_with_potential

end