#!/usr/bin/env julia
# benchmark.jl — Quick standalone benchmark comparing local & global optimizers
# Usage:
#   julia --project=ThomsonMeths benchmark.jl [m]

using Pkg
Pkg.develop(path="PTShared")

using ThomsonMeths
using Printf
using Random

Random.seed!(42)

m = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
n_reservoir = m + 20

println("="^68)
println("  THOMSON PROBLEM BENCHMARK — $m PARTICLES ON S²")
println("="^68)
println("Generating random initial configuration...")
X0 = random_configuration(m)
E_init = thomson_energy(X0)
@printf("Initial random energy: %12.6f\n\n", E_init)

# 1. PGD
t0 = time()
X_pgd, _ = solve_pgd_fast(copy(X0), lr=0.005, max_iter=2000)
t_pgd = time() - t0
E_pgd = thomson_energy_fast(X_pgd)

# 2. PGD + Momentum
t0 = time()
X_mom, _ = solve_pgd_momentum_fast(copy(X0), lr=0.005, max_iter=2000)
t_mom = time() - t0
E_mom = thomson_energy_fast(X_mom)

# 3. Manifold L-BFGS
t0 = time()
X_lbfgs, _ = solve_manifold_lbfgs_fast(copy(X0), max_iter=500)
t_lbfgs = time() - t0
E_lbfgs = thomson_energy_fast(X_lbfgs)

# 4. SGD + Combinatorial Relaxation (Our method)
t0 = time()
X_pool = random_configuration(n_reservoir)
X_opt, _, _ = sgd_loo(X_pool, m, n_reservoir; strategy=:boltzmann, max_iter=1500, lr=0.01)
t_sgd = time() - t0
E_sgd = thomson_energy(X_opt[:, 1:m])

# Global known reference
E_ref = get_min_thomson_energy(m)

println("-"^68)
@printf("%-28s | %12s | %10s | %8s\n", "Method", "Energy", "Δ to Ref", "Time (s)")
println("-"^68)
@printf("%-28s | %12.6f | %+10.6f | %8.3f\n", "1. PGD", E_pgd, E_pgd - E_ref, t_pgd)
@printf("%-28s | %12.6f | %+10.6f | %8.3f\n", "2. PGD + Momentum", E_mom, E_mom - E_ref, t_mom)
@printf("%-28s | %12.6f | %+10.6f | %8.3f\n", "3. Manifold L-BFGS", E_lbfgs, E_lbfgs - E_ref, t_lbfgs)
@printf("%-28s | %12.6f | %+10.6f | %8.3f\n", "4. Ours (SGD+Boltzmann)", E_sgd, E_sgd - E_ref, t_sgd)
println("-"^68)
@printf("%-28s | %12.6f | %10s | %8s\n", "Reference Global Minimum", E_ref, "0.000000", "-")
println("="^68)
