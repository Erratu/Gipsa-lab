### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ Cell 1
md"""
# Thomson Problem — Combinatorial Relaxation Demo

This notebook demonstrates the key numerical methods implemented in the repository:
1. **Local vs. Global optimizers** on the Thomson problem
2. **The combinatorial relaxation** framework with subset sampling heuristics
3. **Configuration visualization** on the unit sphere
"""

# ╔═╡ Cell 2
begin
    using Pkg
    Pkg.develop(path="../PTShared")
    Pkg.develop(path="../ThomsonMeths")
end

# ╔═╡ Cell 3
using ThomsonMeths
using Plots, LinearAlgebra, Random
Random.seed!(42)

# ╔═╡ Cell 4
md"""
## 1. Comparing Local Optimizers
"""

# ╔═╡ Cell 5
begin
    m_pts = 20
    X_init = random_configuration(m_pts)

    # Projected Gradient Descent
    X_pgd, h_pgd = solve_pgd_fast(copy(X_init), lr=0.005, max_iter=2000)

    # PGD with Momentum
    X_mom, h_mom = solve_pgd_momentum_fast(copy(X_init), lr=0.005, max_iter=2000)

    # Riemannian L-BFGS
    X_lbfgs, h_lbfgs = solve_manifold_lbfgs_fast(copy(X_init), max_iter=400)

    println("Known minimum for m=$m_pts: ", get_min_thomson_energy(m_pts))
    println("PGD energy:      ", round(thomson_energy_fast(X_pgd), digits=6))
    println("Momentum energy: ", round(thomson_energy_fast(X_mom), digits=6))
    println("L-BFGS energy:   ", round(thomson_energy_fast(X_lbfgs), digits=6))
end

# ╔═╡ Cell 6
begin
    plot(h_pgd, label="PGD", lw=2, xlabel="Iteration", ylabel="Energy",
         title="Convergence of local optimizers (m=$m_pts)")
    plot!(h_mom, label="PGD+Momentum", lw=2)
    plot!(h_lbfgs, label="Manifold L-BFGS", lw=2)
    hline!([get_min_thomson_energy(m_pts)], label="Global min", ls=:dash, color=:red, lw=2)
end

# ╔═╡ Cell 7
md"""
## 2. Combinatorial Relaxation via SGD
"""

# ╔═╡ Cell 8
begin
    n_res = 40
    X_pool = random_configuration(n_res)

    # SGD with uniform sampling
    X_uni, h_uni, _ = sgd_loo(copy(X_pool), m_pts, n_res,
        strategy=:uniform, max_iter=1500, lr=0.01)

    # SGD with Boltzmann heuristic
    X_boltz, h_boltz, _ = sgd_loo(copy(X_pool), m_pts, n_res,
        strategy=:boltzmann, max_iter=1500, lr=0.01)

    # SGD with Belief Propagation heuristic
    X_bp, h_bp, _ = sgd_loo(copy(X_pool), m_pts, n_res,
        strategy=:bp, max_iter=1500, lr=0.01)

    plot(h_uni, label="Uniform", lw=2, xlabel="Iteration (x10)",
         ylabel="Subset Energy", title="SGD on G_n(Y) - Heuristic comparison")
    plot!(h_boltz, label="Boltzmann", lw=2)
    plot!(h_bp, label="Belief Propagation", lw=2)
end

# ╔═╡ Cell 9
begin
    scatter3d(X_boltz[1, :], X_boltz[2, :], X_boltz[3, :],
        markersize=4, label="Reservoir (n=$n_res)",
        xlabel="x", ylabel="y", zlabel="z",
        title="Point configuration on S²", camera=(30, 30))
end
