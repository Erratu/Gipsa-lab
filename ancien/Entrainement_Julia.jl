#using LinearAlgebra → matrices, valeurs propres, normes
#Statistics → mean, std, var
#Random → reproductibilité (seed!) Random.seed!(42)

#using BenchmarkTools
#@btime somme(1000) -> comparer l'optimisation

# y = A * x non -> mul!(y, A, x) allocation mémoire 
# x = A \ b -> inv(A) * b non, solveur linéaire

# Optimisation : 
#using JuMP, HiGHS
#model = Model(HiGHS.Optimizer)
#@variable(model, x >= 0)
#@objective(model, Min, x^2)
#optimize!(model)
#value(x)

# JuMP : langage de modélisation mathématique
# HiGHS : solveur rapide (quadratique / linéaire)

using JuMP, HiGHS

grid = range(-2, 2, length=50)
kmax = 5

# Matrice des moments
M = [x^k for k in 0:kmax, x in grid]

# Moments observés
μ = rand(kmax+1)

model = Model(HiGHS.Optimizer)
@variable(model, w[1:length(grid)] >= 0)
@constraint(model, sum(w) == 1)
@objective(model, Min, sum((M*w - μ).^2))

optimize!(model)
