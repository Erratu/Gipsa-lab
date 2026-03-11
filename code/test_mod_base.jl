include("ising_structures.jl")
include("pt_gen_models.jl")

using Distributions, Plots, Random

gauss_energy(x::Float64) = 0.5 * x^2

gauss_obs(x::Float64) = x

function gauss_local_update!(x::Float64, beta::Float64; step_size::Float64=2.0)
    x_prop = x + randn() * step_size
    
    E_old = gauss_energy(x)
    E_new = gauss_energy(x_prop)
    
    if rand() < exp(-beta * (E_new - E_old))
        return x_prop
    end
    return x
end

Random.seed!(42)
betas_test = [0.05, 0.1, 0.25, 0.5, 1.0] 
nb_sim = length(betas_test)

sampler_pt = PTSampler(betas_test)
sampler_nrpt = PTSampler(copy(betas_test)) 

initial_states = randn(Float64, nb_sim) * 5.0 
state_pt = PTState{Float64}(copy(initial_states))
state_nrpt = PTState{Float64}(copy(initial_states))

algo_pt_complet!(sampler_pt, state_pt, gauss_energy, gauss_local_update!, gauss_obs, 5000, 20000)
algo4_nrpt_complet!(sampler_nrpt, state_nrpt, gauss_energy, gauss_local_update!, gauss_obs, 4000, 20000)

samples_pt = state_pt.history.obs_hist[end]
samples_nrpt = state_nrpt.history.obs_hist[end]

x_grid = range(-4, 4, length=200)
y_true = pdf.(Normal(0,1), x_grid)

p1 = histogram(samples_pt, normalize=:pdf, bins=50, alpha=0.5, label="PT Echantillons", title="Test PT Classique")
plot!(p1, x_grid, y_true, linewidth=3, label="Vraie N(0,1)")

p2 = histogram(samples_nrpt, normalize=:pdf, bins=50, alpha=0.5, label="NRPT Echantillons", title="Test NRPT")
plot!(p2, x_grid, y_true, linewidth=3, label="Vraie N(0,1)")

plt = plot(p1, p2, layout=(2,1), size=(800, 600))

mkpath("graphiques")
savefig(plt, "graphiques/test_gauss_pt_nrpt.png")

display(plt)