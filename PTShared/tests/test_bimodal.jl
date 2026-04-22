using IsingSimu

using Distributions, Plots, Random

const d1 = Normal(-5.0, 1.0)
const d2 = Normal(5.0, 1.0)

function bimodal_energy(x::Float64)
    p = 0.5 * pdf(d1, x) + 0.5 * pdf(d2, x)
    return -log(max(p, 1e-300))
end

bimodal_obs(x::Float64) = x

function bimodal_local_update!(x::Float64, beta::Float64; step_size::Float64=2.0)
    x_prop = x + randn() * step_size
    
    E_old = bimodal_energy(x)
    E_new = bimodal_energy(x_prop)
    
    if rand() < exp(-beta * (E_new - E_old))
        return x_prop
    end
    return x
end

Random.seed!(42)
betas_test = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0] 
nb_sim = length(betas_test)

sampler_pt = PTSampler(betas_test)
sampler_nrpt = PTSampler(copy(betas_test)) 

initial_states = fill(5.0, nb_sim)

state_pt = PTState{Float64}(copy(initial_states))
state_nrpt = PTState{Float64}(copy(initial_states))

algo_pt_complet!(sampler_pt, state_pt, bimodal_energy, bimodal_local_update!, bimodal_obs, 5000, 30000)
algo4_nrpt_complet!(sampler_nrpt, state_nrpt, bimodal_energy, bimodal_local_update!, bimodal_obs, 4000, 30000)

samples_pt = state_pt.history.obs_hist[end]
samples_nrpt = state_nrpt.history.obs_hist[end]

x_grid = range(-10, 10, length=400)
y_true = 0.5 .* pdf.(d1, x_grid) .+ 0.5 .* pdf.(d2, x_grid)

p1 = histogram(samples_pt, normalize=:pdf, bins=80, alpha=0.5, label="PT Echantillons", title="Test PT Bimodal")
plot!(p1, x_grid, y_true, linewidth=3, label="Vraie Mixture")

p2 = histogram(samples_nrpt, normalize=:pdf, bins=80, alpha=0.5, label="NRPT Echantillons", title="Test NRPT Bimodal")
plot!(p2, x_grid, y_true, linewidth=3, label="Vraie Mixture")

plt = plot(p1, p2, layout=(2,1), size=(800, 600))

mkpath("graphiques")
savefig(plt, "graphiques/test_bimodal_pt_nrpt.png")

display(plt)