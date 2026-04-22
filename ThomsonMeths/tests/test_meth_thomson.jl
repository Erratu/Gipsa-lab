using ThomsonMeths
using Printf 
using LinearAlgebra 
using Random 
import GLMakie as GL 
using Plots
using StatsBase

function matrice_interactions(X)
    m = size(X, 2)
    H = zeros(m, m)
    for i in 1:m, j in (i+1):m
        dist = norm(X[:, i] - X[:, j])
        H[i, j] = H[j, i] = 1.0 / dist
    end
    return H
end

function plot_diffusion_measure(n_target=46; n_large=300)
    # On récupère les amas et les centres
    X_optimal, all_points = sgd_diffusion_measure(n_target, n_large=n_large, max_iter=3000, burn_in=1000)
    
    fig = GL.Figure(size = (800, 800))
    ax = GL.LScene(fig[1, 1], show_axis=false)
    
    # 1. Dessiner la sphère de base (semi-transparente)
    GL.mesh!(ax, GL.Sphere(GL.Point3f(0), 0.95f0), color=(:white, 0.2), transparency=true)
    
    # 2. Afficher le nuage de points de la mesure stationnaire (très transparent)
    # all_points contient l'historique complet
    GL.scatter!(ax, all_points[1, :], all_points[2, :], all_points[3, :], 
                markersize=3, color=(:blue, 0.05), transparency=true)
    
    # 3. Afficher les pics détectés par BLASSO/K-Means (gros points rouges)
    GL.scatter!(ax, X_optimal[1, :], X_optimal[2, :], X_optimal[3, :], 
                markersize=15, color=:red)
    
    GL.display(fig)
    return fig
end

function plot_configuration_heatmap(X)
    n = size(X, 2)
    H = matrice_interactions(X)
    
    # Calcul du potentiel local pour chaque point (somme de sa colonne/ligne dans H)
    potentiels_locaux = [sum(H[i, :]) for i in 1:n]
    
    fig = GL.Figure(size = (800, 600))
    ax = GL.LScene(fig[1, 1], show_axis=false)
    GL.mesh!(ax, GL.Sphere(GL.Point3f(0), 0.98f0), color=(:white, 0.5), transparency=true)
    
    # Tracer les points avec une colormap basée sur la répulsion locale
    # Les points jaunes/blancs (haute énergie) sont les candidats à l'exclusion
    scatter_plot = GL.scatter!(ax, X[1, :], X[2, :], X[3, :], 
                color=potentiels_locaux, colormap=:inferno, markersize=15)
    
    GL.Colorbar(fig[1, 2], scatter_plot, label="Potentiel de Répulsion Local")
    
    GL.display(fig)
    return fig
end

function comparaison(m, n)
    t_sgd = @elapsed X, history = sgd_loo(m, n)
    println("Config n trouvée. Temps = $(round(t_sgd, digits=5)) s")
    
    println("\nSélection du meilleur sous-ensemble (m=$m, n=$n)")
    H = matrice_interactions(X)
    
    # Recherche Exhaustive
    t_ex = @elapsed idx_ex, E_ex = methode_exhaustive(H, m)
    println("Exhaustive : E = $(round(E_ex, digits=6)) | Point exclu : $(setdiff(1:n, idx_ex)) | Temps = $(round(t_ex, digits=5)) s")
    
    # Glouton + Swaps
    t_heu = @elapsed idx_heu, E_heu = methode_heuristique(H, m)
    println("Heuristique: E = $(round(E_heu, digits=6)) | Point exclu : $(setdiff(1:n, idx_heu)) | Temps = $(round(t_heu, digits=5)) s")
    
    # Relaxation Continue (JuMP/Ipopt)
    t_jump = @elapsed idx_jump, E_jump = methode_relaxation_jump(H, m)
    println("Relaxation : E = $(round(E_jump, digits=6)) | Point exclu : $(setdiff(1:n, idx_jump)) | Temps = $(round(t_jump, digits=5)) s")
    
    # Belief Propagation
    #t_bp = @elapsed idx_bp, E_bp = methode_belief_propagation(H, m)
    #println("Belief Propagation: E = $(round(E_bp, digits=6)) | Point exclu : $(setdiff(1:n, idx_bp)) | Temps = $(round(t_bp, digits=5)) s")
    
    # Belief Propagation avec post-processing
    #t_bp = @elapsed idx_bp, E_bp = methode_belief_propagation_post_process(H, m)
    #println("Belief Propagation avec post-processing: E = $(round(E_bp, digits=6)) | Point exclu : $(setdiff(1:n, idx_bp)) | Temps = $(round(t_bp, digits=5)) s")
    
    # Gumbel-Softmax
    t_gs = @elapsed idx_gs, E_gs = methode_gumbel_softmax(H, m)
    println("Gumbel-Softmax: E = $(round(E_gs, digits=6)) | Point exclu : $(setdiff(1:n, idx_gs)) | Temps = $(round(t_gs, digits=5)) s")

    #t_diff = @elapsed X_optimal, _ = sgd_diffusion_measure(m)
    #E_diff = evaluer_energie(H, X_optimal)
    #println("Diffusion Measure: E = $(round(E_diff, digits=6)) | Temps = $(round(t_diff, digits=5)) s")
    
    return X, history
end

m = 46
n = m+4
X_final, hist = comparaison(m,n)

p1 = plot(hist, 
    title="Convergence de l'énergie (SGD)", 
    xlabel="Itérations (x500)", 
    ylabel="Énergie de Thomson", 
    linewidth=2, 
    legend=false,
    color=:blue
)

x_coords = X_final[1, :]
y_coords = X_final[2, :]
z_coords = X_final[3, :]

H_final = matrice_interactions(X_final)
idx_gardes, _ = methode_heuristique(H_final, 20)
couleurs = [i in idx_gardes ? :green : :red for i in 1:21]

p2 = scatter(x_coords, y_coords, z_coords, 
    title="Configuration finale (n=$n sur m=$(n+1))",
    markersize=5,
    color=couleurs,
    legend=false,
    camera=(45, 30)
)
p = plot(p1, p2, layout=(1, 2), size=(900, 400))
display(p)
