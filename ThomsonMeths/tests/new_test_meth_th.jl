using ThomsonMeths
using PTShared
using Printf 
using LinearAlgebra 
using Random 
import GLMakie as GL 
using Plots
using StatsBase

# Création du dossier pour sauvegarder les graphiques proprement
const OUT_DIR = "thomson_meth_graphiques"
mkpath(OUT_DIR)

function benchmark_q_distributions_and_save(m, n)
    println("\n--- Lancement du Benchmark des distributions q pour m=$m, n=$n ---")
    
    t_uni = @elapsed _, hist_uni = sgd_loo(m, n, strategy=:uniform)
    println("Temps sgd_loo (Uniforme)  : $(round(t_uni, digits=5)) s")
    
    t_boltz = @elapsed _, hist_boltz = sgd_loo(m, n, strategy=:boltzmann)
    println("Temps sgd_loo (Boltzmann) : $(round(t_boltz, digits=5)) s")
    
    t_bp = @elapsed _, hist_bp = sgd_loo(m, n, strategy=:bp)
    println("Temps sgd_loo (BP)       : $(round(t_bp, digits=5)) s")
    
    p = plot(hist_uni, label="Uniforme", linewidth=2)
    plot!(p, hist_boltz, label="Boltzmann", linewidth=2)
    plot!(p, hist_bp, label="Belief Propagation", linewidth=2,
          title="Comparaison des distributions q (n=$n, m=$m)",
          xlabel="Itérations (x10)", ylabel="Énergie", legend=:topright)
          
    filepath = joinpath(OUT_DIR, "1_benchmark_q_distributions.png")
    savefig(p, filepath)
    println("-> Graphique sauvegardé : $filepath")
end

# Visualisation de la limite de diffusion et BLASSO
function plot_diffusion_measure_and_save(n_target=46; n_large=300)
    println("\n--- Lancement de la Diffusion SGD ---")
    # On récupère les amas et les centres
    X_optimal, all_points = sgd_diffusion_measure(n_target, n_large=n_large, max_iter=3000, burn_in=1000)
    
    fig = GL.Figure(size = (800, 800))
    ax = GL.LScene(fig[1, 1], show_axis=false)
    
    # 1. Dessiner la sphère de base (semi-transparente)
    GL.mesh!(ax, GL.Sphere(GL.Point3f(0), 0.95f0), color=(:white, 0.2), transparency=true)
    
    # 2. Afficher le nuage de points de la mesure stationnaire (très transparent)
    GL.scatter!(ax, all_points[1, :], all_points[2, :], all_points[3, :], 
                markersize=3, color=(:blue, 0.05), transparency=true)
    
    # 3. Afficher les pics détectés par BLASSO/K-Means (gros points rouges)
    GL.scatter!(ax, X_optimal[1, :], X_optimal[2, :], X_optimal[3, :], 
                markersize=15, color=:red)
    
    filepath = joinpath(OUT_DIR, "2_diffusion_measure_blasso.png")
    GL.save(filepath, fig)
    println("-> Graphique 3D sauvegardé : $filepath")
    
    # Décommenter la ligne suivante si vous voulez que la fenêtre interactive s'ouvre
    # GL.display(fig) 
end

# Cartographie de l'énergie locale (Heatmap)
function plot_configuration_heatmap_and_save(X)
    println("\n--- Lancement de la Heatmap d'énergie locale ---")
    n = size(X, 2)
    H = matrice_interactions(X)
    
    # Calcul du potentiel local pour chaque point (somme de sa colonne/ligne dans H)
    potentiels_locaux = [sum(H[i, :]) for i in 1:n]
    
    fig = GL.Figure(size = (800, 600))
    ax = GL.LScene(fig[1, 1], show_axis=false)
    GL.mesh!(ax, GL.Sphere(GL.Point3f(0), 0.98f0), color=(:white, 0.5), transparency=true)
    
    # Tracer les points avec une colormap basée sur la répulsion locale
    scatter_plot = GL.scatter!(ax, X[1, :], X[2, :], X[3, :], 
                color=potentiels_locaux, colormap=:inferno, markersize=15)
    
    GL.Colorbar(fig[1, 2], scatter_plot, label="Potentiel de Répulsion Local")
    
    filepath = joinpath(OUT_DIR, "3_heatmap_energie_locale.png")
    GL.save(filepath, fig)
    println("-> Graphique 3D sauvegardé : $filepath")
end

# Parallel Tempering Connection
function run_and_plot_parallel_tempering(m, k_max)
    println("\n--- Lancement du Parallel Tempering ---")
    sampler, state, E_func, upd_func, obs_func = setup_pt_thomson(m, k_max)
    
    # Exécution de l'algorithme (50 burn-in, 200 samples par exemple)
    algo_pt_complet!(sampler, state, E_func, upd_func, obs_func, 50, 200)

    # Graphique 1 : Trajectoire de la Réplique 1 à travers les températures
    p_traj = plot(state.history.trajectoires[1], 
        title="Trajectoire PT (Réplique 1)", 
        xlabel="Scans", ylabel="Indice de Température β", 
        legend=false, color=:purple, linewidth=1.5, yticks=1:k_max)
    
    filepath = joinpath(OUT_DIR, "4_parallel_tempering_traj.png")
    savefig(p_traj, filepath)

    # s=1 correspond à beta=1.0 (la chaîne cible, cherchant le minimum strict pour m points)
    energie_cible = state.history.ene_hist[1]
    
    p = plot(energie_cible, 
        title="Convergence d'Énergie - Chaîne PT (m=$m, β=1.0)", 
        xlabel="Itérations (Échantillonnage)", 
        ylabel="Énergie", 
        linewidth=2, 
        color=:purple,
        legend=false
    )
    
    filepath = joinpath(OUT_DIR, "4_parallel_tempering_convergence.png")
    savefig(p, filepath)
    println("-> Graphique sauvegardé : $filepath")
end

# Comparaison Globale des Méthodes
function comparaison_and_save(m, n)
    println("\n--- Lancement de la Comparaison des méthodes (m=$m, n=$n) ---")
    t_sgd = @elapsed X_final, history = sgd_loo(m, n)
    println("Config n trouvée. Temps = $(round(t_sgd, digits=5)) s")
    
    H = matrice_interactions(X_final)
    
    t_ex = @elapsed idx_ex, E_ex = methode_exhaustive(H, m)
    println("Exhaustive : E = $(round(E_ex, digits=6)) | Temps = $(round(t_ex, digits=5)) s")
    
    t_heu = @elapsed idx_heu, E_heu = methode_heuristique(H, m)
    println("Heuristique: E = $(round(E_heu, digits=6)) | Temps = $(round(t_heu, digits=5)) s")
    
    t_jump = @elapsed idx_jump, E_jump = methode_relaxation_jump(H, m)
    println("Relaxation : E = $(round(E_jump, digits=6)) | Temps = $(round(t_jump, digits=5)) s")
    
    t_gs = @elapsed idx_gs, E_gs = methode_gumbel_softmax(H, m)
    println("Gumbel-Softmax: E = $(round(E_gs, digits=6)) | Temps = $(round(t_gs, digits=5)) s")

    t_diff = @elapsed X_optimal, _ = sgd_diffusion_measure(m, n_large=300, max_iter=3000, burn_in=1000)
    E_diff = thomson_energy_fast(X_optimal)
    println("Diffusion Measure: E = $(round(E_diff, digits=6)) | Temps = $(round(t_diff, digits=5)) s")

    # Création du graphique de synthèse
    p1 = plot(history, 
        title="Convergence (SGD Loo)", 
        xlabel="Itérations (x10)", 
        ylabel="Énergie", 
        linewidth=2, 
        legend=false,
        color=:blue
    )

    x_coords = X_final[1, :]
    y_coords = X_final[2, :]
    z_coords = X_final[3, :]

    H_final = matrice_interactions(X_final)
    idx_gardes, _ = methode_heuristique(H_final, m)
    couleurs = [i in idx_gardes ? :green : :red for i in 1:n]

    p2 = scatter(x_coords, y_coords, z_coords, 
        title="Config finale (n=$n sur m=$m)",
        markersize=5,
        color=couleurs,
        legend=false,
        camera=(45, 30)
    )
    
    p = plot(p1, p2, layout=(1, 2), size=(900, 400))
    filepath = joinpath(OUT_DIR, "5_comparaison_globale.png")
    savefig(p, filepath)
    println("-> Graphique de comparaison sauvegardé : $filepath")
    
    return X_final
end

function plot_scaling_n_and_save(m_cible; max_n_extra=10)
    println("\n--- Benchmark de Scalabilité (m=$m_cible fixe, n variable) ---")
    
    # On fait varier n de m+1 jusqu'à m + max_n_extra
    plages_n = (m_cible + 1):(m_cible + max_n_extra)
    
    # Tableaux pour stocker les résultats
    E_heu = Float64[]; T_heu = Float64[]
    E_gs  = Float64[]; T_gs  = Float64[]
    E_sgd = Float64[]; T_sgd = Float64[]
    
    for n in plages_n
        @printf("Calcul pour n = %d...\n", n)
        
        # 1. Base commune : On génère d'abord une configuration stable de n points
        X_base, _ = sgd_loo(m, n) 
        H_base = matrice_interactions(X_base)
        
        # --- Méthode 1 : Heuristique (Glouton) ---
        t1 = @elapsed _, e1 = methode_heuristique(H_base, m_cible)
        push!(E_heu, e1)
        push!(T_heu, t1)
        
        # --- Méthode 2 : Gumbel-Softmax ---
        t2 = @elapsed _, e2 = methode_gumbel_softmax(H_base, m_cible)
        push!(E_gs, e2)
        push!(T_gs, t2)
    end
    
    # --- Création du Graphique ---
    
    # Graphique Haut : Énergie trouvée
    p_energie = plot(plages_n, E_heu, label="Heuristique", marker=:circle, linewidth=2, color=:limegreen)
    plot!(p_energie, plages_n, E_gs, label="Gumbel-Softmax", marker=:square, linewidth=2, color=:orange)
    plot!(p_energie, plages_n, E_sgd, label="SGD Boltzmann", marker=:diamond, linewidth=2, color=:red,
          title="Énergie Minimale selon la taille du réservoir (m=$m_cible)",
          ylabel="Énergie", legend=:topright, gridalpha=0.4)
          
    # Graphique Bas : Temps d'exécution (Échelle Logarithmique Y recommandée)
    # On ajoute 1e-6 pour éviter les problèmes de log10(0) sur des temps très courts
    p_temps = plot(plages_n, T_heu .+ 1e-6, label="", marker=:circle, linewidth=2, color=:limegreen)
    plot!(p_temps, plages_n, T_gs .+ 1e-6, label="", marker=:square, linewidth=2, color=:orange)
    plot!(p_temps, plages_n, T_sgd .+ 1e-6, label="", marker=:diamond, linewidth=2, color=:red,
          title="Coût de Calcul (Échelle Log)",
          xlabel="Nombre total de points (n)", ylabel="Temps (secondes)",
          yaxis=:log10, gridalpha=0.4)
          
    # Assemblage
    p_final = plot(p_energie, p_temps, layout=(2, 1), size=(800, 700), margin=5Plots.mm)
    
    filepath = joinpath(OUT_DIR, "6_scaling_energie_vs_temps.png")
    savefig(p_final, filepath)
    println("-> Graphique de scalabilité sauvegardé : $filepath")
end

# script d'exécution de tous les tests
function run_all_experiments()
    m_test = 46
    
    # 1. Comparaison des distributions q (Leave-one-out strict: n = m + 1)
    benchmark_q_distributions_and_save(m_test, m_test + 1)
    
    # 2. Diffusion SGD et BLASSO
    # Note: On utilise un n_large plus petit (ex: 150) si 300 est trop lent
    #plot_diffusion_measure_and_save(m_test, n_large=150)
    
    # 3. Comparaison globale classique (n = m + 4 selon l'ancien script)
    #X_fin = comparaison_and_save(m_test, m_test + 4)
    
    # 4. Heatmap d'énergie locale (en utilisant la configuration X_fin)
    #plot_configuration_heatmap_and_save(X_fin)
    
    # 5. Parallel Tempering (k_max = 5 "températures" exploratoires)
    #run_and_plot_parallel_tempering(m_test, 5)

    plot_scaling_n_and_save(m_test)
    
    println("\n=== Toutes les expériences sont terminées ===")
    println("Tous les graphiques ont été générés dans le dossier : ./$OUT_DIR/")
end

# Lancer la batterie de tests
run_all_experiments()