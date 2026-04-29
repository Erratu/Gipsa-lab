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

    X = random_configuration(n)
    
    t_uni = @elapsed _, hist_uni = sgd_loo(X, m, n, strategy=:uniform)
    println("Temps sgd_loo (Uniforme)  : $(round(t_uni, digits=5)) s")
    
    t_boltz = @elapsed _, hist_boltz = sgd_loo(X, m, n, strategy=:boltzmann)
    println("Temps sgd_loo (Boltzmann) : $(round(t_boltz, digits=5)) s")
    
    t_bp = @elapsed _, hist_bp = sgd_loo(X, m, n, strategy=:bp)
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
    t_sgd = @elapsed X_final, history = sgd_loo(X, m, n)
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
        X_base, _ = sgd_loo(X, m, n)
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

function test_sampling_diversity_and_video(m, n, strat=:boltzmann)
    println("\n--- Test de diversité d'échantillonnage ($strat) ---")
    
    X = random_configuration(n)
    video_file = joinpath(OUT_DIR, "sgd_evolution_$strat.mp4")
    
    # On lance sgd_loo avec l'enregistrement vidéo activé
    t = @elapsed X_final, history, counts = sgd_loo(X, m, n, strategy=strat, record_video=true, video_path=video_file)
    println("Temps d'exécution : $(round(t, digits=2)) s")
    
    # Création du graphique des sélections
    p = bar(1:n, counts, 
        title="Nombre de sélections par point ($strat)\n(m=$m, n=$n, 2000 iters)",
        xlabel="Indice du point (1 à $n)",
        ylabel="Nombre de fois sélectionné",
        legend=false,
        color=:teal,
        bar_width=0.8
    )
    
    # On ajoute une ligne horizontale pour la moyenne théorique
    # Si on tire m points parmi n à chaque itération uniforme, la moyenne est (m/n) * max_iter
    moyenne_theo = (m / n) * 2000
    hline!(p, [moyenne_theo], color=:red, linestyle=:dash, label="Moyenne (Uniforme)")
    
    filepath_bar = joinpath(OUT_DIR, "selection_counts_$strat.png")
    savefig(p, filepath_bar)
    println("-> Graphique des sélections sauvegardé : $filepath_bar")
end

function plot_performances_n_m_and_save(plages_m::AbstractVector{Int}, max_k_extra::Int=8)
    println("\n--- Benchmark Global 2D : Performances des Sélections (m et n variables) ---")
    
    # On va utiliser "k" comme variable d'axe X (k = n - m, le nombre de points à rejeter)
    plages_k = 1:max_k_extra
    
    # On prépare un grand graphique contenant des sous-graphiques (un par valeur de m)
    plots_energie = []
    plots_temps = []
    
    for m in plages_m
        @printf("-> Calculs pour la cible m = %d...\n", m)
        
        E_heu = Float64[]; T_heu = Float64[]
        E_gs  = Float64[]; T_gs  = Float64[]
        E_bp  = Float64[]; T_bp  = Float64[]
        E_rnd = Float64[]; # Référence aléatoire (sans optimisation)
        
        for k in plages_k
            n = m + k
            
            X = random_configuration(n) # Configuration de base pour ce n
            # 1. Base commune : Configuration partiellement relaxée de n points
            # On prend un nombre d'itérations réduit pour ne pas bloquer le benchmark global
            X_base, _ = sgd_loo(X, m, n, max_iter=200, strategy=:uniform) 
            H_base = matrice_interactions(X_base)
            
            # --- Baseline : Tirage Aléatoire (Pire cas) ---
            idx_rnd = randperm(n)[1:m]
            push!(E_rnd, evaluer_energie(H_base, idx_rnd))

            # --- Méthode 1 : Heuristique (Glouton) ---
            t1 = @elapsed _, e1 = methode_heuristique(H_base, m)
            push!(E_heu, e1)
            push!(T_heu, t1)
            
            # --- Méthode 2 : Gumbel-Softmax ---
            # On baisse n_starts pour le benchmark pour gagner du temps
            t2 = @elapsed _, e2 = methode_gumbel_softmax(H_base, m, n_starts=3)
            push!(E_gs, e2)
            push!(T_gs, t2)

            # --- Méthode 3 : Belief Propagation ---
            t3 = @elapsed _, e3 = methode_belief_propagation(H_base, m, max_iter=100)
            push!(E_bp, e3)
            push!(T_bp, t3)
        end
        
        # --- Création des sous-graphiques pour ce m ---
        
        # 1. Graphique d'Énergie (On trace l'écart par rapport au tirage aléatoire pour y voir clair)
        p_e = plot(plages_k, E_rnd .- E_heu, label="Heuristique", marker=:circle, linewidth=2, color=:limegreen)
        plot!(p_e, plages_k, E_rnd .- E_gs, label="Gumbel-Softmax", marker=:square, linewidth=2, color=:orange)
        plot!(p_e, plages_k, E_rnd .- E_bp, label="Belief Prop.", marker=:diamond, linewidth=2, color=:purple,
              title="Gain d'Énergie (m=$m)", ylabel="Gain vs Aléatoire", xlabel="Points rejetés (k = n - m)", 
              legend=(m == plages_m[1] ? :topleft : false), gridalpha=0.4)
        push!(plots_energie, p_e)
              
        # 2. Graphique de Temps
        p_t = plot(plages_k, T_heu .+ 1e-6, label="Heuristique", marker=:circle, linewidth=2, color=:limegreen)
        plot!(p_t, plages_k, T_gs .+ 1e-6, label="Gumbel-Softmax", marker=:square, linewidth=2, color=:orange)
        plot!(p_t, plages_k, T_bp .+ 1e-6, label="Belief Prop.", marker=:diamond, linewidth=2, color=:purple,
              title="Temps d'exécution (m=$m)", ylabel="Temps (s) [Log]", xlabel="Points rejetés (k)", 
              yaxis=:log10, legend=false, gridalpha=0.4)
        push!(plots_temps, p_t)
    end
    
    # --- Assemblage Final ---
    # On empile les graphes d'énergie en haut, et de temps en bas
    p_final_energie = plot(plots_energie..., layout=(1, length(plages_m)), size=(400 * length(plages_m), 350))
    p_final_temps = plot(plots_temps..., layout=(1, length(plages_m)), size=(400 * length(plages_m), 350))
    
    p_global = plot(p_final_energie, p_final_temps, layout=(2, 1), size=(400 * length(plages_m), 750), margin=5Plots.mm)
    
    filepath = joinpath(OUT_DIR, "6_performances_n_m.png")
    savefig(p_global, filepath)
    println("-> Graphique 2D de performances sauvegardé : $filepath")
end

# script d'exécution de tous les tests
function run_all_experiments()
    m_test = 46
    
    # 1. Comparaison des distributions q (Leave-one-out strict: n = m + 1)
    #benchmark_q_distributions_and_save(m_test, m_test + 1)
    
    # 2. Diffusion SGD et BLASSO
    # Note: On utilise un n_large plus petit (ex: 150) si 300 est trop lent
    #plot_diffusion_measure_and_save(m_test, n_large=150)
    
    # 3. Comparaison globale classique (n = m + 4 selon l'ancien script)
    #X_fin = comparaison_and_save(m_test, m_test + 4)
    
    # 4. Heatmap d'énergie locale (en utilisant la configuration X_fin)
    #plot_configuration_heatmap_and_save(X_fin)
    
    # 5. Parallel Tempering (k_max = 5 "températures" exploratoires)
    #run_and_plot_parallel_tempering(m_test, 5)

    #plot_scaling_n_and_save(m_test)

    plot_performances_n_m_and_save([20, 46, 80], 8)

    #test_sampling_diversity_and_video(46, 50, :boltzmann)
    #test_sampling_diversity_and_video(46, 50, :bp)
    #test_sampling_diversity_and_video(46, 50, :uniform)
    
    println("\n=== Toutes les expériences sont terminées ===")
    println("Tous les graphiques ont été générés dans le dossier : ./$OUT_DIR/")
end

# Lancer la batterie de tests
run_all_experiments()