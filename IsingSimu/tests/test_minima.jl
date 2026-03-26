import GLMakie as GL 
using Printf 
using Random
using IsingSimu

# Calcule une distance invariante par rotation entre deux configurations
function invariant_distance(X1::Matrix{Float64}, X2::Matrix{Float64})
    G1 = X1' * X1
    G2 = X2' * X2
    return norm(G1 - G2) # Norme de Frobenius de la différence
end

# Fonction de clustering
function cluster_configurations(configs::Vector{Matrix{Float64}}, energies::Vector{Float64}; threshold_E=1e-3, threshold_geom=1e-2)
    n_samples = length(configs)
    cluster_ids = zeros(Int, n_samples)
    current_cluster = 1

    # Tri par énergie
    perm = sortperm(energies)
    
    for i in 1:n_samples
        idx = perm[i]
        if cluster_ids[idx] != 0
            continue
        end
        
        # Initialisation d'un nouveau cluster
        cluster_ids[idx] = current_cluster
        
        # Chercher tous les points qui ont convergé au même endroit
        for j in (i+1):n_samples
            idx_neighbor = perm[j]
            if cluster_ids[idx_neighbor] == 0
                # Énergie quasi-identique
                if abs(energies[idx] - energies[idx_neighbor]) < threshold_E
                    # Même géométrie relative
                    if invariant_distance(configs[idx], configs[idx_neighbor]) < threshold_geom
                        cluster_ids[idx_neighbor] = current_cluster
                    end
                end
            end
        end
        current_cluster += 1
    end
    return cluster_ids, current_cluster - 1
end

function analyze_landscape_ipopt(n::Int; n_samples::Int=100)
    @info "Recherche de minima avec Ipopt (n=$n)"
    configs = Vector{Matrix{Float64}}(undef, n_samples)
    energies = zeros(n_samples)
    
    for i in 1:n_samples
        X0 = init_fixed_symmetry(n)
        X_opt, _ = solve_jump_ipopt_fixed(X0)
        configs[i] = X_opt
        energies[i] = thomson_energy(X_opt)
    end
    
    @info "Clustering des minima locaux trouvés"
    # Fonction de clustering basée sur la matrice de Gram 
    cluster_ids, n_clusters = cluster_configurations(configs, energies)
    sizes = [count(x -> x == c, cluster_ids) for c in 1:n_clusters]
    
    @info "Topographie : $n_clusters minima locaux distincts découverts."
    
    # Calcul des énergies moyennes par cluster pour l'affichage
    cluster_energies = Float64[]
    for c in 1:n_clusters
        push!(cluster_energies, mean(energies[cluster_ids .== c]))
    end
    
    # Trier les clusters du meilleur au pire minimum
    sort_idx = sortperm(cluster_energies)
    
    println("\n--- Bilan des Bassins d'Attraction (via Ipopt) ---")
    println(@sprintf("| %-10s | %-15s | %-12s |", "Cluster", "Énergie", "Taille Bassin"))
    for c in sort_idx[1:min(n_clusters, 15)] # Afficher le top 15
        println(@sprintf("| %-10d | %15.6f | %12d |", c, cluster_energies[c], sizes[c]))
    end
    
    return sizes[sort_idx], cluster_energies[sort_idx]
end

# Bassins d'attraction : visualisation
function plot_landscape_analysis(sizes::Vector{Int}, energies::Vector{Float64}, n::Int)
    fig = GL.Figure(size = (1200, 700))
    
    ax = GL.Axis(fig[1, 1], 
        title="Tailles des Bassins d'Attraction pour n=$n (Triés par Énergie)",
        xlabel="Index du Minimum Local (1 = Meilleur trouvé)",
        ylabel="Taille du Bassin (Nombre de convergences)")
    
    GL.barplot!(ax, 1:length(sizes), sizes, 
        color=energies, 
        colormap=:plasma, # Plasma permet de bien distinguer les variations
        strokewidth=1, 
        strokecolor=:black)
    
    GL.Colorbar(fig[1, 2], 
        limits=(minimum(energies), maximum(energies)), 
        colormap=:plasma, 
        label="Énergie de Thomson du Minimum")
    
    screen = GL.Screen()
    GL.display(screen, fig)
    
    return fig, screen
end

function main_landscape_analysis()
    n = 188 
    n_samples = 150
    
    @info "Démarrage de l'analyse du paysage énergétique pour n = $n"
    @info "Nombre d'échantillons aléatoires : $n_samples"
    
    Random.seed!(1234) 
    
    temps_analyse = @elapsed begin
        sizes, energies = analyze_landscape_ipopt(n, n_samples=n_samples)
    end
    
    @printf("\nAnalyse terminée en %.2f secondes.\n", temps_analyse)
    
    fig, screen = plot_landscape_analysis(sizes, energies, n)
    
    save_path = "graphiques/thomson_landscape_n$(n).png"
    GL.save(save_path, fig)
    @info "Graphique sauvegardé sous : $save_path"
    wait(screen) 
end

main_landscape_analysis()