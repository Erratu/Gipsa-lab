using GLMakie
using Graphs
using NetworkLayout
using Colors

function graph_from_Q(Q)
    n = size(Q,1)
    g = SimpleGraph(n)
    for i in 1:n-1, j in i+1:n
        if Q[i,j] == 1
            add_edge!(g, i, j)
        end
    end
    return g
end

function init_ising_view(Q::Matrix{Int}, X::Vector{Int})

    g = graph_from_Q(Q)

    #Layout
    layout = stress(g)
    xs = [p[1] for p in layout]
    ys = [p[2] for p in layout]

    Xobs = Observable(copy(X))

    fig = Figure(resolution = (400,400))
    ax = Axis(fig[1,1])
    hidedecorations!(ax)
    hidespines!(ax)

    # Arêtes
    for e in edges(g)
        i, j = src(e), dst(e)
        lines!(
            ax,
            [xs[i], xs[j]],
            [ys[i], ys[j]],
            color = :gray,
            linewidth = 1
        )
    end

    # xi
    scatter!(
        ax,
        xs,
        ys,
        markersize = 18,
        color = lift(Xobs) do x
            [xi == 1 ? :red : :blue for xi in x]
        end
    )

    display(fig)

    return Xobs
end

function update_ising_view!(Xobs::Observable{Vector{Int}}, X::Vector{Int})
    Xobs[] = copy(X)
    return nothing
end

function init_replicas_grid(Q::Matrix{Int}, Xs::Vector{Vector{Int}})
    n_replicas = length(Xs)
    n_rows = ceil(Int, sqrt(n_replicas))
    n_cols = ceil(Int, n_replicas / n_rows)

    fig = Figure(resolution = (300*n_cols, 300*n_rows))
    replicas = Vector{Observable{Vector{Int}}}(undef, n_replicas)

    for i in 1:n_replicas
        row = ceil(Int, i / n_cols)
        col = i - (row-1)*n_cols
        g = graph_from_Q(Q)

        # Layout
        layout = stress(g)
        xs = [p[1] for p in layout]
        ys = [p[2] for p in layout]

        Xobs = Observable(copy(Xs[i]))
        replicas[i] = Xobs

        ax = Axis(fig[row, col])
        hidedecorations!(ax)
        hidespines!(ax)

        # Arêtes
        for e in edges(g)
            i1, j1 = src(e), dst(e)
            lines!(ax, [xs[i1], xs[j1]], [ys[i1], ys[j1]], color=:gray, linewidth=1)
        end

        # Sommets dynamiques
        scatter!(ax, xs, ys, markersize=15, color=lift(Xobs) do x
            [xi==1 ? :red : :blue for xi in x]
        end)
    end

    display(fig)
    return fig, replicas
end