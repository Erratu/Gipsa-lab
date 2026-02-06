using Karnak
using Graphs

g = barabasi_albert(100, 1)
@drawsvg begin
    background("black")
    sethue("white")
    drawgraph(g, layout=stress, vertexlabels = 1:nv(g))
end