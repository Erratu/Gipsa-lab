using Test
using PTShared

@testset "PTShared.jl Test Suite" begin
    include("../tests/test_gauss.jl")
    include("../tests/test_bimodal.jl")
end
