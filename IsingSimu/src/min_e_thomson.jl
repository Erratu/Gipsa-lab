# Tableau des énergies minimales connues pour le problème de Thomson (N de 2 à 32)
const THOMSON_ENERGIES = Float64[
    0.500000000,     # N = 2
    1.732050808,     # N = 3  
    3.674234614,     # N = 4
    6.474691495,     # N = 5
    9.985281374,     # N = 6
    14.452977414,    # N = 7
    19.675287861,    # N = 8
    25.759986531,    # N = 9
    32.716949460,    # N = 10
    40.596450510,    # N = 11
    49.165253058,    # N = 12
    58.853230612,    # N = 13
    69.306363297,    # N = 14
    80.670244114,    # N = 15
    92.911655302,    # N = 16
    106.050404829,   # N = 17
    120.084467447,   # N = 18
    135.089467557,   # N = 19
    150.881568334,   # N = 20
    167.641622399,   # N = 21
    185.287536149,   # N = 22
    203.930190663,   # N = 23
    223.347074052,   # N = 24
    243.812760299,   # N = 25
    265.133326317,   # N = 26
    287.302615033,   # N = 27
    310.491542358,   # N = 28
    334.634439920,   # N = 29
    359.603945904,   # N = 30
    385.530838063,   # N = 31
    412.261274651    # N = 32
]

"""
    get_thomson_energy(N::Int)

Retourne l'énergie minimale connue pour N points (2 ≤ N ≤ 32).
"""
function get__min_thomson_energy(N::Int)
    @boundscheck 2 <= N <= 32 || throw(BoundsError(THOMSON_ENERGIES, N - 1))
    return @inbounds THOMSON_ENERGIES[N - 1]
end