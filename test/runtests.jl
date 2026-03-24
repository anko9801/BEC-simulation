using Test
using SpinorBEC

@testset "SpinorBEC" begin
    include("test_grid.jl")
    include("test_spin_matrices.jl")
    include("test_propagators.jl")
    include("test_spin_mixing.jl")
    include("test_split_step.jl")
    include("test_observables.jl")
    include("test_simulation.jl")
    include("test_3d.jl")
    include("test_ddi.jl")
    include("test_experiment.jl")
    include("test_physics_level0.jl")
    include("test_physics_level1.jl")
    include("test_physics_level2.jl")
    include("test_physics_level3.jl")
end
