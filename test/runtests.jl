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
    include("test_optics.jl")
    include("test_thomas_fermi.jl")
    include("test_laser_potential.jl")
    include("test_raman.jl")
    include("test_adaptive_dt.jl")
    include("test_unitful.jl")
    include("test_angular_momentum.jl")
end
