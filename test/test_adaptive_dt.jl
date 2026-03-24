using Test
using SpinorBEC

@testset "Adaptive dt" begin

    @testset "adaptive converges to same ground state" begin
        grid = make_grid(GridConfig(64, 15.0))
        atom = Rb87
        interactions = compute_interaction_params(atom)
        potential = HarmonicTrap(1.0)

        result_fixed = find_ground_state(;
            grid, atom, interactions, potential,
            dt=0.001, n_steps=5000, tol=1e-8,
        )

        result_adaptive = find_ground_state(;
            grid, atom, interactions, potential,
            dt=0.001, n_steps=5000, tol=1e-8,
            adaptive_dt=true, dt_max=0.01,
        )

        @test abs(result_fixed.energy - result_adaptive.energy) / abs(result_fixed.energy) < 0.01
    end

    @testset "adaptive ground state converges" begin
        grid = make_grid(GridConfig(64, 15.0))
        atom = Rb87
        interactions = compute_interaction_params(atom)
        potential = HarmonicTrap(1.0)

        result = find_ground_state(;
            grid, atom, interactions, potential,
            dt=0.0005, n_steps=10000, tol=1e-8,
            adaptive_dt=true, dt_max=0.01,
        )

        @test result.energy < 0 || result.energy < 1.0
        norm = total_norm(result.workspace.state.psi, grid)
        @test norm ≈ 1.0 rtol = 1e-6
    end

end
