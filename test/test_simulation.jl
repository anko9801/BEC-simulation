using Test
using SpinorBEC

@testset "Simulation" begin
    @testset "Ground state: 87Rb ferromagnetic (c1 < 0)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        interactions = InteractionParams(10.0, -0.5)
        trap = HarmonicTrap(1.0)

        result = find_ground_state(;
            grid, atom=Rb87, interactions, potential=trap,
            dt=0.005, n_steps=5000, initial_state=:ferromagnetic,
        )

        psi = result.workspace.state.psi
        n1 = sum(abs2, psi[:, 1]) * cell_volume(grid)
        n2 = sum(abs2, psi[:, 2]) * cell_volume(grid)
        n3 = sum(abs2, psi[:, 3]) * cell_volume(grid)

        @test n1 > 0.9
        @test n2 < 0.05
        @test n3 < 0.05
    end

    @testset "Ground state: 23Na polar (c1 > 0)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        interactions = InteractionParams(10.0, 0.5)
        trap = HarmonicTrap(1.0)

        result = find_ground_state(;
            grid, atom=Na23, interactions, potential=trap,
            dt=0.005, n_steps=5000, initial_state=:polar,
        )

        psi = result.workspace.state.psi
        n1 = sum(abs2, psi[:, 1]) * cell_volume(grid)
        n2 = sum(abs2, psi[:, 2]) * cell_volume(grid)
        n3 = sum(abs2, psi[:, 3]) * cell_volume(grid)

        @test n2 > 0.9
        @test n1 < 0.05
        @test n3 < 0.05
    end

    @testset "run_simulation! returns result" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        interactions = InteractionParams(1.0, 0.1)
        sp = SimParams(; dt=0.01, n_steps=50, imaginary_time=false, save_every=10)

        ws = make_workspace(;
            grid, atom=Rb87, interactions, sim_params=sp,
        )

        result = run_simulation!(ws)
        @test length(result.times) == 6   # initial + 5 saves
        @test length(result.energies) == 6
        @test all(n -> abs(n - 1.0) < 1e-8, result.norms)
    end
end
