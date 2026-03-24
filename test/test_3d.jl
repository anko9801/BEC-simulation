@testset "3D Support" begin
    @testset "3D grid construction" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)

        @test length(grid.x) == 3
        @test length(grid.k) == 3
        @test size(grid.k_squared) == (16, 16, 16)
        @test grid.k_squared[1, 1, 1] ≈ 0.0 atol = 1e-14

        for d in 1:3
            @test length(grid.x[d]) == 16
            @test length(grid.k[d]) == 16
        end

        @test grid.dx == (10.0 / 16, 10.0 / 16, 10.0 / 16)
        @test cell_volume(grid) ≈ prod(grid.dx)
    end

    @testset "3D harmonic trap" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        trap = HarmonicTrap(1.0, 2.0, 3.0)
        V = evaluate_potential(trap, grid)

        @test size(V) == (16, 16, 16)
        @test V[8, 8, 8] > 0.0
        min_idx = argmin(V)
        @test V[min_idx] < V[1, 1, 1]
    end

    @testset "3D Gaussian init and norm" begin
        config = GridConfig((16, 16, 16), (20.0, 20.0, 20.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)

        @test size(psi) == (16, 16, 16, 3)
        @test total_norm(psi, grid) ≈ 1.0 atol = 1e-10
    end

    @testset "3D free particle norm conservation" begin
        config = GridConfig((16, 16, 16), (20.0, 20.0, 20.0))
        grid = make_grid(config)
        atom = Rb87
        interactions = InteractionParams(0.0, 0.0)
        sp = SimParams(dt=0.001, n_steps=20, imaginary_time=false, normalize_every=0, save_every=20)
        ws = make_workspace(; grid, atom, interactions, sim_params=sp)

        N0 = total_norm(ws.state.psi, grid)
        for _ in 1:20
            split_step!(ws)
        end
        N1 = total_norm(ws.state.psi, grid)

        @test abs(N1 - N0) / N0 < 1e-10
    end

    @testset "3D ground state (spin-1, small grid)" begin
        config = GridConfig((16, 16, 16), (20.0, 20.0, 20.0))
        grid = make_grid(config)
        atom = Rb87
        interactions = compute_interaction_params(atom; N_atoms=100, dims=3)
        trap = HarmonicTrap(1.0, 1.0, 1.0)

        result = find_ground_state(;
            grid, atom, interactions, potential=trap,
            dt=0.001, n_steps=500, tol=1e-8,
            initial_state=:ferromagnetic,
        )

        @test total_norm(result.workspace.state.psi, grid) ≈ 1.0 atol = 1e-8
    end
end
