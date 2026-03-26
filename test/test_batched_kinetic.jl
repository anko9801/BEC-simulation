@testset "Batched Kinetic Step" begin
    @testset "Batched matches per-component 1D" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        psi_ref = copy(psi)

        plans = make_fft_plans(grid.config.n_points)
        dt = 0.01
        kp = prepare_kinetic_phase(grid, dt)
        fft_buf = zeros(ComplexF64, grid.config.n_points)

        apply_kinetic_step!(psi_ref, fft_buf, kp, plans, sys.n_components, 1)

        bk = SpinorBEC._make_batched_kinetic_cache(psi, kp, 1)
        apply_kinetic_step_batched!(psi, bk)

        @test psi ≈ psi_ref rtol = 1e-12
    end

    @testset "Batched matches per-component 2D" begin
        config = GridConfig((32, 32), (10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        psi_ref = copy(psi)

        plans = make_fft_plans(grid.config.n_points)
        dt = 0.01
        kp = prepare_kinetic_phase(grid, dt)
        fft_buf = zeros(ComplexF64, grid.config.n_points)

        apply_kinetic_step!(psi_ref, fft_buf, kp, plans, sys.n_components, 2)

        bk = SpinorBEC._make_batched_kinetic_cache(psi, kp, 2)
        apply_kinetic_step_batched!(psi, bk)

        @test psi ≈ psi_ref rtol = 1e-12
    end

    @testset "Phase update works" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        dt = 0.01
        kp = prepare_kinetic_phase(grid, dt)

        bk = SpinorBEC._make_batched_kinetic_cache(psi, kp, 1)

        new_dt = 0.005
        SpinorBEC._update_batched_kinetic_phase!(bk, grid.k_squared, new_dt)

        kp_new = prepare_kinetic_phase(grid, new_dt)
        @test bk.kinetic_phase_bc[:, 1] ≈ kp_new rtol = 1e-14
    end

    @testset "Norm preserved by batched step" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=100)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
        )
        N0 = total_norm(ws.state.psi, ws.grid)
        for _ in 1:100
            split_step!(ws)
        end
        @test total_norm(ws.state.psi, ws.grid) ≈ N0 rtol = 1e-6
    end

    @testset "Workspace has batched_kinetic field" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=10)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(1.0, 0.0),
            sim_params=sp,
        )
        @test ws.batched_kinetic isa BatchedKineticCache
        @test ws.ddi_padded === nothing
    end
end
