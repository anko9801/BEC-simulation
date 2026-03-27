@testset "DDI" begin
    @testset "Q tensor symmetry and tracelessness" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        atom = Eu151
        ddi = make_ddi_params(grid, atom)

        @test ddi.Q_xy ≈ ddi.Q_xy
        @test ddi.Q_xz ≈ ddi.Q_xz
        @test ddi.Q_yz ≈ ddi.Q_yz

        trace = ddi.Q_xx .+ ddi.Q_yy .+ ddi.Q_zz
        for I in CartesianIndices(size(trace))
            if grid.k_squared[I] > 0.0
                @test abs(trace[I]) < 1e-12
            end
        end
    end

    @testset "C_dd value for Eu151" begin
        C_dd = compute_c_dd(Eu151)
        @test C_dd > 0.0
        @test C_dd ≈ SpinorBEC.Units.MU_0 * Eu151.mu_mag^2 rtol = 1e-10
    end

    @testset "a_dd and epsilon_dd for Eu151" begin
        a_dd = compute_a_dd(Eu151)
        @test a_dd > 0.0

        eps_dd = a_dd / Eu151.a0
        @test 0.4 < eps_dd < 0.7
    end

    @testset "C_dd = 0 for non-dipolar atoms" begin
        @test compute_c_dd(Rb87) == 0.0
        @test compute_a_dd(Rb87) == 0.0
    end

    @testset "DDI norm conservation (1D)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        psi = init_psi(grid, sys; state=:uniform)
        ddi = make_ddi_params(grid, Eu151)
        bufs = make_ddi_buffers(grid.config.n_points)

        N0 = total_norm(psi, grid)
        apply_ddi_step!(psi, sm, ddi, bufs, 0.001, 1; imaginary_time=false)
        N1 = total_norm(psi, grid)

        @test abs(N1 - N0) / N0 < 1e-10
    end

    @testset "DDI magnetization conservation (1D)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        psi = init_psi(grid, sys; state=:ferromagnetic)
        ddi = make_ddi_params(grid, Eu151)
        bufs = make_ddi_buffers(grid.config.n_points)

        M0 = magnetization(psi, grid, sys)
        apply_ddi_step!(psi, sm, ddi, bufs, 0.001, 1; imaginary_time=false)
        M1 = magnetization(psi, grid, sys)

        @test abs(M1 - M0) < 1e-10
    end

    @testset "C_dd=0 identity" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        psi = init_psi(grid, sys; state=:uniform)
        psi_orig = copy(psi)

        noddi_atom = AtomSpecies("test", Rb87.mass, 1, Rb87.a0, Rb87.a2, 0.0)
        ddi = make_ddi_params(grid, noddi_atom)
        bufs = make_ddi_buffers(grid.config.n_points)

        apply_ddi_step!(psi, sm, ddi, bufs, 0.01, 1; imaginary_time=false)

        @test psi ≈ psi_orig atol = 1e-12
    end

    @testset "DDI norm conservation (3D)" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        psi = init_psi(grid, sys; state=:uniform)
        ddi = make_ddi_params(grid, Eu151)
        bufs = make_ddi_buffers(grid.config.n_points)

        N0 = total_norm(psi, grid)
        apply_ddi_step!(psi, sm, ddi, bufs, 0.001, 3; imaginary_time=false)
        N1 = total_norm(psi, grid)

        @test abs(N1 - N0) / N0 < 1e-10
    end

    @testset "Secular DDI Q tensor" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        ddi_sec = make_ddi_params(grid, Eu151; secular=true)
        ddi_full = make_ddi_params(grid, Eu151; secular=false)

        # Off-diagonal components vanish in secular approximation
        @test maximum(abs, ddi_sec.Q_xy) == 0.0
        @test maximum(abs, ddi_sec.Q_xz) == 0.0
        @test maximum(abs, ddi_sec.Q_yz) == 0.0

        # Q_zz is unchanged
        @test ddi_sec.Q_zz ≈ ddi_full.Q_zz

        # Q_xx = Q_yy = -Q_zz/2
        @test ddi_sec.Q_xx ≈ ddi_sec.Q_yy
        @test ddi_sec.Q_xx ≈ -ddi_sec.Q_zz ./ 2.0

        # Still traceless
        trace = ddi_sec.Q_xx .+ ddi_sec.Q_yy .+ ddi_sec.Q_zz
        for I in CartesianIndices(size(trace))
            if grid.k_squared[I] > 0.0
                @test abs(trace[I]) < 1e-12
            end
        end
    end

    @testset "Secular DDI norm conservation (3D)" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        psi = init_psi(grid, sys; state=:uniform)
        ddi = make_ddi_params(grid, Eu151; secular=true)
        bufs = make_ddi_buffers(grid.config.n_points)

        N0 = total_norm(psi, grid)
        apply_ddi_step!(psi, sm, ddi, bufs, 0.001, 3; imaginary_time=false)
        N1 = total_norm(psi, grid)

        @test abs(N1 - N0) / N0 < 1e-10
    end

    @testset "Secular DDI: z-polarized state same as full" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        sm = spin_matrices(1)

        psi = init_psi(grid, sys; state=:ferromagnetic)

        ddi_full = make_ddi_params(grid, Eu151; secular=false)
        bufs_full = make_ddi_buffers(grid.config.n_points)
        psi_full = copy(psi)
        apply_ddi_step!(psi_full, sm, ddi_full, bufs_full, 0.001, 3; imaginary_time=false)

        ddi_sec = make_ddi_params(grid, Eu151; secular=true)
        bufs_sec = make_ddi_buffers(grid.config.n_points)
        psi_sec = copy(psi)
        apply_ddi_step!(psi_sec, sm, ddi_sec, bufs_sec, 0.001, 3; imaginary_time=false)

        @test psi_sec ≈ psi_full atol = 1e-12
    end

    @testset "Secular DDI via make_workspace" begin
        config = GridConfig((8, 8, 8), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        atom = Eu151
        interactions = InteractionParams(100.0, 0.0)
        sp = SimParams(dt=0.001, n_steps=1, imaginary_time=false, normalize_every=0, save_every=1)

        ws = make_workspace(; grid, atom, interactions, sim_params=sp, enable_ddi=true, secular_ddi=true)
        @test ws.ddi !== nothing
        @test maximum(abs, ws.ddi.Q_xy) == 0.0
    end

    @testset "DDI integrated into split-step (3D)" begin
        config = GridConfig((16, 16, 16), (10.0, 10.0, 10.0))
        grid = make_grid(config)
        atom = Eu151
        interactions = InteractionParams(compute_c0(atom; N_atoms=100, dims=3), 0.0)
        trap = HarmonicTrap(1.0, 1.0, 1.0)
        sp = SimParams(dt=0.001, n_steps=10, imaginary_time=false, normalize_every=0, save_every=10)

        ws = make_workspace(; grid, atom, interactions, potential=trap, sim_params=sp, enable_ddi=true)

        @test ws.ddi !== nothing
        @test ws.ddi_bufs !== nothing

        N0 = total_norm(ws.state.psi, ws.grid)
        for _ in 1:10
            split_step!(ws)
        end
        N1 = total_norm(ws.state.psi, ws.grid)

        @test abs(N1 - N0) / N0 < 1e-6
    end
end
