@testset "Nematic / Singlet Pair Amplitude" begin
    @testset "F=1 polar state: |A₀₀|² = n²/3" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        A = singlet_pair_amplitude(psi, 1, 1)
        n = total_density(psi, 1)
        # Polar: ψ₀ = f(x), ψ₊₁=ψ₋₁=0
        # A₀₀ = (-1)^{1-0} ψ₀² / √3 = -ψ₀²/√3
        # |A₀₀|² = |ψ₀|⁴/3 = n²/3
        @test sum(abs2, A) ≈ sum(n .^ 2) / 3 rtol = 1e-12
    end

    @testset "F=1 ferromagnetic state: A₀₀ = 0" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:ferromagnetic)
        A = singlet_pair_amplitude(psi, 1, 1)
        # Ferromagnetic: ψ₊₁ = f(x), rest zero → A₀₀ = 0
        @test sum(abs2, A) < 1e-28
    end

    @testset "F=1 uniform state: non-zero A₀₀" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        A = singlet_pair_amplitude(psi, 1, 1)
        @test sum(abs2, A) > 0
    end

    @testset "2D singlet pair amplitude" begin
        config = GridConfig((32, 32), (10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        A = singlet_pair_amplitude(psi, 1, 2)
        @test size(A) == (32, 32)
        @test sum(abs2, A) > 0
    end

    @testset "F=2 polar state" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(2)
        psi = init_psi(grid, sys; state=:polar)
        A = singlet_pair_amplitude(psi, 2, 1)
        # c=3→m=0 is populated, pairs with itself (c_pair = 5-3+1 = 3)
        # sign = (-1)^{2-0} = +1
        # A₀₀ = ψ₀² / √5 (only m=0 contributes)
        n = total_density(psi, 1)
        @test sum(abs2, A) ≈ sum(n .^ 2) / 5 rtol = 1e-12
    end

    @testset "_nematic_energy positive for polar state" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        n_pts = grid.config.n_points
        dV = cell_volume(grid)
        E = SpinorBEC._nematic_energy(psi, 1, 1.0, 1, n_pts, dV)
        @test E > 0.0
    end

    @testset "_nematic_energy zero for ferromagnetic" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:ferromagnetic)
        n_pts = grid.config.n_points
        dV = cell_volume(grid)
        E = SpinorBEC._nematic_energy(psi, 1, 1.0, 1, n_pts, dV)
        @test E < 1e-28
    end

    @testset "c2 in total_energy via get_cn" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=10)

        # Without c2
        ws0 = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
        )
        E0 = total_energy(ws0)

        # With c2 = 50.0 (stored in c_extra[1])
        ws1 = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5, [50.0]),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
        )
        E1 = total_energy(ws1)

        # Polar state has non-zero A₀₀, so E1 should differ from E0
        @test E1 != E0
    end
end
