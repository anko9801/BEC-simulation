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

    @testset "apply_nematic_step! c2=0 identity" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        psi_orig = copy(psi)
        interactions = InteractionParams(10.0, -0.5)  # c2=0
        apply_nematic_step!(psi, interactions, 1, 0.01, 1)
        @test psi ≈ psi_orig atol = 1e-15
    end

    @testset "apply_nematic_step! norm conservation (F=1)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        interactions = InteractionParams(10.0, -0.5, [50.0])  # c2=50
        N0 = total_norm(psi, grid)
        apply_nematic_step!(psi, interactions, 1, 0.01, 1)
        N1 = total_norm(psi, grid)
        @test abs(N1 - N0) / N0 < 1e-12
    end

    @testset "apply_nematic_step! norm conservation (F=2)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(2)
        psi = init_psi(grid, sys; state=:uniform)
        interactions = InteractionParams(10.0, -0.5, [30.0])
        N0 = total_norm(psi, grid)
        apply_nematic_step!(psi, interactions, 2, 0.01, 1)
        N1 = total_norm(psi, grid)
        @test abs(N1 - N0) / N0 < 1e-12
    end

    @testset "apply_nematic_step! ferromagnetic invariance" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:ferromagnetic)
        psi_orig = copy(psi)
        interactions = InteractionParams(10.0, -0.5, [50.0])
        apply_nematic_step!(psi, interactions, 1, 0.01, 1)
        # Ferromagnetic: A₀₀=0, so nematic step is identity
        @test psi ≈ psi_orig atol = 1e-14
    end

    @testset "apply_nematic_step! 2D norm conservation" begin
        config = GridConfig((32, 32), (10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        interactions = InteractionParams(10.0, -0.5, [50.0])
        N0 = total_norm(psi, grid)
        apply_nematic_step!(psi, interactions, 1, 0.01, 2)
        N1 = total_norm(psi, grid)
        @test abs(N1 - N0) / N0 < 1e-12
    end

    @testset "apply_nematic_step! per-pair norm with complex A₀₀ (F=1)" begin
        psi = zeros(ComplexF64, 1, 3)
        psi[1, 1] = 0.5 * cis(0.7)
        psi[1, 2] = 0.3 + 0.2im
        psi[1, 3] = 0.4 * cis(-0.3)
        interactions = InteractionParams(10.0, -0.5, [50.0])

        pair_norm_before = abs2(psi[1, 1]) + abs2(psi[1, 3])

        apply_nematic_step!(psi, interactions, 1, 0.05, 1)

        pair_norm_after = abs2(psi[1, 1]) + abs2(psi[1, 3])

        # (m=+1, m=-1) pair norm is exactly conserved by Bogoliubov transform
        @test abs(pair_norm_after - pair_norm_before) < 1e-14
    end

    @testset "apply_nematic_step! per-pair norm with complex A₀₀ (F=2)" begin
        psi = zeros(ComplexF64, 1, 5)
        psi[1, 1] = 0.4 * cis(0.5)    # m=+2
        psi[1, 2] = 0.3 * cis(-0.8)   # m=+1
        psi[1, 3] = 0.2 + 0.1im       # m=0
        psi[1, 4] = 0.35 * cis(1.2)   # m=-1
        psi[1, 5] = 0.25 * cis(-0.3)  # m=-2
        interactions = InteractionParams(10.0, -0.5, [30.0])

        pair1_before = abs2(psi[1, 1]) + abs2(psi[1, 5])  # (m=2, m=-2)
        pair2_before = abs2(psi[1, 2]) + abs2(psi[1, 4])  # (m=1, m=-1)

        apply_nematic_step!(psi, interactions, 2, 0.05, 1)

        pair1_after = abs2(psi[1, 1]) + abs2(psi[1, 5])
        pair2_after = abs2(psi[1, 2]) + abs2(psi[1, 4])

        @test abs(pair1_after - pair1_before) < 1e-14
        @test abs(pair2_after - pair2_before) < 1e-14
    end

    @testset "apply_nematic_step! ITP norm decrease" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        interactions = InteractionParams(10.0, -0.5, [50.0])
        N0 = total_norm(psi, grid)
        apply_nematic_step!(psi, interactions, 1, 0.001, 1; imaginary_time=true)
        N1 = total_norm(psi, grid)
        # ITP step doesn't conserve norm (damping), but shouldn't explode
        @test N1 < 2 * N0
        @test N1 > 0.0
    end

    @testset "apply_nematic_step! ITP symmetry with complex A₀₀ (F=1)" begin
        # ITP formula: ψ_m' = ch*ψ_m - ph*sh*ψ*_{-m}, ψ_{-m}' = ch*ψ_{-m} - ph*sh*ψ*_m
        # Both have minus sign → symmetric under m ↔ -m exchange
        psi = zeros(ComplexF64, 1, 3)
        psi[1, 1] = 0.5 * cis(0.7)    # m=+1
        psi[1, 2] = 0.3 + 0.2im       # m=0
        psi[1, 3] = 0.4 * cis(-0.3)   # m=-1
        interactions = InteractionParams(10.0, -0.5, [50.0])

        # Swap m=+1 ↔ m=-1
        psi_swapped = zeros(ComplexF64, 1, 3)
        psi_swapped[1, 1] = psi[1, 3]
        psi_swapped[1, 2] = psi[1, 2]
        psi_swapped[1, 3] = psi[1, 1]

        apply_nematic_step!(psi, interactions, 1, 0.05, 1; imaginary_time=true)
        apply_nematic_step!(psi_swapped, interactions, 1, 0.05, 1; imaginary_time=true)

        # Symmetric ITP: swap(input) → swap(output)
        @test psi[1, 1] ≈ psi_swapped[1, 3] atol = 1e-14
        @test psi[1, 3] ≈ psi_swapped[1, 1] atol = 1e-14
    end

    @testset "apply_nematic_step! integrated in split_step" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        interactions = InteractionParams(10.0, -0.5, [50.0])
        sp = SimParams(; dt=0.001, n_steps=10)
        ws = make_workspace(;
            grid, atom=Rb87, interactions,
            potential=HarmonicTrap(1.0), sim_params=sp,
        )
        N0 = total_norm(ws.state.psi, ws.grid)
        for _ in 1:10
            split_step!(ws)
        end
        N1 = total_norm(ws.state.psi, ws.grid)
        @test abs(N1 - N0) / N0 < 1e-6
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
