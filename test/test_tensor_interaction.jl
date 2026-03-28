@testset "Tensor interaction" begin

    @testset "make_tensor_interaction_cache returns nothing for c2-only" begin
        ip = InteractionParams(100.0, 10.0, 0.0, [5.0])
        cache = make_tensor_interaction_cache(1, ip)
        @test cache === nothing

        ip2 = InteractionParams(100.0, 10.0, 0.0, Float64[])
        cache2 = make_tensor_interaction_cache(2, ip2)
        @test cache2 === nothing
    end

    @testset "make_tensor_interaction_cache activates for higher channels" begin
        # F=2: c_extra = [c2=5.0, c3=0.0, c4=3.0]
        # get_cn(ip, 2) = c_extra[1] = 5.0
        # get_cn(ip, 4) = c_extra[3] = 3.0
        ip = InteractionParams(100.0, 10.0, 0.0, [5.0, 0.0, 3.0])
        cache = make_tensor_interaction_cache(2, ip)
        @test cache !== nothing
        @test cache.F == 2
        @test cache.D == 5
        @test 4 in cache.active_channels
    end

    @testset "F=1 pair amplitudes match known results" begin
        F = 1
        D = 3
        psi_1d = zeros(ComplexF64, 16, D)
        for i in 1:16
            psi_1d[i, :] = randn(ComplexF64, D)
        end

        # Verify |A_0|^2 + |A_2|^2 = n^2 (completeness)
        for i in 1:16
            sp = [psi_1d[i, c] for c in 1:D]
            n = sum(abs2, sp)

            a0_sq = 0.0
            for M in 0:0
                a = 0.0 + 0.0im
                for m1 in -1:1
                    m2 = M - m1
                    abs(m2) > 1 && continue
                    a += clebsch_gordan(1, m1, 1, m2, 0, M) * sp[1-m1+1] * sp[1-m2+1]
                end
                a0_sq += abs2(a)
            end

            a2_sq = 0.0
            for M in -2:2
                a = 0.0 + 0.0im
                for m1 in -1:1
                    m2 = M - m1
                    abs(m2) > 1 && continue
                    a += clebsch_gordan(1, m1, 1, m2, 2, M) * sp[1-m1+1] * sp[1-m2+1]
                end
                a2_sq += abs2(a)
            end

            @test a0_sq + a2_sq ≈ n^2 rtol = 1e-10
        end
    end

    @testset "Norm conservation (real-time)" begin
        F = 2
        D = 2F + 1
        grid = make_grid(GridConfig((32,), (10.0,)))
        n_pts = grid.config.n_points
        psi = randn(ComplexF64, n_pts[1], D)
        dV = cell_volume(grid)
        norm0 = sum(abs2, psi) * dV

        ip = InteractionParams(0.0, 0.0, 0.0, [5.0, 0.0, 3.0])
        cache = make_tensor_interaction_cache(F, ip)
        @test cache !== nothing

        sm = spin_matrices(F)
        dt = 0.01
        for _ in 1:50
            apply_tensor_interaction_step!(psi, cache, sm, dt, 1)
        end

        norm_final = sum(abs2, psi) * dV
        @test norm_final ≈ norm0 rtol = 1e-8
    end

    @testset "Zero coupling = identity" begin
        F = 2
        ip_zero = InteractionParams(0.0, 0.0)
        cache_zero = make_tensor_interaction_cache(F, ip_zero)
        @test cache_zero === nothing
    end

    @testset "Magnetization conservation (real-time)" begin
        F = 2
        D = 2F + 1
        grid = make_grid(GridConfig((32,), (10.0,)))
        n_pts = grid.config.n_points
        psi = randn(ComplexF64, n_pts[1], D)
        sm = spin_matrices(F)

        mag0 = magnetization(psi, grid, sm.system)

        ip = InteractionParams(0.0, 0.0, 0.0, [5.0, 0.0, 3.0])
        cache = make_tensor_interaction_cache(F, ip)
        @test cache !== nothing

        dt = 0.001
        for _ in 1:30
            apply_tensor_interaction_step!(psi, cache, sm, dt, 1)
        end

        mag_final = magnetization(psi, grid, sm.system)
        # Splitting error: exp(-ih dt) with frozen h has O(dt) magnetization drift
        # because [h(ψ), Fz] ≠ 0 for general states (only continuous evolution conserves exactly).
        # Use atol for near-zero magnetization where rtol is unreliable.
        @test abs(mag_final - mag0) < max(0.01, abs(mag0) * 0.1)
    end

    @testset "Energy consistency" begin
        F = 2
        D = 2F + 1
        grid = make_grid(GridConfig((16,), (10.0,)))
        n_pts = grid.config.n_points
        dV = cell_volume(grid)
        psi = randn(ComplexF64, n_pts[1], D)
        norm = sqrt(sum(abs2, psi) * dV)
        psi ./= norm

        ip = InteractionParams(0.0, 0.0, 0.0, [5.0, 0.0, 3.0])
        cache = make_tensor_interaction_cache(F, ip)
        @test cache !== nothing

        E = SpinorBEC._tensor_interaction_energy(psi, cache, 1, n_pts, dV)
        @test isfinite(E)
    end

    @testset "Full split_step with tensor cache" begin
        F = 2
        D = 2F + 1
        grid = make_grid(GridConfig((16,), (10.0,)))

        interactions = InteractionParams(0.0, 0.0, 0.0, [5.0, 0.0, 3.0])
        atom = AtomSpecies("test-f2", 1e-25, 2, 0.0, 0.0, 0.0, Dict(0 => 1e-9, 2 => 2e-9, 4 => 1.5e-9))
        sp = SimParams(; dt=0.001, n_steps=10)

        ws = make_workspace(;
            grid, atom, interactions,
            sim_params=sp,
        )

        @test ws.tensor_cache !== nothing

        dV = cell_volume(grid)
        norm0 = sum(abs2, ws.state.psi) * dV

        for _ in 1:5
            split_step!(ws)
        end

        norm_final = sum(abs2, ws.state.psi) * dV
        @test norm_final ≈ norm0 rtol = 1e-4
    end

    @testset "basis=:channel skips 6j transform" begin
        F = 2
        ip = InteractionParams(0.0, 0.0, 0.0, [5.0, 0.0, 3.0])

        cache_coupling = make_tensor_interaction_cache(F, ip; basis=:coupling)
        cache_channel = make_tensor_interaction_cache(F, ip; basis=:channel)

        @test cache_coupling !== nothing
        @test cache_channel !== nothing

        @test cache_channel.g_values[findfirst(==(2), cache_channel.active_channels)] ≈ 5.0
        @test cache_channel.g_values[findfirst(==(4), cache_channel.active_channels)] ≈ 3.0

        @test 0 in cache_coupling.active_channels
        @test !(0 in cache_channel.active_channels)
    end

    @testset "basis=:channel norm conservation" begin
        F = 2
        D = 2F + 1
        grid = make_grid(GridConfig((32,), (10.0,)))
        n_pts = grid.config.n_points
        psi = randn(ComplexF64, n_pts[1], D)
        dV = cell_volume(grid)
        norm0 = sum(abs2, psi) * dV

        ip = InteractionParams(0.0, 0.0, 0.0, [5.0, 0.0, 3.0])
        cache = make_tensor_interaction_cache(F, ip; basis=:channel)
        @test cache !== nothing

        sm = spin_matrices(F)
        dt = 0.01
        for _ in 1:50
            apply_tensor_interaction_step!(psi, cache, sm, dt, 1)
        end

        norm_final = sum(abs2, psi) * dV
        @test norm_final ≈ norm0 rtol = 1e-8
    end

    @testset "_make_tensor_cache_from_channels" begin
        F = 2
        g_dict = Dict(0 => 100.0, 2 => 50.0, 4 => 25.0)
        cache = SpinorBEC._make_tensor_cache_from_channels(F, g_dict)
        @test cache !== nothing
        @test cache.F == 2
        @test cache.D == 5
        @test Set(cache.active_channels) == Set([0, 2, 4])
        @test cache.g_values[findfirst(==(0), cache.active_channels)] ≈ 100.0
        @test cache.g_values[findfirst(==(4), cache.active_channels)] ≈ 25.0

        empty_dict = Dict{Int,Float64}(0 => 0.0, 2 => 0.0)
        @test SpinorBEC._make_tensor_cache_from_channels(2, empty_dict) === nothing
    end

    @testset "Dict cache norm conservation" begin
        F = 2
        D = 2F + 1
        grid = make_grid(GridConfig((32,), (10.0,)))
        psi = randn(ComplexF64, grid.config.n_points[1], D)
        dV = cell_volume(grid)
        norm0 = sum(abs2, psi) * dV

        g_dict = Dict(0 => 100.0, 2 => 50.0, 4 => 25.0)
        cache = SpinorBEC._make_tensor_cache_from_channels(F, g_dict)
        sm = spin_matrices(F)

        for _ in 1:50
            apply_tensor_interaction_step!(psi, cache, sm, 0.01, 1)
        end
        @test sum(abs2, psi) * dV ≈ norm0 rtol = 1e-8
    end

    @testset "Workspace with c0+c1+c4 → tensor_cache active" begin
        F = 6
        grid = make_grid(GridConfig((16,), (10.0,)))

        c_extra = zeros(Float64, 5)
        c_extra[3] = 50.0  # c4
        interactions = InteractionParams(4000.0, 20.0, 0.0, c_extra)
        atom = AtomSpecies("test-f6", 1e-25, 6, 0.0, 0.0, 0.0, 0.0)
        sp = SimParams(; dt=0.001, n_steps=5)

        ws = make_workspace(; grid, atom, interactions, sim_params=sp)
        @test ws.tensor_cache !== nothing
        @test ws.interactions.c0 ≈ 0.0
        @test ws.interactions.c1 ≈ 0.0

        dV = cell_volume(grid)
        norm0 = sum(abs2, ws.state.psi) * dV
        for _ in 1:3
            split_step!(ws)
        end
        @test sum(abs2, ws.state.psi) * dV ≈ norm0 rtol = 1e-4
    end

    @testset "Workspace without tensor cache" begin
        grid = make_grid(GridConfig((16,), (10.0,)))
        interactions = InteractionParams(100.0, 10.0)
        sp = SimParams(; dt=0.001, n_steps=5)

        ws = make_workspace(;
            grid, atom=Rb87, interactions,
            sim_params=sp,
        )

        @test ws.tensor_cache === nothing

        for _ in 1:5
            split_step!(ws)
        end
    end
end
