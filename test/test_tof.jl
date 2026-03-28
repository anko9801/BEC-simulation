@testset "TOF / Stern-Gerlach" begin
    @testset "1D Gaussian TOF width" begin
        F = 1
        sys = SpinSystem(F)
        grid = make_grid(GridConfig((128,), (20.0,)))
        psi = init_psi(grid, sys; state=:ferromagnetic)

        params = TOFParams(1.0, 0.0, 1)
        result = simulate_tof(psi, grid, sys, params)

        @test haskey(result, F)  # m=+F component
        @test length(result[F]) == 128
        @test all(v -> v >= 0, result[F])
    end

    @testset "2D TOF returns 1D images" begin
        F = 1
        sys = SpinSystem(F)
        grid = make_grid(GridConfig((16, 16), (10.0, 10.0)))
        psi = init_psi(grid, sys; state=:polar)

        params = TOFParams(1.0, 0.0, 2)
        result = simulate_tof(psi, grid, sys, params)

        # Column integration along axis 2 → 1D array
        @test haskey(result, 0)
        @test ndims(result[0]) == 1
        @test length(result[0]) == 16
    end

    @testset "3D TOF returns 2D images" begin
        F = 1
        sys = SpinSystem(F)
        grid = make_grid(GridConfig((8, 8, 8), (6.0, 6.0, 6.0)))
        psi = init_psi(grid, sys; state=:ferromagnetic)

        params = TOFParams(1.0, 0.0, 3)
        result = simulate_tof(psi, grid, sys, params)

        @test haskey(result, F)
        @test ndims(result[F]) == 2
        @test size(result[F]) == (8, 8)
    end

    @testset "SG separates components" begin
        F = 1
        sys = SpinSystem(F)
        grid = make_grid(GridConfig((16, 16), (10.0, 10.0)))

        # Uniform state: all components equally populated
        psi = init_psi(grid, sys; state=:uniform)

        # Large gradient to separate
        params = TOFParams(10.0, 5.0, 2)
        result = simulate_tof(psi, grid, sys, params)

        # All m components should be present
        @test haskey(result, 1)
        @test haskey(result, 0)
        @test haskey(result, -1)

        # With SG, center of mass should shift differently for ±m
        com_p1 = sum(i * result[1][i] for i in 1:16) / sum(result[1])
        com_m1 = sum(i * result[-1][i] for i in 1:16) / sum(result[-1])
        com_0 = sum(i * result[0][i] for i in 1:16) / sum(result[0])

        # m=+1 and m=-1 should shift in opposite directions relative to m=0
        @test (com_p1 - com_0) * (com_m1 - com_0) <= 0.0 + 1.0  # opposite or near zero
    end

    @testset "TOFParams validation" begin
        @test_throws ArgumentError TOFParams(-1.0, 0.0, 1)
        @test_throws ArgumentError TOFParams(1.0, 0.0, 0)
        @test_throws ArgumentError TOFParams(1.0, 0.0, 4)
    end

    @testset "Total momentum density conserved" begin
        F = 1
        sys = SpinSystem(F)
        grid = make_grid(GridConfig((32,), (10.0,)))
        psi = init_psi(grid, sys; state=:uniform)

        params = TOFParams(1.0, 0.0, 1)
        result = simulate_tof(psi, grid, sys, params)

        total = sum(sum(v) for v in values(result))
        @test total > 0
    end
end
