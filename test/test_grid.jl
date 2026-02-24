using Test
using SpinorBEC

@testset "Grid" begin
    @testset "1D grid construction" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)

        @test length(grid.x[1]) == 128
        @test grid.dx[1] ≈ 20.0 / 128
        @test length(grid.k[1]) == 128
        @test grid.dk[1] ≈ 2π / 20.0
        @test size(grid.k_squared) == (128,)
        @test grid.k_squared[1] ≈ 0.0 atol = 1e-14
    end

    @testset "2D grid construction" begin
        config = GridConfig((64, 64), (10.0, 10.0))
        grid = make_grid(config)

        @test length(grid.x[1]) == 64
        @test length(grid.x[2]) == 64
        @test size(grid.k_squared) == (64, 64)
        @test grid.k_squared[1, 1] ≈ 0.0 atol = 1e-14
    end

    @testset "Grid validation" begin
        @test_throws ArgumentError GridConfig(0, 10.0)
        @test_throws ArgumentError GridConfig(127, 10.0)
        @test_throws ArgumentError GridConfig(128, -1.0)
    end

    @testset "FFT plans" begin
        plans = make_fft_plans((128,))
        @test plans.forward !== nothing
        @test plans.inverse !== nothing
    end

    @testset "Cell volume" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        @test cell_volume(grid) ≈ 20.0 / 128

        config2 = GridConfig((64, 32), (10.0, 8.0))
        grid2 = make_grid(config2)
        @test cell_volume(grid2) ≈ (10.0 / 64) * (8.0 / 32)
    end
end
