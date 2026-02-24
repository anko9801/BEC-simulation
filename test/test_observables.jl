using Test
using SpinorBEC

@testset "Observables" begin
    @testset "Total density (1D)" begin
        psi = zeros(ComplexF64, 10, 3)
        psi[:, 1] .= 1.0
        psi[:, 2] .= 2.0
        psi[:, 3] .= 3.0

        n = total_density(psi, 1)
        @test n ≈ fill(14.0, 10)
    end

    @testset "Component density" begin
        psi = zeros(ComplexF64, 10, 3)
        psi[:, 2] .= 0.5 + 0.5im

        nc = component_density(psi, 1, 2)
        @test nc ≈ fill(0.5, 10)
    end

    @testset "Magnetization for polarized state" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        sys = SpinSystem(1)

        psi = init_psi(grid, sys; state=:ferromagnetic)
        Mz = magnetization(psi, grid, sys)
        @test Mz ≈ 1.0 atol = 1e-10
    end

    @testset "Magnetization for polar state" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        sys = SpinSystem(1)

        psi = init_psi(grid, sys; state=:polar)
        Mz = magnetization(psi, grid, sys)
        @test abs(Mz) < 1e-14
    end

    @testset "Norm of initialized state" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)

        for state in [:polar, :ferromagnetic, :uniform]
            psi = init_psi(grid, sys; state)
            N = total_norm(psi, grid)
            @test N ≈ 1.0 atol = 1e-12
        end
    end
end
