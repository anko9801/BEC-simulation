using Test
using SpinorBEC
using LinearAlgebra

@testset "Spin Mixing" begin
    @testset "Norm conservation (1D)" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        dx = grid.dx[1]

        psi = zeros(ComplexF64, 64, 3)
        x = grid.x[1]
        psi[:, 1] .= 0.5 .* exp.(-x .^ 2 ./ 2)
        psi[:, 2] .= 0.3 .* exp.(-x .^ 2 ./ 2)
        psi[:, 3] .= 0.2 .* exp.(-x .^ 2 ./ 2)

        norm_before = sum(abs2, psi) * dx
        sm = spin_matrices(1)

        apply_spin_mixing_step!(psi, sm, 1.0, 0.1, 1)
        norm_after = sum(abs2, psi) * dx

        @test norm_after ≈ norm_before atol = 1e-12
    end

    @testset "Magnetization conservation (1D)" begin
        config = GridConfig(64, 10.0)
        grid = make_grid(config)
        dx = grid.dx[1]
        sys = SpinSystem(1)

        psi = zeros(ComplexF64, 64, 3)
        x = grid.x[1]
        psi[:, 1] .= 0.6 .* exp.(-x .^ 2 ./ 2)
        psi[:, 2] .= 0.3 .* exp.(-x .^ 2 ./ 2)
        psi[:, 3] .= 0.1 .* exp.(-x .^ 2 ./ 2)

        Mz_before = magnetization(psi, grid, sys)
        sm = spin_matrices(1)

        apply_spin_mixing_step!(psi, sm, 1.0, 0.1, 1)
        Mz_after = magnetization(psi, grid, sys)

        @test Mz_after ≈ Mz_before atol = 1e-12
    end

    @testset "c1=0 is identity" begin
        config = GridConfig(32, 10.0)
        grid = make_grid(config)

        psi = zeros(ComplexF64, 32, 3)
        x = grid.x[1]
        psi[:, 1] .= 0.5 .* exp.(-x .^ 2 ./ 2) .* (1 + 0.1im)
        psi[:, 2] .= 0.3 .* exp.(-x .^ 2 ./ 2)
        psi[:, 3] .= 0.2 .* exp.(-x .^ 2 ./ 2) .* (1 - 0.2im)

        psi_orig = copy(psi)
        sm = spin_matrices(1)

        apply_spin_mixing_step!(psi, sm, 0.0, 0.1, 1)
        @test psi ≈ psi_orig atol = 1e-14
    end
end
