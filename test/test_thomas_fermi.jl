using Test
using SpinorBEC

@testset "Thomas-Fermi" begin

    @testset "1D harmonic trap: analytical TF profile" begin
        grid = make_grid(GridConfig(256, 20.0))
        V = evaluate_potential(HarmonicTrap(1.0), grid)
        dV = cell_volume(grid)
        g = 100.0
        N_target = 1.0

        result = thomas_fermi_density(V, g, dV; N_target)
        n_TF = result.density
        μ = result.mu

        norm = sum(n_TF) * dV
        @test norm ≈ N_target rtol = 1e-6

        R_TF = sqrt(2μ)
        for i in eachindex(grid.x[1])
            x = grid.x[1][i]
            if abs(x) < R_TF * 0.8
                expected = max(0.0, (μ - 0.5 * x^2) / g)
                @test n_TF[i] ≈ expected rtol = 1e-6
            end
        end
    end

    @testset "2D harmonic trap" begin
        grid = make_grid(GridConfig((64, 64), (10.0, 10.0)))
        V = evaluate_potential(HarmonicTrap(1.0, 1.0), grid)
        dV = cell_volume(grid)

        result = thomas_fermi_density(V, 50.0, dV; N_target=1.0)
        norm = sum(result.density) * dV
        @test norm ≈ 1.0 rtol = 1e-6

        @test result.density[32, 32] > 0
        @test result.density[1, 1] == 0.0
    end

    @testset "3D harmonic trap" begin
        grid = make_grid(GridConfig((16, 16, 16), (8.0, 8.0, 8.0)))
        V = evaluate_potential(HarmonicTrap(1.0, 1.0, 1.0), grid)
        dV = cell_volume(grid)

        result = thomas_fermi_density(V, 200.0, dV; N_target=1.0)
        norm = sum(result.density) * dV
        @test norm ≈ 1.0 rtol = 1e-4
    end

    @testset "init_psi_thomas_fermi: normalization" begin
        grid = make_grid(GridConfig(256, 20.0))
        sys = SpinSystem(1)
        pot = HarmonicTrap(1.0)
        c0 = 100.0
        N_target = 1000.0

        psi = init_psi_thomas_fermi(grid, sys, pot, c0; N_target)
        dV = cell_volume(grid)
        norm = sum(abs2, psi) * dV
        @test norm ≈ N_target rtol = 1e-6
    end

    @testset "init_psi_thomas_fermi: polar state (m=0 only)" begin
        grid = make_grid(GridConfig(128, 15.0))
        sys = SpinSystem(1)
        pot = HarmonicTrap(1.0)

        psi = init_psi_thomas_fermi(grid, sys, pot, 100.0; N_target=1.0)

        n_pts = grid.config.n_points
        dV = cell_volume(grid)

        n_m1 = sum(abs2, psi[:, 1]) * dV
        n_0 = sum(abs2, psi[:, 2]) * dV
        n_p1 = sum(abs2, psi[:, 3]) * dV

        @test n_m1 < 1e-15
        @test n_p1 < 1e-15
        @test n_0 ≈ 1.0 rtol = 1e-6
    end

    @testset "larger g → flatter profile" begin
        grid = make_grid(GridConfig(256, 20.0))
        V = evaluate_potential(HarmonicTrap(1.0), grid)
        dV = cell_volume(grid)

        r1 = thomas_fermi_density(V, 50.0, dV; N_target=1.0)
        r2 = thomas_fermi_density(V, 200.0, dV; N_target=1.0)

        @test maximum(r1.density) > maximum(r2.density)
        @test r2.mu > r1.mu
    end

end
