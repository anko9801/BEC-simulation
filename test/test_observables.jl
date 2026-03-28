using Test
using SpinorBEC
using Random

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

    @testset "Pair amplitude" begin
        grid = make_grid(GridConfig(32, 10.0))
        cg_table = precompute_cg_table(1)

        @testset "singlet channel matches singlet_pair_amplitude for F=1" begin
            sys = SpinSystem(1)
            psi = init_psi(grid, sys; state=:polar)
            A_new = pair_amplitude(psi, 1, 0, 0, 1, cg_table)
            A_old = singlet_pair_amplitude(psi, 1, 1)
            @test A_new ≈ A_old atol = 1e-14
        end

        @testset "singlet channel matches for uniform state" begin
            sys = SpinSystem(1)
            psi = init_psi(grid, sys; state=:uniform)
            A_new = pair_amplitude(psi, 1, 0, 0, 1, cg_table)
            A_old = singlet_pair_amplitude(psi, 1, 1)
            @test A_new ≈ A_old atol = 1e-14
        end

        @testset "sum rule: Σ g_S|A_{SM}|² = c₀n² + c₁|F|²" begin
            grid32 = make_grid(GridConfig(32, 10.0))
            sm = spin_matrices(1)

            psi = zeros(ComplexF64, 32, 3)
            rng = Random.MersenneTwister(42)
            psi .= randn(rng, ComplexF64, 32, 3)
            dV = cell_volume(grid32)
            norm_sq = sum(abs2, psi) * dV
            psi ./= sqrt(norm_sq)

            c0 = 10.0
            c1 = -0.5

            spec = pair_amplitude_spectrum(psi, 1, grid32)

            n = total_density(psi, 1)
            fx, fy, fz = spin_density_vector(psi, sm, 1)

            lhs = c0 * sum(n .^ 2) * dV + c1 * sum(fx .^ 2 .+ fy .^ 2 .+ fz .^ 2) * dV

            g0 = c0 + c1 * (0 * (0 + 1) - 2 * 1 * 2) / 2  # S=0: c0 - 2c1
            g2 = c0 + c1 * (2 * 3 - 2 * 1 * 2) / 2         # S=2: c0 + c1
            rhs = g0 * spec.channel_weights[0] + g2 * spec.channel_weights[2]

            @test lhs ≈ rhs rtol = 1e-10
        end
    end

    @testset "Pair amplitude spectrum" begin
        grid = make_grid(GridConfig(32, 10.0))
        sys = SpinSystem(1)

        @testset "polar state |m=0⟩: S=0 weight = 1/3, S=2 weight = 2/3" begin
            psi = init_psi(grid, sys; state=:polar)
            spec = pair_amplitude_spectrum(psi, 1, grid)
            total = spec.channel_weights[0] + spec.channel_weights[2]
            @test spec.channel_weights[0] / total ≈ 1 / 3 rtol = 1e-10
            @test spec.channel_weights[2] / total ≈ 2 / 3 rtol = 1e-10
        end

        @testset "ferromagnetic state: dominant S=2" begin
            psi = init_psi(grid, sys; state=:ferromagnetic)
            spec = pair_amplitude_spectrum(psi, 1, grid)
            total = spec.channel_weights[0] + spec.channel_weights[2]
            @test spec.channel_weights[2] / total > 0.99
        end
    end
end
