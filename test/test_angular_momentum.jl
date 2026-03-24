using Test
using SpinorBEC
using LinearAlgebra

@testset "Angular Momentum & Current" begin

    @testset "probability_current: stationary state has zero current" begin
        grid = make_grid(GridConfig((32, 32), (10.0, 10.0)))
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        plans = make_fft_plans(grid.config.n_points)

        jx, jy = probability_current(psi, grid, plans)

        @test maximum(abs, jx) < 1e-12
        @test maximum(abs, jy) < 1e-12
    end

    @testset "probability_current: plane wave has uniform current" begin
        grid = make_grid(GridConfig((64,), (20.0,)))
        sys = SpinSystem(1)
        plans = make_fft_plans(grid.config.n_points)

        k0 = 2π / grid.config.box_size[1] * 3  # 3rd harmonic
        psi = zeros(ComplexF64, 64, 3)
        for i in 1:64
            psi[i, 2] = exp(im * k0 * grid.x[1][i])
        end
        dV = cell_volume(grid)
        psi ./= sqrt(sum(abs2, psi) * dV)

        (jx,) = probability_current(psi, grid, plans)

        n = total_density(psi, 1)
        v_expected = k0  # v = hbar*k/m = k in dim-less
        @test all(abs.(jx .- n .* v_expected) .< 1e-8)
    end

    @testset "orbital_angular_momentum: L_z = 0 for real Gaussian" begin
        grid = make_grid(GridConfig((32, 32), (10.0, 10.0)))
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        plans = make_fft_plans(grid.config.n_points)

        Lz = orbital_angular_momentum(psi, grid, plans)
        @test abs(Lz) < 1e-12
    end

    @testset "orbital_angular_momentum: vortex with winding 1" begin
        N = 64
        L = 20.0
        grid = make_grid(GridConfig((N, N), (L, L)))
        plans = make_fft_plans(grid.config.n_points)
        dV = cell_volume(grid)

        # Single-component vortex: psi = f(r) * exp(i*phi)
        psi = zeros(ComplexF64, N, N, 1)
        sigma = L / 8
        for j in 1:N, i in 1:N
            x, y = grid.x[1][i], grid.x[2][j]
            r = sqrt(x^2 + y^2)
            phi = atan(y, x)
            # r * exp(i*phi) * Gaussian with smooth core
            core = r / (r + 0.5)
            psi[i, j, 1] = core * exp(-(r / sigma)^2) * exp(im * phi)
        end
        psi ./= sqrt(sum(abs2, psi) * dV)

        Lz = orbital_angular_momentum(psi, grid, plans)
        # For winding=1, Lz/N should be close to 1 (= hbar per particle)
        @test abs(Lz - 1.0) < 0.05
    end

    @testset "orbital_angular_momentum: winding 2" begin
        N = 64
        L = 20.0
        grid = make_grid(GridConfig((N, N), (L, L)))
        plans = make_fft_plans(grid.config.n_points)
        dV = cell_volume(grid)

        psi = zeros(ComplexF64, N, N, 1)
        sigma = L / 8
        for j in 1:N, i in 1:N
            x, y = grid.x[1][i], grid.x[2][j]
            r = sqrt(x^2 + y^2)
            phi = atan(y, x)
            core = r^2 / (r^2 + 1.0)
            psi[i, j, 1] = core * exp(-(r / sigma)^2) * exp(2im * phi)
        end
        psi ./= sqrt(sum(abs2, psi) * dV)

        Lz = orbital_angular_momentum(psi, grid, plans)
        @test abs(Lz - 2.0) < 0.1
    end

    @testset "orbital_angular_momentum: returns 0.0 for 1D" begin
        grid = make_grid(GridConfig(64, 10.0))
        sys = SpinSystem(1)
        psi = init_psi(grid, sys)
        plans = make_fft_plans(grid.config.n_points)

        @test orbital_angular_momentum(psi, grid, plans) == 0.0
    end

    @testset "J_z = L_z + S_z conservation (free spinor)" begin
        N = 32
        L = 10.0
        grid = make_grid(GridConfig((N, N), (L, L)))
        plans = make_fft_plans(grid.config.n_points)
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        dV = cell_volume(grid)

        # Two-component state: m=+1 with winding 0, m=0 with winding 1
        psi = zeros(ComplexF64, N, N, 3)
        sigma = L / 8
        for j in 1:N, i in 1:N
            x, y = grid.x[1][i], grid.x[2][j]
            r = sqrt(x^2 + y^2)
            phi = atan(y, x)
            env = exp(-(r / sigma)^2)
            psi[i, j, 1] = 0.8 * env                              # m=+1, L=0
            psi[i, j, 2] = 0.2 * env * r / (r + 0.5) * exp(im * phi)  # m=0, L=1
        end
        psi ./= sqrt(sum(abs2, psi) * dV)

        Lz = orbital_angular_momentum(psi, grid, plans)
        Sz = magnetization(psi, grid, sys)
        Jz = Lz + Sz

        @test Lz > 0.01   # has orbital angular momentum
        @test Sz > 0.5     # mostly m=+1
        @test Jz > 0       # total
    end

    @testset "probability_current: 2D components" begin
        grid = make_grid(GridConfig((32, 32), (10.0, 10.0)))
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        plans = make_fft_plans(grid.config.n_points)

        j = probability_current(psi, grid, plans)
        @test length(j) == 2
        @test size(j[1]) == (32, 32)
        @test size(j[2]) == (32, 32)
    end

    @testset "probability_current: 1D" begin
        grid = make_grid(GridConfig(64, 10.0))
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        plans = make_fft_plans(grid.config.n_points)

        j = probability_current(psi, grid, plans)
        @test length(j) == 1
        @test maximum(abs, j[1]) < 1e-12
    end

end
