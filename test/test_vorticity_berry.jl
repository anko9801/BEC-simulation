using Test
using SpinorBEC
using LinearAlgebra

@testset "Vorticity & Berry Curvature" begin

    @testset "superfluid_vorticity" begin
        @testset "1D returns 0.0" begin
            grid = make_grid(GridConfig(64, 10.0))
            plans = make_fft_plans(grid.config.n_points)
            psi = init_psi(grid, SpinSystem(1); state=:polar)
            @test superfluid_vorticity(psi, grid, plans) == 0.0
        end

        @testset "stationary state has zero vorticity" begin
            grid = make_grid(GridConfig((32, 32), (10.0, 10.0)))
            plans = make_fft_plans(grid.config.n_points)
            psi = init_psi(grid, SpinSystem(1); state=:polar)

            omega = superfluid_vorticity(psi, grid, plans)
            @test size(omega) == (32, 32)
            @test maximum(abs, omega) < 1e-10
        end

        @testset "plane wave has zero vorticity" begin
            N = 64
            L = 20.0
            grid = make_grid(GridConfig((N, N), (L, L)))
            plans = make_fft_plans(grid.config.n_points)
            dV = cell_volume(grid)

            k0 = 2π / L * 3
            psi = zeros(ComplexF64, N, N, 1)
            for j in 1:N, i in 1:N
                psi[i, j, 1] = exp(im * k0 * grid.x[1][i])
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            omega = superfluid_vorticity(psi, grid, plans)
            @test maximum(abs, omega) < 1e-8
        end

        @testset "vortex with winding 1 has peaked vorticity" begin
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
                core = r / (r + 0.5)
                psi[i, j, 1] = core * exp(-(r / sigma)^2) * exp(im * phi)
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            omega = superfluid_vorticity(psi, grid, plans)
            center = N ÷ 2
            @test omega[center, center] > 0.0
            @test abs(omega[center, center]) > abs(omega[1, 1])
        end
    end

    @testset "berry_curvature" begin
        @testset "1D returns zeros" begin
            grid = make_grid(GridConfig(64, 10.0))
            plans = make_fft_plans(grid.config.n_points)
            sm = spin_matrices(1)
            psi = init_psi(grid, SpinSystem(1); state=:polar)

            omega = berry_curvature(psi, grid, plans, sm)
            @test size(omega) == (64,)
            @test maximum(abs, omega) == 0.0
        end

        @testset "uniform spin gives zero curvature" begin
            N = 32
            L = 10.0
            grid = make_grid(GridConfig((N, N), (L, L)))
            plans = make_fft_plans(grid.config.n_points)
            sm = spin_matrices(1)

            psi = init_psi(grid, SpinSystem(1); state=:ferromagnetic)
            omega = berry_curvature(psi, grid, plans, sm)
            @test maximum(abs, omega) < 1e-10
        end

        @testset "skyrmion consistency with spin_texture_charge" begin
            N = 128
            L = 20.0
            grid = make_grid(GridConfig((N, N), (L, L)))
            plans = make_fft_plans(grid.config.n_points)
            sm = spin_matrices(1)
            dV = cell_volume(grid)

            R = 3.0
            psi = zeros(ComplexF64, N, N, 3)
            for j in 1:N, i in 1:N
                x, y = grid.x[1][i], grid.x[2][j]
                r = sqrt(x^2 + y^2)
                phi = atan(y, x)
                beta = π * exp(-(r / R)^2)
                cb = cos(beta / 2)
                sb = sin(beta / 2)
                psi[i, j, 1] = cb^2
                psi[i, j, 2] = sqrt(2) * sb * cb * exp(im * phi)
                psi[i, j, 3] = sb^2 * exp(2im * phi)
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            omega = berry_curvature(psi, grid, plans, sm)
            Q_from_omega = sum(omega) * dV / (4π)
            Q_from_charge = spin_texture_charge(psi, grid, plans, sm)

            @test abs(Q_from_omega - Q_from_charge) < 1e-12
            @test abs(Q_from_charge + 1.0) < 0.15
        end
    end

    @testset "Mermin-Ho: both functions nonzero for skyrmion" begin
        N = 128
        L = 20.0
        grid = make_grid(GridConfig((N, N), (L, L)))
        plans = make_fft_plans(grid.config.n_points)
        sm = spin_matrices(1)
        dV = cell_volume(grid)

        R = 3.0
        psi = zeros(ComplexF64, N, N, 3)
        for j in 1:N, i in 1:N
            x, y = grid.x[1][i], grid.x[2][j]
            r = sqrt(x^2 + y^2)
            phi = atan(y, x)
            beta = π * exp(-(r / R)^2)
            cb = cos(beta / 2)
            sb = sin(beta / 2)
            psi[i, j, 1] = cb^2
            psi[i, j, 2] = sqrt(2) * sb * cb * exp(im * phi)
            psi[i, j, 3] = sb^2 * exp(2im * phi)
        end
        psi ./= sqrt(sum(abs2, psi) * dV)

        omega_berry = berry_curvature(psi, grid, plans, sm)
        omega_vort = superfluid_vorticity(psi, grid, plans)

        int_berry = sum(omega_berry) * dV
        # Berry curvature integrates to -4π (Q = -1 for spin-1)
        @test abs(int_berry + 4π) < 2.0
        # Vorticity is nonzero locally (peaked near vortex core)
        center = N ÷ 2
        @test abs(omega_vort[center, center]) > 1.0
    end

end
