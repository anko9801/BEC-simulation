using Test
using SpinorBEC
using LinearAlgebra

@testset "Texture Observables & Extended Interactions" begin

    @testset "InteractionParams backward compatibility" begin
        @testset "2-arg constructor" begin
            ip = InteractionParams(1.0, 2.0)
            @test ip.c0 == 1.0
            @test ip.c1 == 2.0
            @test ip.c_extra == Float64[]
        end

        @testset "3-arg constructor" begin
            ip = InteractionParams(1.0, 2.0, [3.0, 4.0, 5.0])
            @test ip.c0 == 1.0
            @test ip.c1 == 2.0
            @test ip.c_extra == [3.0, 4.0, 5.0]
        end

        @testset "get_cn" begin
            ip = InteractionParams(10.0, 20.0, [30.0, 40.0])
            @test get_cn(ip, 0) == 10.0
            @test get_cn(ip, 1) == 20.0
            @test get_cn(ip, 2) == 30.0
            @test get_cn(ip, 3) == 40.0
            @test get_cn(ip, 4) == 0.0
            @test get_cn(ip, 99) == 0.0
        end

        @testset "get_cn with empty c_extra" begin
            ip = InteractionParams(5.0, 6.0)
            @test get_cn(ip, 0) == 5.0
            @test get_cn(ip, 1) == 6.0
            @test get_cn(ip, 2) == 0.0
        end
    end

    @testset "superfluid_velocity" begin
        @testset "stationary state has zero velocity" begin
            grid = make_grid(GridConfig((32, 32), (10.0, 10.0)))
            sys = SpinSystem(1)
            psi = init_psi(grid, sys; state=:polar)
            plans = make_fft_plans(grid.config.n_points)

            vx, vy = superfluid_velocity(psi, grid, plans)

            @test maximum(abs, vx) < 1e-10
            @test maximum(abs, vy) < 1e-10
        end

        @testset "plane wave gives uniform v = k0" begin
            grid = make_grid(GridConfig((64,), (20.0,)))
            plans = make_fft_plans(grid.config.n_points)
            dV = cell_volume(grid)

            k0 = 2π / grid.config.box_size[1] * 3
            psi = zeros(ComplexF64, 64, 1)
            for i in 1:64
                psi[i, 1] = exp(im * k0 * grid.x[1][i])
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            (vx,) = superfluid_velocity(psi, grid, plans)
            @test all(abs.(vx .- k0) .< 1e-8)
        end

        @testset "zero density gives zero velocity" begin
            grid = make_grid(GridConfig((32,), (10.0,)))
            plans = make_fft_plans(grid.config.n_points)
            psi = zeros(ComplexF64, 32, 1)

            (vx,) = superfluid_velocity(psi, grid, plans)
            @test all(vx .== 0.0)
        end
    end

    @testset "total_angular_momentum" begin
        @testset "J_z = L_z + S_z identity" begin
            N = 32
            L = 10.0
            grid = make_grid(GridConfig((N, N), (L, L)))
            plans = make_fft_plans(grid.config.n_points)
            sys = SpinSystem(1)
            dV = cell_volume(grid)

            psi = zeros(ComplexF64, N, N, 3)
            sigma = L / 8
            for j in 1:N, i in 1:N
                x, y = grid.x[1][i], grid.x[2][j]
                r = sqrt(x^2 + y^2)
                phi = atan(y, x)
                env = exp(-(r / sigma)^2)
                psi[i, j, 1] = 0.8 * env
                psi[i, j, 2] = 0.2 * env * r / (r + 0.5) * exp(im * phi)
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            Lz = orbital_angular_momentum(psi, grid, plans)
            Sz = magnetization(psi, grid, sys)
            Jz = total_angular_momentum(psi, grid, plans, sys)

            @test abs(Jz - (Lz + Sz)) < 1e-12
        end

        @testset "1D: J_z = S_z only" begin
            grid = make_grid(GridConfig(64, 10.0))
            sys = SpinSystem(1)
            plans = make_fft_plans(grid.config.n_points)
            psi = init_psi(grid, sys; state=:polar)

            Jz = total_angular_momentum(psi, grid, plans, sys)
            Sz = magnetization(psi, grid, sys)
            @test abs(Jz - Sz) < 1e-12
        end
    end

    @testset "spin_texture_charge" begin
        @testset "uniform spin gives Q=0" begin
            N = 32
            L = 10.0
            grid = make_grid(GridConfig((N, N), (L, L)))
            plans = make_fft_plans(grid.config.n_points)
            sys = SpinSystem(1)
            sm = spin_matrices(1)

            psi = init_psi(grid, sys; state=:polar)
            Q = spin_texture_charge(psi, grid, plans, sm)
            @test abs(Q) < 1e-10
        end

        @testset "returns 0.0 for 1D" begin
            grid = make_grid(GridConfig(64, 10.0))
            sys = SpinSystem(1)
            sm = spin_matrices(1)
            plans = make_fft_plans(grid.config.n_points)
            psi = init_psi(grid, sys; state=:polar)

            @test spin_texture_charge(psi, grid, plans, sm) == 0.0
        end

        @testset "analytic skyrmion |Q| ≈ 1" begin
            N = 128
            L = 20.0
            grid = make_grid(GridConfig((N, N), (L, L)))
            plans = make_fft_plans(grid.config.n_points)
            sm = spin_matrices(1)
            dV = cell_volume(grid)

            # Spin-1 baby skyrmion: spin texture wraps the sphere once
            # β(r) = π·exp(-r²/R²) goes from π (south pole) at center to 0 (north pole)
            # Analytic Q = (1/2)[-cos β]_{β=π}^{β=0} = -1
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

            Q = spin_texture_charge(psi, grid, plans, sm)
            @test abs(Q + 1.0) < 0.15
        end

        @testset "ferromagnetic state Q=0" begin
            N = 32
            L = 10.0
            grid = make_grid(GridConfig((N, N), (L, L)))
            plans = make_fft_plans(grid.config.n_points)
            sm = spin_matrices(1)
            dV = cell_volume(grid)

            # All spin-up: m=+1 only
            psi = zeros(ComplexF64, N, N, 3)
            sigma = L / 4
            for j in 1:N, i in 1:N
                x, y = grid.x[1][i], grid.x[2][j]
                psi[i, j, 1] = exp(-(x^2 + y^2) / sigma^2)
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            Q = spin_texture_charge(psi, grid, plans, sm)
            @test abs(Q) < 1e-10
        end
    end

    @testset "YAML parsing with higher-order c_n" begin
        @testset "backward compat (no c_extra)" begin
            yaml = """
            experiment:
              name: test
              system:
                atom: Rb87
                grid:
                  n_points: 32
                  box_size: 10.0
                interactions:
                  c0: 100.0
                  c1: -5.0
              sequence: []
            """
            config = load_experiment_from_string(yaml)
            ip = config.system.interactions
            @test ip.c0 == 100.0
            @test ip.c1 == -5.0
            @test ip.c_extra == Float64[]
        end

        @testset "with c2, c3" begin
            yaml = """
            experiment:
              name: test
              system:
                atom: Rb87
                grid:
                  n_points: 32
                  box_size: 10.0
                interactions:
                  c0: 100.0
                  c1: -5.0
                  c2: 3.0
                  c3: 1.5
              sequence: []
            """
            config = load_experiment_from_string(yaml)
            ip = config.system.interactions
            @test ip.c0 == 100.0
            @test ip.c1 == -5.0
            @test ip.c_extra == [3.0, 1.5]
            @test get_cn(ip, 2) == 3.0
            @test get_cn(ip, 3) == 1.5
            @test get_cn(ip, 4) == 0.0
        end
    end

end
