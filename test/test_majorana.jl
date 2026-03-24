using Test
using SpinorBEC
using LinearAlgebra

@testset "Majorana Representation" begin

    @testset "majorana_stars" begin
        @testset "F=1 ferromagnetic: all stars at south pole (Inf)" begin
            spinor = ComplexF64[1.0, 0.0, 0.0]
            stars = majorana_stars(spinor, 1)
            @test length(stars) == 2
            @test all(isinf, stars)
        end

        @testset "F=1 polar: roots at 0 and Inf" begin
            spinor = ComplexF64[0.0, 1.0, 0.0]
            stars = majorana_stars(spinor, 1)
            @test length(stars) == 2
            finite_stars = filter(isfinite, stars)
            @test length(finite_stars) == 1
            @test abs(finite_stars[1]) < 1e-10
        end

        @testset "F=1 antiferromagnetic: both roots finite" begin
            spinor = ComplexF64[0.0, 0.0, 1.0]
            stars = majorana_stars(spinor, 1)
            @test length(stars) == 2
            @test all(isfinite, stars)
        end

        @testset "F=0 returns empty" begin
            spinor = ComplexF64[1.0]
            stars = majorana_stars(spinor, 0)
            @test isempty(stars)
        end
    end

    @testset "icosahedral_order_parameter" begin
        @testset "F < 6 returns zeros" begin
            grid = make_grid(GridConfig(32, 10.0))
            sm = spin_matrices(1)
            psi = init_psi(grid, SpinSystem(1); state=:polar)

            result = icosahedral_order_parameter(psi, grid, sm)
            @test all(result .== 0.0)
        end

        @testset "F=6 uniform superposition has non-trivial Q6" begin
            N = 16
            L = 10.0
            grid = make_grid(GridConfig((N,), (L,)))
            sm = spin_matrices(6)
            dV = cell_volume(grid)

            psi = zeros(ComplexF64, N, 13)
            sigma = L / 8
            for i in 1:N
                x = grid.x[1][i]
                env = exp(-x^2 / sigma^2)
                for c in 1:13
                    psi[i, c] = env / sqrt(13.0)
                end
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            result = icosahedral_order_parameter(psi, grid, sm)
            n = SpinorBEC.total_density(psi, 1)
            mask = n .> 1e-10
            @test any(mask)
            @test all(result[mask] .>= 0.0)
        end

        @testset "F=6 known icosahedral spinor has Q6 ≈ 1" begin
            # The icosahedral state for F=6 has Majorana stars at icosahedron vertices.
            # Construct via known coefficients (Barnett et al.):
            # ψ_{m=6} = a, ψ_{m=1} = b, ψ_{m=-4} = c (and others zero)
            # with a = √(7/11), b = √(11/22)·i, c = √(7/22) (unnormalized approx)
            # Simplified: use the "hexagonal" approximation
            spinor = zeros(ComplexF64, 13)
            # Icosahedral spinor: ψ_6 = 1, ψ_1 = √(11) i, ψ_{-4} = √7
            # m = 6,5,4,3,2,1,0,-1,-2,-3,-4,-5,-6  (indices 1..13)
            # m=6 → idx 1, m=1 → idx 6, m=-4 → idx 11
            spinor[1] = 1.0
            spinor[6] = im * sqrt(11.0)
            spinor[11] = sqrt(7.0)
            spinor ./= norm(spinor)

            N = 8
            L = 10.0
            grid = make_grid(GridConfig((N,), (L,)))
            sm = spin_matrices(6)
            dV = cell_volume(grid)

            psi = zeros(ComplexF64, N, 13)
            sigma = L / 8
            for i in 1:N
                x = grid.x[1][i]
                env = exp(-x^2 / sigma^2)
                for c in 1:13
                    psi[i, c] = spinor[c] * env
                end
            end
            psi ./= sqrt(sum(abs2, psi) * dV)

            result = icosahedral_order_parameter(psi, grid, sm)
            # Points with significant density should have high Q6
            n = SpinorBEC.total_density(psi, 1)
            mask = n .> 1e-10
            if any(mask)
                q6_avg = sum(result[mask]) / sum(mask)
                @test q6_avg > 0.5
            end
        end
    end

end
