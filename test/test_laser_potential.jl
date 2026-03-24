using Test
using SpinorBEC

@testset "LaserBeamPotential" begin

    @testset "1D: peak at beam center" begin
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=50e-6)
        α = 1e-36
        lp = LaserBeamPotential(beam, α, (0.0,), (1.0,))

        grid = make_grid(GridConfig(128, 500e-6))
        V = evaluate_potential(lp, grid)

        mid = argmin(V)
        center_idx = length(grid.x[1]) ÷ 2
        @test abs(mid - center_idx) <= 1

        @test V[mid] < 0
    end

    @testset "2D: Gaussian transverse profile" begin
        w0 = 50e-6
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=w0)
        α = 1e-36
        lp = LaserBeamPotential(beam, α, (0.0, 0.0), (0.0, 1.0))

        grid = make_grid(GridConfig((128, 128), (300e-6, 300e-6)))
        V = evaluate_potential(lp, grid)

        V_center = V[64, 64]
        @test V_center < 0

        I0 = 2 * beam.power / (π * w0^2)
        @test V_center ≈ -α * I0 rtol = 0.05
    end

    @testset "3D: divergence along propagation axis" begin
        w0 = 50e-6
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=w0)
        z_R = rayleigh_length(beam)
        α = 1e-36

        lp = LaserBeamPotential(beam, α, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

        L = 4 * z_R
        grid = make_grid(GridConfig((16, 16, 128), (200e-6, 200e-6, L)))
        V = evaluate_potential(lp, grid)

        cx, cy = 8, 8
        iz_center = argmin(abs.(grid.x[3]))
        iz_zR = argmin(abs.(grid.x[3] .- z_R))

        V_waist = V[cx, cy, iz_center]
        V_zR = V[cx, cy, iz_zR]

        @test abs(V_zR) < abs(V_waist)
        @test abs(V_zR / V_waist) ≈ 0.5 atol = 0.05
    end

    @testset "crossed_laser_trap" begin
        beam1 = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=50e-6)
        beam2 = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=50e-6)
        α = 1e-36

        lp1 = LaserBeamPotential(beam1, α, (0.0, 0.0), (1.0, 0.0))
        lp2 = LaserBeamPotential(beam2, α, (0.0, 0.0), (0.0, 1.0))

        trap = crossed_laser_trap([lp1, lp2])

        grid = make_grid(GridConfig((64, 64), (300e-6, 300e-6)))
        V = evaluate_potential(trap, grid)

        mid = CartesianIndex(32, 32)
        @test V[mid] < 0
        @test V[mid] < V[1, 1]
    end

    @testset "direction normalization" begin
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=50e-6)
        α = 1e-36

        lp = LaserBeamPotential(beam, α, (0.0, 0.0), (3.0, 4.0))

        @test lp.direction[1] ≈ 0.6
        @test lp.direction[2] ≈ 0.8
    end

end
