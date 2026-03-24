using Test
using SpinorBEC
using Unitful
using Unitful: @u_str

@testset "Unitful Integration" begin

    @testset "OpticalBeam with units" begin
        beam = OpticalBeam(460u"nm", 3.5u"mW", 50u"μm")

        @test beam.wavelength ≈ 460e-9
        @test beam.power ≈ 3.5e-3
        @test waist_radius(beam) ≈ 50e-6 rtol = 1e-10
    end

    @testset "OpticalBeam with M²" begin
        beam = OpticalBeam(1064u"nm", 1.0u"W", 100u"μm"; M2=1.2)

        @test beam.wavelength ≈ 1064e-9
        @test beam.M2 == 1.2
        @test rayleigh_length(beam) ≈ π * (100e-6)^2 / (1.2 * 1064e-9) rtol = 1e-10
    end

    @testset "OpticalBeam: units vs Float64 equivalence" begin
        b_units = OpticalBeam(532u"nm", 500u"mW", 75u"μm")
        b_float = OpticalBeam(wavelength=532e-9, power=0.5, waist=75e-6)

        @test b_units.q ≈ b_float.q
        @test waist_radius(b_units) ≈ waist_radius(b_float)
        @test peak_intensity(b_units) ≈ peak_intensity(b_float)
    end

    @testset "GridConfig with units" begin
        grid = make_grid(GridConfig(64, 200u"μm"))
        @test grid.config.box_size[1] ≈ 200e-6

        grid2d = make_grid(GridConfig((32, 32), (100u"μm", 200u"μm")))
        @test grid2d.config.box_size[1] ≈ 100e-6
        @test grid2d.config.box_size[2] ≈ 200e-6
    end

    @testset "HarmonicTrap with frequency units" begin
        trap = HarmonicTrap(100u"Hz")
        @test trap.omega[1] ≈ 100 * 2π

        trap2d = HarmonicTrap(100u"Hz", 200u"Hz")
        @test trap2d.omega[1] ≈ 100 * 2π
        @test trap2d.omega[2] ≈ 200 * 2π

        trap3d = HarmonicTrap(100u"Hz", 200u"Hz", 50u"Hz")
        @test trap3d.omega[3] ≈ 50 * 2π
    end

    @testset "LaserBeamPotential with units" begin
        beam = OpticalBeam(1064u"nm", 1.0u"W", 50u"μm")
        α = 1e-36

        lp = LaserBeamPotential(beam, α, (0.0u"μm", 0.0u"μm"), (1.0, 0.0))
        @test lp.position[1] ≈ 0.0
        @test lp.position[2] ≈ 0.0

        lp2 = LaserBeamPotential(beam, α, (100u"μm",), (1.0,))
        @test lp2.position[1] ≈ 100e-6
    end

    @testset "unit mismatch throws" begin
        @test_throws Unitful.DimensionError OpticalBeam(460u"nm", 3.5u"mW", 50u"kg")
    end

end
