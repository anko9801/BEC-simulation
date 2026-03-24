using Test
using SpinorBEC
using LinearAlgebra
using StaticArrays

@testset "Optics" begin

    @testset "OpticalBeam construction and properties" begin
        λ = 460e-9
        w0 = 50e-6
        P = 1e-3
        beam = OpticalBeam(; wavelength=λ, power=P, waist=w0)

        @test waist_radius(beam) ≈ w0 rtol = 1e-10
        @test rayleigh_length(beam) ≈ π * w0^2 / λ rtol = 1e-10
        @test radius_of_curvature(beam) == Inf
        @test peak_intensity(beam) ≈ 2P / (π * w0^2) rtol = 1e-10
        @test divergence_angle(beam) ≈ λ / (π * w0) rtol = 1e-10
    end

    @testset "OpticalBeam with M² > 1" begin
        λ = 460e-9
        w0 = 50e-6
        beam = OpticalBeam(; wavelength=λ, power=1e-3, waist=w0, M2=1.1)

        @test waist_radius(beam) ≈ w0 rtol = 1e-10
        @test rayleigh_length(beam) ≈ π * w0^2 / (1.1 * λ) rtol = 1e-10
        @test divergence_angle(beam) ≈ 1.1 * λ / (π * w0) rtol = 1e-10
    end

    @testset "ABCD: free space propagation" begin
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=100e-6)
        z_R = rayleigh_length(beam)

        propagated = propagate(beam, abcd_free_space(z_R))
        @test waist_radius(propagated) ≈ waist_radius(beam) * √2 rtol = 1e-6
    end

    @testset "ABCD: thin lens focuses to waist" begin
        λ = 1064e-9
        w0 = 1e-3
        f = 0.1
        beam = OpticalBeam(; wavelength=λ, power=1.0, waist=w0)

        # Collimated beam through lens, then propagate to focal plane
        focused = propagate(beam, [abcd_thin_lens(f), abcd_free_space(f)])

        # Focused waist ≈ f λ / (π w0) for collimated input
        w_focus_expected = f * λ / (π * w0)
        @test waist_radius(focused) ≈ w_focus_expected rtol = 0.01
    end

    @testset "ABCD: lens pair (telescope)" begin
        beam = OpticalBeam(; wavelength=532e-9, power=1.0, waist=500e-6)
        f1, f2 = 0.05, 0.1
        d = f1 + f2

        telescope = [abcd_thin_lens(f1), abcd_free_space(d), abcd_thin_lens(f2)]
        output = propagate(beam, telescope)

        magnification = f2 / f1
        @test waist_radius(output) ≈ waist_radius(beam) * magnification rtol = 0.05
    end

    @testset "beam intensity profile" begin
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=100e-6)

        @test beam_intensity(beam, 0.0, 0.0) ≈ peak_intensity(beam) rtol = 1e-10

        w0 = waist_radius(beam)
        @test beam_intensity(beam, w0, 0.0) ≈ peak_intensity(beam) * exp(-2) rtol = 1e-10

        z_R = rayleigh_length(beam)
        @test beam_intensity(beam, 0.0, z_R) ≈ peak_intensity(beam) / 2 rtol = 1e-6
    end

    @testset "mode overlap: identical beams" begin
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=100e-6)
        @test mode_overlap(beam, beam) ≈ 1.0 rtol = 1e-10
    end

    @testset "mode overlap: mismatched waists" begin
        b1 = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=100e-6)
        b2 = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=200e-6)

        η = mode_overlap(b1, b2)
        w1, w2 = 100e-6, 200e-6
        expected = (2 * w1 * w2 / (w1^2 + w2^2))^2
        @test η ≈ expected rtol = 1e-10
        @test 0 < η < 1
    end

    @testset "mode overlap: lateral offset" begin
        beam = OpticalBeam(; wavelength=1064e-9, power=1.0, waist=100e-6)
        η = mode_overlap(beam, beam; lateral_offset=50e-6)
        @test 0 < η < 1

        η_zero = mode_overlap(beam, beam; lateral_offset=0.0)
        @test η < η_zero
    end

    @testset "fiber coupling" begin
        beam = OpticalBeam(; wavelength=460e-9, power=3.5e-3, waist=3.0e-6)
        mfd = 3.5e-6
        η = fiber_coupling(beam, mfd)
        @test 0 < η <= 1.0
    end

end
