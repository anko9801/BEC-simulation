@testset "Spherical Harmonics" begin
    @testset "Y_00 = 1/√(4π)" begin
        val = spherical_harmonic(0, 0, 0.5, 1.0)
        @test real(val) ≈ 1 / sqrt(4π) atol=1e-12
        @test imag(val) ≈ 0.0 atol=1e-12
    end

    @testset "Y_10 = √(3/4π) cos(θ)" begin
        for theta in [0.0, 0.3, π / 4, π / 2, π]
            val = spherical_harmonic(1, 0, theta, 0.0)
            @test real(val) ≈ sqrt(3 / (4π)) * cos(theta) atol=1e-12
            @test imag(val) ≈ 0.0 atol=1e-12
        end
    end

    @testset "Y_11 = -√(3/8π) sin(θ) e^{iφ}" begin
        for (theta, phi) in [(0.5, 0.0), (1.0, 1.0), (π / 3, π / 4)]
            val = spherical_harmonic(1, 1, theta, phi)
            expected = -sqrt(3 / (8π)) * sin(theta) * cis(phi)
            @test abs(val - expected) < 1e-12
        end
    end

    @testset "Y_l,-m = (-1)^m conj(Y_lm)" begin
        for l in 0:4
            for m in 1:l
                theta, phi = 0.7, 1.3
                ylm = spherical_harmonic(l, m, theta, phi)
                ylnm = spherical_harmonic(l, -m, theta, phi)
                @test ylnm ≈ (iseven(m) ? 1 : -1) * conj(ylm) atol=1e-12
            end
        end
    end

    @testset "Addition theorem: Σ_m |Y_{lm}|² = (2l+1)/(4π)" begin
        for l in 0:6
            theta = 0.8
            total = sum(abs2(spherical_harmonic(l, m, theta, 0.0)) for m in -l:l)
            @test total ≈ (2l + 1) / (4π) atol=1e-10
        end
    end

    @testset "Orthonormality ∫ Y*_{l1m1} Y_{l2m2} dΩ = δ" begin
        n_theta = 100
        n_phi = 200
        theta = range(1e-6, π - 1e-6, length=n_theta)
        phi = range(0, 2π * (1 - 1 / n_phi), length=n_phi)
        dth = theta[2] - theta[1]
        dph = phi[2] - phi[1]

        for l1 in 0:2, m1 in -l1:l1, l2 in 0:2, m2 in -l2:l2
            integral = 0.0 + 0.0im
            for th in theta
                for ph in phi
                    y1 = spherical_harmonic(l1, m1, th, ph)
                    y2 = spherical_harmonic(l2, m2, th, ph)
                    integral += conj(y1) * y2 * sin(th) * dth * dph
                end
            end
            expected = (l1 == l2 && m1 == m2) ? 1.0 : 0.0
            @test abs(integral - expected) < 0.02
        end
    end

    @testset "spinor_angular_density" begin
        @testset "|F=1,+1> peaks at equator" begin
            F = 1
            spinor = ComplexF64[1.0, 0.0, 0.0]
            result = spinor_angular_density(spinor, F; n_theta=33, n_phi=64)

            # Y_{1,1} ~ sin(θ), so |Y_{1,1}|² peaks at θ=π/2
            mid_theta = argmin(abs.(result.theta .- π / 2))
            max_idx = argmax(result.rho)
            @test max_idx[1] == mid_theta
        end

        @testset "|F=1,0> peaks at poles" begin
            F = 1
            spinor = ComplexF64[0.0, 1.0, 0.0]
            result = spinor_angular_density(spinor, F; n_theta=65, n_phi=128)

            # Y_10 ~ cos(θ), so |Y_10|² peaks at θ=0 and θ=π
            @test result.rho[1, 1] > result.rho[33, 1]  # pole > equator
        end

        @testset "Normalization: ∫ ρ dΩ = 1 for normalized spinor" begin
            F = 2
            spinor = ComplexF64[0, 0, 1, 0, 0]  # |F=2,m=0>
            result = spinor_angular_density(spinor, F; n_theta=100, n_phi=200)

            dth = result.theta[2] - result.theta[1]
            dph = result.phi[2] - result.phi[1]
            integral = 0.0
            for (it, th) in enumerate(result.theta)
                for ip in eachindex(result.phi)
                    integral += result.rho[it, ip] * sin(th) * dth * dph
                end
            end
            @test integral ≈ 1.0 atol=0.02
        end

        @testset "Dimension mismatch throws" begin
            @test_throws DimensionMismatch spinor_angular_density(ComplexF64[1, 0], 2)
        end
    end
end
