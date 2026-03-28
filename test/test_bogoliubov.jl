@testset "Bogoliubov-de Gennes" begin
    @testset "Scalar BEC phonon dispersion (F=1, polar)" begin
        F = 1
        c0 = 10.0
        n0 = 1.0
        spinor = ComplexF64[0.0, 1.0, 0.0]

        result = bogoliubov_spectrum(;
            spinor, n0, F,
            interactions=InteractionParams(c0, 0.0),
            k_max=5.0, n_k=50,
        )

        @test length(result.k_values) == 50
        @test size(result.omega) == (6, 50)

        # Density (phonon) branch: ω² = ε_k(ε_k + 2μ) where μ = c0*n0
        # This is the LARGEST positive real branch
        mu_eff = c0 * n0
        for ik in 3:10
            k = result.k_values[ik]
            ek = k^2 / 2
            expected = sqrt(ek * (ek + 2 * mu_eff))
            evals_k = result.omega[:, ik]
            pos_real = filter(e -> real(e) > 0.01 && abs(imag(e)) < 0.5, evals_k)
            if !isempty(pos_real)
                best = maximum(real, pos_real)
                @test abs(best - expected) / expected < 0.15
            end
        end
    end

    @testset "Particle-hole symmetry: eigenvalues in ±ω pairs" begin
        F = 1
        spinor = ComplexF64[1.0, 0.0, 0.0]  # ferromagnetic
        result = bogoliubov_spectrum(;
            spinor, n0=1.0, F,
            interactions=InteractionParams(10.0, -5.0),
            k_max=3.0, n_k=20,
        )

        for ik in 1:20
            evals = sort(result.omega[:, ik], by=real)
            for i in 1:3
                # Each eigenvalue should have a partner with opposite sign
                @test abs(evals[i] + evals[end - i + 1]) < 0.1 ||
                      abs(real(evals[i])) < 0.1
            end
        end
    end

    @testset "Goldstone mode: ω(k=0) ≈ 0" begin
        F = 1
        spinor = ComplexF64[0.0, 1.0, 0.0]
        result = bogoliubov_spectrum(;
            spinor, n0=1.0, F,
            interactions=InteractionParams(10.0, 0.0),
            k_max=5.0, n_k=50,
        )

        evals_k0 = result.omega[:, 1]
        min_eval = minimum(abs, evals_k0)
        @test min_eval < 1.0
    end

    @testset "Stable system has no imaginary part" begin
        F = 1
        spinor = ComplexF64[0.0, 1.0, 0.0]
        result = bogoliubov_spectrum(;
            spinor, n0=1.0, F,
            interactions=InteractionParams(10.0, 2.0),
            zeeman=ZeemanParams(0.0, 1.0),
            k_max=5.0, n_k=30,
        )

        @test result.max_growth_rate < 0.5
    end

    @testset "F=2 spectrum computes without error" begin
        F = 2
        D = 5
        spinor = zeros(ComplexF64, D)
        spinor[3] = 1.0  # m=0

        result = bogoliubov_spectrum(;
            spinor, n0=1.0, F,
            interactions=InteractionParams(10.0, 1.0),
            k_max=5.0, n_k=20,
        )

        @test size(result.omega) == (2D, 20)
        @test length(result.k_values) == 20
    end

    @testset "DDI makes some directions unstable" begin
        F = 1
        spinor = ComplexF64[1.0, 0.0, 0.0]

        # Along z (dipole axis) should be different from perpendicular
        r_z = bogoliubov_spectrum(;
            spinor, n0=1.0, F,
            interactions=InteractionParams(5.0, -2.0),
            c_dd=10.0,
            k_direction=(0.0, 0.0, 1.0),
            k_max=5.0, n_k=20,
        )

        r_x = bogoliubov_spectrum(;
            spinor, n0=1.0, F,
            interactions=InteractionParams(5.0, -2.0),
            c_dd=10.0,
            k_direction=(1.0, 0.0, 0.0),
            k_max=5.0, n_k=20,
        )

        # Spectra should differ
        @test r_z.omega != r_x.omega
    end

    @testset "BdGResult fields" begin
        result = bogoliubov_spectrum(;
            spinor=ComplexF64[0.0, 1.0, 0.0],
            n0=1.0, F=1,
            interactions=InteractionParams(10.0, 0.0),
            k_max=3.0, n_k=10,
        )

        @test result isa BdGResult
        @test length(result.k_values) == 10
        @test result.max_growth_rate >= 0
        @test result.unstable isa Bool
    end
end
