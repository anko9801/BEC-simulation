@testset "Clebsch-Gordan / Wigner coefficients" begin

    @testset "Wigner 3j known values" begin
        # (1,1,0; 0,0,0) = -1/√3 (standard Condon-Shortley convention)
        @test wigner_3j(1, 1, 0, 0, 0, 0) ≈ -1 / sqrt(3) atol = 1e-12

        # (1,1,0; 1,-1,0) = 1/√3
        @test wigner_3j(1, 1, 0, 1, -1, 0) ≈ 1 / sqrt(3) atol = 1e-12

        # selection rule: m1+m2+m3 ≠ 0 → 0
        @test wigner_3j(1, 1, 1, 1, 1, 0) == 0.0

        # (1,1,2; 0,0,0) = √(2/15)
        @test wigner_3j(1, 1, 2, 0, 0, 0) ≈ sqrt(2 / 15) atol = 1e-12

        # triangle inequality violation
        @test wigner_3j(1, 1, 5, 0, 0, 0) == 0.0

        # odd l with nonzero m (should be nonzero)
        @test wigner_3j(1, 1, 1, 0, 1, -1) != 0.0
    end

    @testset "CG known values for F=1" begin
        # ⟨1,1;1,-1|0,0⟩ = 1/√3
        @test clebsch_gordan(1, 1, 1, -1, 0, 0) ≈ 1 / sqrt(3) atol = 1e-12

        # ⟨1,0;1,0|0,0⟩ = -1/√3
        @test clebsch_gordan(1, 0, 1, 0, 0, 0) ≈ -1 / sqrt(3) atol = 1e-12

        # ⟨1,-1;1,1|0,0⟩ = 1/√3
        @test clebsch_gordan(1, -1, 1, 1, 0, 0) ≈ 1 / sqrt(3) atol = 1e-12

        # selection rule: m1+m2 ≠ M → 0
        @test clebsch_gordan(1, 1, 1, 1, 0, 0) == 0.0

        # maximal stretch: ⟨F,F;F,F|2F,2F⟩ = 1
        @test clebsch_gordan(2, 2, 2, 2, 4, 4) ≈ 1.0 atol = 1e-10
    end

    @testset "CG orthogonality (fixed l,M)" begin
        for F in [1, 2, 3]
            for l in 0:2F
                for M in -l:l
                    s = 0.0
                    for m1 in -F:F
                        m2 = M - m1
                        abs(m2) > F && continue
                        s += clebsch_gordan(F, m1, F, m2, l, M)^2
                    end
                    @test s ≈ 1.0 atol = 1e-10
                end
            end
        end
    end

    @testset "CG completeness (fixed m1,m2)" begin
        for F in [1, 2]
            for m1 in -F:F, m2 in -F:F
                M = m1 + m2
                s = 0.0
                for l in 0:2F
                    abs(M) > l && continue
                    s += clebsch_gordan(F, m1, F, m2, l, M)^2
                end
                @test s ≈ 1.0 atol = 1e-10
            end
        end
    end

    @testset "Wigner 6j known values" begin
        # {j j 0; j j S} = (-1)^{2j+S} / (2j+1) for triangle-valid S
        @test wigner_6j(1, 1, 0, 1, 1, 0) ≈ 1 / 3 atol = 1e-12
        @test wigner_6j(1, 1, 0, 1, 1, 2) ≈ 1 / 3 atol = 1e-12

        # 6j orthogonality: Σ_l (2l+1) {F F l; F F S1} {F F l; F F S2} = δ_{S1,S2}/(2S1+1)
        for S1 in 0:2, S2 in 0:2
            s = 0.0
            for l in 0:2
                s += (2l + 1) * wigner_6j(1, 1, l, 1, 1, S1) * wigner_6j(1, 1, l, 1, 1, S2)
            end
            expected = S1 == S2 ? 1.0 / (2S1 + 1) : 0.0
            @test s ≈ expected atol = 1e-10
        end
    end

    @testset "F=1 pair amplitude consistency" begin
        # Verify: Σ_S g_S Σ_M |A_{SM}|² = c_0 n² + c_1 |F|²
        # for known F=1 c_0, c_1
        a0, a2 = 5.0, 10.0
        c0 = (a0 + 2a2) / 3
        c1 = (a2 - a0) / 3

        psi = [0.5+0.1im, 0.3-0.2im, 0.4+0.3im]
        sm = spin_matrices(1)
        Fx_m = Matrix(sm.Fx); Fy_m = Matrix(sm.Fy); Fz_m = Matrix(sm.Fz)

        n = sum(abs2, psi)
        fx = real(psi' * Fx_m * psi)
        fy = real(psi' * Fy_m * psi)
        fz = real(psi' * Fz_m * psi)
        V_standard = c0 * n^2 + c1 * (fx^2 + fy^2 + fz^2)

        # Pair amplitude calculation
        V_pair = 0.0
        for (S, gS) in [(0, a0), (2, a2)]
            for M in -S:S
                a = 0.0 + 0.0im
                for m1 in -1:1
                    m2 = M - m1
                    abs(m2) > 1 && continue
                    a += clebsch_gordan(1, m1, 1, m2, S, M) * psi[1-m1+1] * psi[1-m2+1]
                end
                V_pair += gS * abs2(a)
            end
        end

        @test V_pair ≈ V_standard rtol = 1e-10
    end

    @testset "precompute_cg_table" begin
        for F in [1, 2, 3]
            table = precompute_cg_table(F)
            @test !isempty(table)

            for ((l, M, m1, m2), val) in table
                @test m1 + m2 == M
                @test abs(m1) <= F
                @test abs(m2) <= F
                @test iseven(l)
                @test 0 <= l <= 2F
                expected = clebsch_gordan(F, m1, F, m2, l, M)
                @test val ≈ expected atol = 1e-14
            end
        end
    end
end
