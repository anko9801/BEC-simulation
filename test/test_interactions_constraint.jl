@testset "Constraint-based interactions" begin

    @testset "interaction_params_from_constraint basic" begin
        ip = interaction_params_from_constraint(; c_total=4689.0, c1_ratio=0.0, F=6)
        @test ip.c0 ≈ 4689.0
        @test ip.c1 ≈ 0.0

        ip2 = interaction_params_from_constraint(; c_total=4689.0, c1_ratio=1.0/36, F=6)
        @test ip2.c0 + 36 * ip2.c1 ≈ 4689.0 rtol=1e-12
        @test ip2.c1 / ip2.c0 ≈ 1.0 / 36 rtol=1e-12

        ip3 = interaction_params_from_constraint(; c_total=4689.0, c1_ratio=-1.0/72, F=6)
        @test ip3.c0 + 36 * ip3.c1 ≈ 4689.0 rtol=1e-12
        @test ip3.c1 / ip3.c0 ≈ -1.0 / 72 rtol=1e-12
        @test ip3.c1 < 0
    end

    @testset "constraint preserves total for F=1" begin
        ip = interaction_params_from_constraint(; c_total=100.0, c1_ratio=-0.1, F=1)
        @test ip.c0 + 1^2 * ip.c1 ≈ 100.0 rtol=1e-12
    end

    @testset "compute_c_total" begin
        omega = 2π * 110.0
        c_total = compute_c_total(Eu151; N_atoms=50_000, omega_ref=omega)
        @test c_total > 4000
        @test c_total < 5000
    end

    @testset "compute_c_dd_dimless" begin
        omega = 2π * 110.0
        c_dd = compute_c_dd_dimless(Eu151; N_atoms=50_000, omega_ref=omega)
        @test c_dd > 7000
        @test c_dd < 8000
    end

    @testset "linear_zeeman_p" begin
        omega = 2π * 110.0
        p = linear_zeeman_p(Eu151, 2.6e-9, omega)
        @test p > 0.3
        @test p < 0.5
    end

    @testset "AtomSpecies g_F field" begin
        @test Eu151.g_F ≈ 1.9934 * 7.0 / 12.0
        @test Rb87.g_F == -0.5
        @test Na23.g_F == -0.5

        a = AtomSpecies("test", 1.0, 1, 0.1, 0.2)
        @test a.g_F == 0.0

        b = AtomSpecies("test", 1.0, 1, 0.1, 0.2, 0.5)
        @test b.g_F == 0.0

        c = AtomSpecies("test", 1.0, 2, 0.0, 0.0, 0.5, 1.5)
        @test c.g_F == 1.5
        @test c.mu_mag == 0.5
    end

    @testset "YAML c_total/c1_ratio parsing" begin
        yaml_str = """
        experiment:
          name: "constraint test"
          system:
            atom: Eu151
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c_total: 4689.0
              c1_ratio: 0.02778
            ddi:
              enabled: true
              c_dd: 7647.0
          sequence: []
        """
        config = load_experiment_from_string(yaml_str)
        ip = config.system.interactions
        @test ip.c0 + 36 * ip.c1 ≈ 4689.0 rtol=1e-4
        @test ip.c1 / ip.c0 ≈ 0.02778 rtol=1e-4
    end

    @testset "YAML c_total with c1_ratio=0" begin
        yaml_str = """
        experiment:
          name: "zero ratio test"
          system:
            atom: Eu151
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c_total: 4689.0
          sequence: []
        """
        config = load_experiment_from_string(yaml_str)
        ip = config.system.interactions
        @test ip.c0 ≈ 4689.0
        @test ip.c1 ≈ 0.0
    end

    @testset "compute_eu151_interactions" begin
        omega = 2π * 110.0
        ip = compute_eu151_interactions(; N_atoms=50_000, omega_ref=omega, c1_ratio=1.0/36)
        c_total = compute_c_total(Eu151; N_atoms=50_000, omega_ref=omega)
        @test ip.c0 + 36 * ip.c1 ≈ c_total rtol=1e-12
        @test ip.c1 / ip.c0 ≈ 1.0/36 rtol=1e-12

        ip0 = compute_eu151_interactions(; N_atoms=50_000, omega_ref=omega, c1_ratio=0.0)
        @test ip0.c0 ≈ c_total
        @test ip0.c1 ≈ 0.0
    end

    @testset "compute_interaction_params fallback for missing scattering lengths" begin
        ip = @test_logs (:warn, r"No channel scattering lengths") compute_interaction_params(
            Eu151; N_atoms=1, dims=3)
        @test ip.c0 > 0
        @test ip.c1 == 0.0
        @test ip.c0 ≈ compute_c0(Eu151; N_atoms=1, dims=3)
    end

    @testset "_c0c1_to_gS analytic F=1" begin
        c0, c1 = 100.0, -5.0
        g = SpinorBEC._c0c1_to_gS(1, c0, c1)
        @test g[0] ≈ c0 - 2c1   # g₀ = c₀ + c₁(0 - 2)/2 = c₀ - c₁
        @test g[2] ≈ c0 + c1    # g₂ = c₀ + c₁(6 - 4)/2 = c₀ + c₁
    end

    @testset "_c0c1_to_gS pair amplitude identity F=1,2,6" begin
        for F in [1, 2, 6]
            D = 2F + 1
            c0, c1 = 100.0, -3.0
            g = SpinorBEC._c0c1_to_gS(F, c0, c1)
            cg_table = precompute_cg_table(F)
            sm = spin_matrices(F)

            for trial in 1:5
                sp = randn(ComplexF64, D)
                n = sum(abs2, sp)

                Fvec = zeros(ComplexF64, 3)
                Fmats = [sm.Fx, sm.Fy, sm.Fz]
                for (a, Fa) in enumerate(Fmats)
                    for i in 1:D, j in 1:D
                        Fvec[a] += conj(sp[i]) * Fa[i, j] * sp[j]
                    end
                end
                Fsq = sum(abs2, Fvec) |> real

                E_pair = 0.0
                for S in 0:2:2F
                    gS = g[S]
                    for M in -S:S
                        A = zero(ComplexF64)
                        for m1 in -F:F
                            m2 = M - m1
                            abs(m2) > F && continue
                            cg_val = get(cg_table, (S, M, m1, m2), 0.0)
                            c1_idx = F - m1 + 1
                            c2_idx = F - m2 + 1
                            A += cg_val * sp[c1_idx] * sp[c2_idx]
                        end
                        E_pair += gS * abs2(A)
                    end
                end

                E_expected = c0 * n^2 + c1 * Fsq
                @test E_pair ≈ E_expected rtol = 1e-10
            end
        end
    end

    @testset "_c_extra_to_delta_gS" begin
        g = SpinorBEC._c_extra_to_delta_gS(2, [0.0, 0.0, 5.0])
        @test !isempty(g)

        g_empty = SpinorBEC._c_extra_to_delta_gS(2, Float64[])
        @test isempty(g_empty)

        g_odd = SpinorBEC._c_extra_to_delta_gS(2, [0.0, 3.0])
        @test isempty(g_odd)
    end

    @testset "interaction_params_from_constraint with c_extra" begin
        ip = interaction_params_from_constraint(; c_total=4689.0, c1_ratio=1.0/36, F=6,
                                                  c_extra=[0.0, 0.0, 50.0])
        @test ip.c0 + 36 * ip.c1 ≈ 4689.0 rtol=1e-12
        @test length(ip.c_extra) == 3
        @test ip.c_extra[3] ≈ 50.0
    end

    @testset "YAML c_total with c_extra" begin
        yaml_str = """
        experiment:
          name: "c_extra test"
          system:
            atom: Eu151
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c_total: 4689.0
              c1_ratio: 0.02778
              c4: 50.0
              c6: -20.0
          sequence: []
        """
        config = load_experiment_from_string(yaml_str)
        ip = config.system.interactions
        @test ip.c0 + 36 * ip.c1 ≈ 4689.0 rtol=1e-4
        @test length(ip.c_extra) >= 5
        @test ip.c_extra[3] ≈ 50.0   # c4 → c_extra[3]
        @test ip.c_extra[5] ≈ -20.0  # c6 → c_extra[5]
    end

    @testset "YAML explicit c0/c1 still works" begin
        yaml_str = """
        experiment:
          name: "explicit test"
          system:
            atom: Rb87
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c0: 10.0
              c1: -0.5
          sequence: []
        """
        config = load_experiment_from_string(yaml_str)
        @test config.system.interactions.c0 == 10.0
        @test config.system.interactions.c1 == -0.5
    end
end
