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
