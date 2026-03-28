@testset "General-F interactions" begin

    @testset "AtomSpecies backward compatibility" begin
        a = AtomSpecies("test", 1.0, 1, 0.1, 0.2)
        @test a.mu_mag == 0.0
        @test a.scattering_lengths == Dict(0 => 0.1, 2 => 0.2)

        b = AtomSpecies("test", 1.0, 1, 0.1, 0.2, 0.5)
        @test b.mu_mag == 0.5
        @test b.scattering_lengths == Dict(0 => 0.1, 2 => 0.2)

        sl = Dict(0 => 1.0, 2 => 2.0, 4 => 3.0)
        c = AtomSpecies("test", 1.0, 2, 0.0, 0.0, 0.0, sl)
        @test c.scattering_lengths == sl

        d = AtomSpecies("test", 1.0, 3, 0.1, 0.0)
        @test isempty(d.scattering_lengths)
    end

    @testset "Existing atoms unchanged" begin
        @test Rb87.F == 1
        @test Rb87.scattering_lengths[0] == Rb87.a0
        @test Rb87.scattering_lengths[2] == Rb87.a2

        @test Na23.F == 1
        @test Na23.scattering_lengths[0] == Na23.a0
        @test Na23.scattering_lengths[2] == Na23.a2

        @test Eu151.F == 6
        @test isempty(Eu151.scattering_lengths)
    end

    @testset "a_s field" begin
        @test Rb87.a_s ≈ (Rb87.a0 + 2 * Rb87.a2) / 3 rtol=1e-12
        @test Na23.a_s ≈ (Na23.a0 + 2 * Na23.a2) / 3 rtol=1e-12
        @test Eu151.a_s == Eu151.a0

        a = AtomSpecies("test", 1.0, 1, 0.3, 0.6)
        @test a.a_s ≈ (0.3 + 2 * 0.6) / 3

        b = AtomSpecies("test", 1.0, 3, 0.5, 0.0)
        @test b.a_s == 0.5
    end

    @testset "compute_interaction_params F=1 unchanged" begin
        params = compute_interaction_params(Rb87; N_atoms=1, dims=3)
        hbar = SpinorBEC.Units.HBAR
        m = Rb87.mass
        a0, a2 = Rb87.a0, Rb87.a2

        c0_expected = 4π * hbar^2 * (a0 + 2a2) / (3m)
        c1_expected = 4π * hbar^2 * (a2 - a0) / (3m)

        @test params.c0 ≈ c0_expected rtol = 1e-10
        @test params.c1 ≈ c1_expected rtol = 1e-10
    end

    @testset "compute_interaction_params_general_f returns zero params" begin
        sl = Dict(0 => 1e-9, 2 => 2e-9, 4 => 1.5e-9)
        atom = AtomSpecies("test-f2", 1e-25, 2, 0.0, 0.0, 0.0, sl)
        params = compute_interaction_params_general_f(atom)
        @test params.c0 == 0.0
        @test params.c1 == 0.0
    end

    @testset "F>1 without scattering_lengths falls back to c0-only" begin
        atom_bare = AtomSpecies("bare", 1.0, 3, 0.1, 0.0)
        ip = @test_logs (:warn, r"No channel scattering lengths") compute_interaction_params(
            atom_bare; N_atoms=1, dims=3)
        @test ip.c0 > 0
        @test ip.c1 == 0.0
    end

    @testset "TensorInteractionCache from scattering_lengths" begin
        sl = Dict(0 => 1e-9, 2 => 2e-9, 4 => 1.5e-9)
        cache = make_tensor_interaction_cache(2, sl; dims=3, N_atoms=100, mass=1e-25)
        @test cache !== nothing
        @test cache.F == 2
        @test cache.D == 5
        @test length(cache.active_channels) == 3
        @test 0 in cache.active_channels
        @test 2 in cache.active_channels
        @test 4 in cache.active_channels
        @test all(isfinite, cache.g_values)
    end

    @testset "Empty scattering_lengths returns nothing" begin
        cache = make_tensor_interaction_cache(2, Dict{Int,Float64}(); dims=3)
        @test cache === nothing
    end
end
