using Test
using SpinorBEC

@testset "Adaptive dt" begin

    @testset "adaptive converges to same ground state" begin
        grid = make_grid(GridConfig(64, 15.0))
        atom = Rb87
        interactions = compute_interaction_params(atom)
        potential = HarmonicTrap(1.0)

        result_fixed = find_ground_state(;
            grid, atom, interactions, potential,
            dt=0.001, n_steps=5000, tol=1e-8,
        )

        result_adaptive = find_ground_state(;
            grid, atom, interactions, potential,
            dt=0.001, n_steps=5000, tol=1e-8,
            adaptive_dt=true, dt_max=0.01,
        )

        @test abs(result_fixed.energy - result_adaptive.energy) / abs(result_fixed.energy) < 0.01
    end

    @testset "AdaptiveDtParams" begin
        p = AdaptiveDtParams()
        @test p.dt_init == 0.001
        @test p.dt_min == 1e-5
        @test p.dt_max == 0.01
        @test p.tol == 1e-3

        p2 = AdaptiveDtParams(dt_init=0.005, dt_min=1e-4, dt_max=0.05, tol=1e-2)
        @test p2.dt_init == 0.005
        @test p2.dt_max == 0.05

        @test_throws ArgumentError AdaptiveDtParams(dt_init=-1.0)
        @test_throws ArgumentError AdaptiveDtParams(dt_min=0.1, dt_max=0.01)
        @test_throws ArgumentError AdaptiveDtParams(tol=-1.0)
    end

    @testset "run_simulation_adaptive!" begin
        grid = make_grid(GridConfig(64, 20.0))
        atom = Rb87
        interactions = InteractionParams(10.0, -0.5)
        potential = HarmonicTrap(1.0)

        sp = SimParams(; dt=0.001, n_steps=1)
        ws = make_workspace(; grid, atom, interactions, potential, sim_params=sp)
        adaptive = AdaptiveDtParams(dt_init=0.001, dt_min=1e-5, dt_max=0.01, tol=1e-3)
        out = run_simulation_adaptive!(ws; adaptive, t_end=0.5, save_interval=0.1)

        # Output structure
        @test out.n_accepted > 0
        @test length(out.result.times) >= 3
        @test out.result.times[1] == 0.0
        @test out.result.times[end] >= 0.5 - 0.01

        # Norm conservation
        N0 = out.result.norms[1]
        for n in out.result.norms
            @test n ≈ N0 rtol = 1e-4
        end

        # dt grows from small initial value
        sp2 = SimParams(; dt=0.001, n_steps=1)
        ws2 = make_workspace(; grid, atom, interactions, potential, sim_params=sp2)
        adaptive2 = AdaptiveDtParams(dt_init=0.0001, dt_min=1e-5, dt_max=0.01, tol=1e-3)
        out2 = run_simulation_adaptive!(ws2; adaptive=adaptive2, t_end=0.5, save_interval=0.5)
        @test out2.final_dt > adaptive2.dt_init

        # Matches fixed dt energy
        sp_fixed = SimParams(; dt=0.001, n_steps=500, save_every=500)
        ws_fixed = make_workspace(; grid, atom, interactions, potential, sim_params=sp_fixed)
        res_fixed = run_simulation!(ws_fixed)
        @test abs(res_fixed.energies[end] - out.result.energies[end]) / abs(res_fixed.energies[end]) < 0.01
    end

    @testset "error estimators" begin
        a = ComplexF64[1.0, 2.0, 3.0]
        @test SpinorBEC._psi_relative_change(a, a) == 0.0
        @test SpinorBEC._density_relative_change(a, a) == 0.0

        c = ComplexF64[1.1, 2.0, 3.0]
        @test 0 < SpinorBEC._psi_relative_change(c, a) < 1
        @test 0 < SpinorBEC._density_relative_change(c, a) < 1

        # Phase rotation: density doesn't change, psi does
        phase = ComplexF64[exp(0.5im), 2exp(0.5im), 3exp(0.5im)]
        @test SpinorBEC._psi_relative_change(phase, a) > 0.1
        @test SpinorBEC._density_relative_change(phase, a) < 1e-14

        # Wavefunction L2 change
        @test SpinorBEC._wavefunction_l2_change(a, a) == 0.0
        @test SpinorBEC._wavefunction_l2_change(c, a) > 0.0
        # Phase rotation IS detected by L2 change (unlike density)
        @test SpinorBEC._wavefunction_l2_change(phase, a) > 0.01
        # Global phase: ψ_new = e^{iφ} ψ_old → L2 change = 2(1 - cos φ)
        global_phase = a .* cis(0.1)
        expected = 2 * (1 - cos(0.1))
        @test SpinorBEC._wavefunction_l2_change(global_phase, a) ≈ expected rtol=1e-10
    end

    @testset "YAML adaptive_dt parsing" begin
        yaml = """
        experiment:
          name: adaptive_test
          system:
            atom: Rb87
            grid:
              n_points: 32
              box_size: 10.0
            interactions:
              c0: 1.0
              c1: 0.0
          sequence:
            - name: fixed_phase
              duration: 1.0
              dt: 0.01
              zeeman:
                p: 0.0
                q: 0.0
            - name: adaptive_phase
              duration: 1.0
              dt: 0.01
              adaptive_dt:
                dt_init: 0.005
                dt_min: 0.0001
                dt_max: 0.05
                tol: 0.002
              zeeman:
                p: 0.0
                q: 0.0
        """
        cfg = load_experiment_from_string(yaml)
        @test cfg.sequence[1].adaptive_dt === nothing
        @test cfg.sequence[2].adaptive_dt !== nothing
        @test cfg.sequence[2].adaptive_dt.dt_init == 0.005
        @test cfg.sequence[2].adaptive_dt.dt_min == 0.0001
        @test cfg.sequence[2].adaptive_dt.dt_max == 0.05
        @test cfg.sequence[2].adaptive_dt.tol == 0.002
    end

    @testset "YAML adaptive_dt defaults" begin
        yaml = """
        experiment:
          name: adaptive_defaults
          system:
            atom: Rb87
            grid:
              n_points: 32
              box_size: 10.0
            interactions:
              c0: 1.0
              c1: 0.0
          sequence:
            - name: phase1
              duration: 1.0
              dt: 0.002
              adaptive_dt: {}
              zeeman:
                p: 0.0
                q: 0.0
        """
        cfg = load_experiment_from_string(yaml)
        ad = cfg.sequence[1].adaptive_dt
        @test ad.dt_init == 0.002   # falls back to phase dt
        @test ad.dt_max == 0.02     # 10 * dt
    end

    @testset "run_experiment with adaptive_dt" begin
        yaml = """
        experiment:
          name: adaptive_integration
          system:
            atom: Rb87
            grid:
              n_points: [32]
              box_size: [20.0]
            interactions:
              c0: 10.0
              c1: -0.5
          ground_state:
            dt: 0.005
            n_steps: 200
            tol: 1.0e-6
            initial_state: polar
            zeeman:
              p: 0.0
              q: 0.1
            potential:
              type: harmonic
              omega: [1.0]
          sequence:
            - name: evolve
              duration: 0.1
              dt: 0.001
              save_every: 50
              adaptive_dt:
                tol: 0.001
              zeeman:
                p: 0.0
                q: 0.1
              potential:
                type: harmonic
                omega: [1.0]
        """
        config = load_experiment_from_string(yaml)
        result = run_experiment(config; verbose=false)
        @test length(result.phase_results) == 1
        sim = result.phase_results[1]
        @test length(sim.times) >= 2
        @test all(n -> n > 0, sim.norms)
    end

end
