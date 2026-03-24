@testset "Experiment" begin
    @testset "interpolate_value" begin
        c = ConstantValue(3.0)
        @test interpolate_value(c, 0.0) == 3.0
        @test interpolate_value(c, 0.5) == 3.0
        @test interpolate_value(c, 1.0) == 3.0

        r = LinearRamp(1.0, 5.0)
        @test interpolate_value(r, 0.0) == 1.0
        @test interpolate_value(r, 0.5) == 3.0
        @test interpolate_value(r, 1.0) == 5.0
        @test interpolate_value(r, -0.1) == 1.0  # clamped
        @test interpolate_value(r, 1.5) == 5.0   # clamped
    end

    @testset "YAML parsing" begin
        yaml_str = """
        experiment:
          name: "test experiment"
          system:
            atom: Rb87
            grid:
              n_points: [64]
              box_size: [20.0]
            interactions:
              c0: 10.0
              c1: -0.5
          ground_state:
            dt: 0.005
            n_steps: 100
            tol: 1.0e-8
            initial_state: polar
            zeeman:
              p: 0.0
              q: 0.1
            potential:
              type: harmonic
              omega: [1.0]
          sequence:
            - name: ramp
              duration: 1.0
              dt: 0.01
              save_every: 10
              zeeman:
                p:
                  from: 0.0
                  to: 0.5
                q: 0.0
              potential:
                type: harmonic
                omega: [1.0]
            - name: hold
              duration: 2.0
              dt: 0.01
              save_every: 20
              zeeman:
                p: 0.5
                q: 0.0
            - name: release
              duration: 0.5
              dt: 0.005
              save_every: 10
              zeeman:
                p: 0.0
                q: 0.0
              potential:
                type: none
        """

        config = load_experiment_from_string(yaml_str)

        @test config.name == "test experiment"
        @test config.system.atom_name == :Rb87
        @test config.system.grid_n_points == [64]
        @test config.system.grid_box_size == [20.0]
        @test config.system.interactions.c0 == 10.0
        @test config.system.interactions.c1 == -0.5
        @test config.system.ddi.enabled == false

        gs = config.ground_state
        @test gs !== nothing
        @test gs.dt == 0.005
        @test gs.n_steps == 100
        @test gs.tol == 1.0e-8
        @test gs.initial_state == :polar
        @test gs.zeeman.p == 0.0
        @test gs.zeeman.q == 0.1
        @test gs.potential.type == :harmonic
        @test gs.potential.params["omega"] == [1.0]

        @test length(config.sequence) == 3

        ramp = config.sequence[1]
        @test ramp.name == "ramp"
        @test ramp.duration == 1.0
        @test ramp.dt == 0.01
        @test ramp.save_every == 10
        @test ramp.zeeman_p isa LinearRamp
        @test ramp.zeeman_p.from == 0.0
        @test ramp.zeeman_p.to == 0.5
        @test ramp.zeeman_q isa ConstantValue
        @test ramp.zeeman_q.value == 0.0
        @test ramp.potential !== nothing

        hold = config.sequence[2]
        @test hold.name == "hold"
        @test hold.potential === nothing  # inherits

        release = config.sequence[3]
        @test release.name == "release"
        @test release.potential.type == :none
    end

    @testset "YAML parsing - DDI" begin
        yaml_str = """
        experiment:
          name: "ddi test"
          system:
            atom: Eu151
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c0: 5.0
              c1: 0.0
            ddi:
              enabled: true
              c_dd: 1.5e-5
          sequence: []
        """

        config = load_experiment_from_string(yaml_str)
        @test config.system.ddi.enabled == true
        @test config.system.ddi.c_dd == 1.5e-5
        @test config.system.atom_name == :Eu151
    end

    @testset "YAML parsing - minimal" begin
        yaml_str = """
        experiment:
          name: "minimal"
          system:
            atom: Na23
            grid:
              n_points: 32
              box_size: 10.0
            interactions:
              c0: 1.0
              c1: 0.1
          sequence: []
        """

        config = load_experiment_from_string(yaml_str)
        @test config.system.atom_name == :Na23
        @test config.system.grid_n_points == [32]
        @test config.ground_state === nothing
        @test isempty(config.sequence)
    end

    @testset "YAML parsing - gravity" begin
        yaml_str = """
        experiment:
          name: "gravity test"
          system:
            atom: Rb87
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c0: 1.0
              c1: 0.0
          ground_state:
            dt: 0.005
            n_steps: 100
            tol: 1.0e-8
            potential:
              type: gravity
              g: 9.81
              axis: 1
          sequence: []
        """
        config = load_experiment_from_string(yaml_str)
        gs = config.ground_state
        @test gs.potential.type == :gravity
        @test gs.potential.params["g"] == 9.81
        @test gs.potential.params["axis"] == 1
    end

    @testset "YAML parsing - crossed_dipole" begin
        yaml_str = """
        experiment:
          name: "dipole test"
          system:
            atom: Rb87
            grid:
              n_points: [8, 8, 8]
              box_size: [4.0, 4.0, 4.0]
            interactions:
              c0: 1.0
              c1: 0.0
          ground_state:
            dt: 0.005
            n_steps: 50
            tol: 1.0e-6
            potential:
              type: crossed_dipole
              polarizability: 1.5e-37
              beams:
                - wavelength: 1064.0e-9
                  power: 10.0
                  waist: 50.0e-6
                  position: [0, 0, 0]
                  direction: [1, 0, 0]
                - wavelength: 1064.0e-9
                  power: 10.0
                  waist: 50.0e-6
                  position: [0, 0, 0]
                  direction: [0, 1, 0]
          sequence: []
        """
        config = load_experiment_from_string(yaml_str)
        gs = config.ground_state
        @test gs.potential.type == :crossed_dipole
        @test gs.potential.params["polarizability"] == 1.5e-37
        @test length(gs.potential.params["beams"]) == 2
    end

    @testset "YAML parsing - composite (list)" begin
        yaml_str = """
        experiment:
          name: "composite test"
          system:
            atom: Rb87
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c0: 1.0
              c1: 0.0
          ground_state:
            dt: 0.005
            n_steps: 100
            tol: 1.0e-8
            potential:
              - type: harmonic
                omega: [1.0]
              - type: gravity
                g: 9.81
                axis: 1
          sequence: []
        """
        config = load_experiment_from_string(yaml_str)
        gs = config.ground_state
        @test gs.potential.type == :composite
        components = gs.potential.params["components"]
        @test length(components) == 2
        @test components[1].type == :harmonic
        @test components[2].type == :gravity
    end

    @testset "YAML parsing - noise_amplitude" begin
        yaml_with_noise = """
        experiment:
          name: "noise test"
          system:
            atom: Rb87
            grid:
              n_points: [32]
              box_size: [10.0]
            interactions:
              c0: 1.0
              c1: 0.0
          sequence:
            - name: noisy
              duration: 1.0
              dt: 0.01
              noise_amplitude: 0.05
              zeeman:
                p: 0.0
                q: 0.0
              potential:
                type: harmonic
                omega: [1.0]
            - name: quiet
              duration: 1.0
              dt: 0.01
              zeeman:
                p: 0.0
                q: 0.0
        """
        config = load_experiment_from_string(yaml_with_noise)
        @test config.sequence[1].noise_amplitude == 0.05
        @test config.sequence[2].noise_amplitude === nothing
    end

    @testset "run_experiment integration" begin
        yaml_str = """
        experiment:
          name: "integration test"
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
              zeeman:
                p:
                  from: 0.0
                  to: 0.1
                q: 0.1
              potential:
                type: harmonic
                omega: [1.0]
        """

        config = load_experiment_from_string(yaml_str)
        result = run_experiment(config; verbose=false)

        @test result.ground_state_energy !== nothing
        @test result.ground_state_energy isa Float64
        @test result.ground_state_converged isa Bool
        @test length(result.phase_results) == 1
        @test result.phase_names == ["evolve"]

        sim = result.phase_results[1]
        @test length(sim.times) > 1
        @test sim.times[1] == 0.0  # t_offset
        @test all(n -> n > 0, sim.norms)
    end

    @testset "run_experiment integration - noise_amplitude" begin
        yaml_str = """
        experiment:
          name: "noise integration"
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
            - name: noisy_evolve
              duration: 0.1
              dt: 0.001
              save_every: 50
              noise_amplitude: 0.01
              zeeman:
                p: 0.0
                q: 0.1
              potential:
                type: harmonic
                omega: [1.0]
        """

        config = load_experiment_from_string(yaml_str)
        @test config.sequence[1].noise_amplitude == 0.01
        result = run_experiment(config; verbose=false)
        @test length(result.phase_results) == 1
        @test result.phase_names == ["noisy_evolve"]
        sim = result.phase_results[1]
        @test all(n -> n > 0, sim.norms)
    end

    @testset "run_experiment integration - composite potential" begin
        yaml_str = """
        experiment:
          name: "composite integration"
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
              - type: harmonic
                omega: [1.0]
              - type: gravity
                g: 0.1
                axis: 1
          sequence: []
        """

        config = load_experiment_from_string(yaml_str)
        @test config.ground_state.potential.type == :composite
        result = run_experiment(config; verbose=false)
        @test result.ground_state_energy !== nothing
        @test result.ground_state_energy isa Float64
    end
end
