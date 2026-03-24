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

    @testset "GravityPotential evaluate_potential" begin
        grid_cfg = GridConfig((32,), (10.0,))
        grid = make_grid(grid_cfg)
        grav = GravityPotential{1}(2.5, 1)
        V = evaluate_potential(grav, grid)
        @test size(V) == (32,)
        for i in 1:32
            @test V[i] ≈ 2.5 * grid.x[1][i]
        end

        grid_cfg2 = GridConfig((8, 8), (4.0, 4.0))
        grid2 = make_grid(grid_cfg2)
        grav2 = GravityPotential{2}(9.81, 2)
        V2 = evaluate_potential(grav2, grid2)
        @test size(V2) == (8, 8)
        for I in CartesianIndices((8, 8))
            @test V2[I] ≈ 9.81 * grid2.x[2][I[2]]
        end
    end

    @testset "GravityPotential validation" begin
        @test_throws ArgumentError GravityPotential{1}(9.81, 2)
        @test_throws ArgumentError GravityPotential{2}(9.81, 0)
        @test_throws ArgumentError GravityPotential{2}(9.81, 3)
    end

    @testset "CompositePotential evaluate_potential" begin
        grid_cfg = GridConfig((32,), (10.0,))
        grid = make_grid(grid_cfg)
        harm = HarmonicTrap((1.0,))
        grav = GravityPotential{1}(2.0, 1)
        comp = CompositePotential{1}([harm, grav])
        V = evaluate_potential(comp, grid)
        V_harm = evaluate_potential(harm, grid)
        V_grav = evaluate_potential(grav, grid)
        @test V ≈ V_harm .+ V_grav
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

    @testset "build_potential - gravity" begin
        pc = PotentialConfig(:gravity, Dict{String,Any}("g" => 9.81, "axis" => 1))
        pot = SpinorBEC._build_potential(pc, 1)
        @test pot isa GravityPotential{1}
        @test pot.g == 9.81
        @test pot.axis == 1
    end

    @testset "build_potential - gravity defaults" begin
        pc = PotentialConfig(:gravity, Dict{String,Any}())
        pot = SpinorBEC._build_potential(pc, 2)
        @test pot isa GravityPotential{2}
        @test pot.g == 9.81
        @test pot.axis == 2
    end

    @testset "build_potential - crossed_dipole" begin
        beams = [
            Dict{String,Any}(
                "wavelength" => 1064e-9, "power" => 10.0, "waist" => 50e-6,
                "position" => [0, 0, 0], "direction" => [1, 0, 0]
            ),
        ]
        pc = PotentialConfig(:crossed_dipole, Dict{String,Any}("polarizability" => 1.5e-37, "beams" => beams))
        pot = SpinorBEC._build_potential(pc, 3)
        @test pot isa CrossedDipoleTrap{3}
        @test pot.polarizability == 1.5e-37
        @test length(pot.beams) == 1
    end

    @testset "build_potential - composite" begin
        c1 = PotentialConfig(:harmonic, Dict{String,Any}("omega" => [1.0]))
        c2 = PotentialConfig(:gravity, Dict{String,Any}("g" => 5.0, "axis" => 1))
        pc = PotentialConfig(:composite, Dict{String,Any}("components" => [c1, c2]))
        pot = SpinorBEC._build_potential(pc, 1)
        @test pot isa CompositePotential{1}
        @test length(pot.components) == 2
        @test pot.components[1] isa HarmonicTrap{1}
        @test pot.components[2] isa GravityPotential{1}
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
