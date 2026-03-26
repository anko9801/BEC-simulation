@testset "Losses" begin
    @testset "LossParams constructors" begin
        lp = LossParams(1e-3, 1e-5)
        @test lp.gamma_dr == 1e-3
        @test lp.L3 == 1e-5

        lp2 = LossParams(1e-3)
        @test lp2.L3 == 0.0
    end

    @testset "No loss: LossParams(0,0) preserves norm" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        loss = LossParams(0.0, 0.0)

        N0 = total_norm(psi, grid)
        apply_loss_step!(psi, loss, 1, 0.01, sys.n_components, 1)
        @test total_norm(psi, grid) ≈ N0 rtol = 1e-14
    end

    @testset "m-dependent decay (spin-1)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        loss = LossParams(0.1, 0.0)
        dt = 0.01

        pop_before = [sum(abs2, psi[:, c]) for c in 1:3]

        for _ in 1:100
            apply_loss_step!(psi, loss, 1, dt, sys.n_components, 1)
        end

        pop_after = [sum(abs2, psi[:, c]) for c in 1:3]

        # c=1→m=+1, c=2→m=0, c=3→m=-1
        # (F+m)(F-m+1): m=+1→2*1=2, m=0→1*2=2, m=-1→0*3=0
        # m=-1 (c=3) is stable: γ=0, no decay
        @test pop_after[3] ≈ pop_before[3] rtol = 1e-14

        # m=0 (c=2) and m=+1 (c=1) decay at the same rate for F=1
        @test pop_after[1] < pop_before[1]
        @test pop_after[2] < pop_before[2]
        @test pop_after[1] ≈ pop_after[2] rtol = 1e-10
    end

    @testset "m-dependent decay (spin-2)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(2)
        psi = init_psi(grid, sys; state=:uniform)
        loss = LossParams(0.1, 0.0)
        dt = 0.01

        pop_before = [sum(abs2, psi[:, c]) for c in 1:5]

        for _ in 1:50
            apply_loss_step!(psi, loss, 2, dt, sys.n_components, 1)
        end

        pop_after = [sum(abs2, psi[:, c]) for c in 1:5]

        # Components: c=1→m=+2, c=2→m=+1, c=3→m=0, c=4→m=-1, c=5→m=-2
        # γ_m = Γ * (F+m)(F-m+1) / (2F(2F+1))
        # m=+2: (4)(1)/20 = 4/20
        # m=+1: (3)(2)/20 = 6/20   (maximum)
        # m=0:  (2)(3)/20 = 6/20   (maximum)
        # m=-1: (1)(4)/20 = 4/20
        # m=-2: (0)(5)/20 = 0      (stable)

        # m=-2 (c=5) unchanged
        @test pop_after[5] ≈ pop_before[5] rtol = 1e-14

        # m=+1 (c=2) and m=0 (c=3) decay at same rate (both 6/20)
        @test pop_after[2] ≈ pop_after[3] rtol = 1e-10

        # m=+2 (c=1) and m=-1 (c=4) decay at same rate (both 4/20)
        @test pop_after[1] ≈ pop_after[4] rtol = 1e-10

        # m=0 (c=3) decays more than m=+2 (c=1)
        @test pop_after[3] < pop_after[1]
    end

    @testset "Density dependence" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = zeros(ComplexF64, 64, 3)

        # Component 2 (m=0): put higher density on left half
        psi[1:32, 2] .= 2.0
        psi[33:64, 2] .= 0.5

        pop_left_before = sum(abs2, psi[1:32, 2])
        pop_right_before = sum(abs2, psi[33:64, 2])

        loss = LossParams(0.05, 0.0)
        apply_loss_step!(psi, loss, 1, 0.01, 3, 1)

        pop_left_after = sum(abs2, psi[1:32, 2])
        pop_right_after = sum(abs2, psi[33:64, 2])

        # Higher density region loses more (fractionally)
        frac_left = (pop_left_before - pop_left_after) / pop_left_before
        frac_right = (pop_right_before - pop_right_after) / pop_right_before
        @test frac_left > frac_right
    end

    @testset "L3 loss: uniform across all m" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:uniform)
        loss = LossParams(0.0, 0.1)
        dt = 0.01

        for _ in 1:50
            apply_loss_step!(psi, loss, 1, dt, sys.n_components, 1)
        end

        pop = [sum(abs2, psi[:, c]) for c in 1:3]
        # All components decay at same rate with L3 only
        @test pop[1] ≈ pop[2] rtol = 1e-10
        @test pop[2] ≈ pop[3] rtol = 1e-10
    end

    @testset "loss=nothing in workspace: no error" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=10)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(1.0, 0.0),
            sim_params=sp,
            loss=nothing,
        )
        N0 = total_norm(ws.state.psi, ws.grid)
        for _ in 1:10
            split_step!(ws)
        end
        @test total_norm(ws.state.psi, ws.grid) ≈ N0 rtol = 1e-6
    end

    @testset "loss in workspace: norm decreases" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=100)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(1.0, 0.0),
            sim_params=sp,
            loss=LossParams(0.5, 0.0),
        )
        N0 = total_norm(ws.state.psi, ws.grid)
        for _ in 1:100
            split_step!(ws)
        end
        N1 = total_norm(ws.state.psi, ws.grid)
        @test N1 < N0
    end

    @testset "Loss skipped during imaginary time" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sp = SimParams(; dt=0.01, n_steps=10, imaginary_time=true)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(1.0, 0.0),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            loss=LossParams(100.0, 100.0),
        )
        # Imaginary time normalizes every step, but loss should not be applied
        # If loss were applied during imaginary time the state would collapse
        for _ in 1:10
            split_step!(ws)
        end
        N1 = total_norm(ws.state.psi, ws.grid)
        @test N1 ≈ 1.0 rtol = 1e-6
    end

    @testset "YAML parsing of losses" begin
        yaml = """
        experiment:
          name: loss_test
          system:
            atom: Rb87
            grid:
              n_points: 64
              box_size: 20.0
            interactions:
              c0: 1.0
              c1: 0.0
            losses:
              gamma_dr: 1.0e-3
              L3: 2.0e-5
          sequence: []
        """
        cfg = load_experiment_from_string(yaml)
        @test cfg.system.loss !== nothing
        @test cfg.system.loss.gamma_dr ≈ 1e-3
        @test cfg.system.loss.L3 ≈ 2e-5
    end

    @testset "YAML without losses: loss is nothing" begin
        yaml = """
        experiment:
          name: no_loss_test
          system:
            atom: Rb87
            grid:
              n_points: 64
              box_size: 20.0
            interactions:
              c0: 1.0
              c1: 0.0
          sequence: []
        """
        cfg = load_experiment_from_string(yaml)
        @test cfg.system.loss === nothing
    end

    @testset "YAML parsing of noise_amplitude" begin
        yaml = """
        experiment:
          name: noise_test
          system:
            atom: Rb87
            grid:
              n_points: 64
              box_size: 20.0
            interactions:
              c0: 1.0
              c1: 0.0
          sequence:
            - name: noisy_phase
              duration: 1.0
              dt: 0.01
              noise_amplitude: 0.001
            - name: quiet_phase
              duration: 1.0
              dt: 0.01
        """
        cfg = load_experiment_from_string(yaml)
        @test cfg.sequence[1].noise_amplitude == 0.001
        @test cfg.sequence[2].noise_amplitude === nothing
    end

    @testset "_add_noise! changes psi but preserves norm" begin
        config = GridConfig((32, 32), (10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:polar)
        dV = cell_volume(grid)

        psi_before = copy(psi)
        SpinorBEC._add_noise!(psi, 0.01, sys.n_components, 2, grid)

        @test psi != psi_before
        N1 = sum(abs2, psi) * dV
        @test N1 ≈ 1.0 rtol = 1e-12
    end

    @testset "_add_noise! skips dominant component" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = init_psi(grid, sys; state=:ferromagnetic)
        dominant_before = copy(psi[:, 1])

        SpinorBEC._add_noise!(psi, 0.01, sys.n_components, 1, grid)

        scale = psi[32, 1] / dominant_before[32]
        @test psi[:, 1] ≈ dominant_before .* scale rtol = 1e-10
        @test sum(abs2, psi[:, 2]) > 0
        @test sum(abs2, psi[:, 3]) > 0
    end

    @testset "GroundStateConfig enable_ddi default" begin
        yaml = """
        experiment:
          name: ddi_gs_test
          system:
            atom: Rb87
            grid:
              n_points: 64
              box_size: 20.0
            interactions:
              c0: 1.0
              c1: 0.0
          ground_state:
            dt: 0.01
            n_steps: 100
            tol: 1.0e-8
            zeeman:
              p: 0.0
              q: 0.0
            potential:
              type: harmonic
              omega: [1.0]
          sequence: []
        """
        cfg = load_experiment_from_string(yaml)
        @test cfg.ground_state.enable_ddi == false
    end

    @testset "2D loss step" begin
        config = GridConfig((32, 32), (10.0, 10.0))
        grid = make_grid(config)
        sys = SpinSystem(1)
        psi = zeros(ComplexF64, 32, 32, 3)
        for c in 1:3
            psi[:, :, c] .= 1.0 / sqrt(3 * 32 * 32)
        end
        loss = LossParams(0.1, 0.0)

        pop_before = [sum(abs2, psi[:, :, c]) for c in 1:3]
        apply_loss_step!(psi, loss, 1, 0.01, 3, 2)
        pop_after = [sum(abs2, psi[:, :, c]) for c in 1:3]

        # c=1→m=+1, c=2→m=0, c=3→m=-1
        # m=-1 (c=3) unchanged
        @test pop_after[3] ≈ pop_before[3] rtol = 1e-14
        # m=+1,0 decay
        @test pop_after[1] < pop_before[1]
        @test pop_after[2] < pop_before[2]
    end
end
