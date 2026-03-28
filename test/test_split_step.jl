using Test
using SpinorBEC

@testset "Split Step" begin
    @testset "Energy conservation (real time, 1D)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)

        psi0 = init_psi(grid, sys; state=:polar)
        interactions = InteractionParams(1.0, 0.1)
        sp = SimParams(; dt=0.001, n_steps=100, imaginary_time=false, save_every=100)

        ws = make_workspace(;
            grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0,
        )

        E0 = total_energy(ws)
        for _ in 1:sp.n_steps
            split_step!(ws)
        end
        E1 = total_energy(ws)

        @test abs(E1 - E0) / abs(E0) < 1e-4
    end

    @testset "Norm conservation (real time, 1D)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)

        psi0 = init_psi(grid, sys; state=:uniform)
        interactions = InteractionParams(0.5, 0.05)
        sp = SimParams(; dt=0.001, n_steps=200, imaginary_time=false, save_every=200)

        ws = make_workspace(;
            grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0,
        )

        N0 = total_norm(ws.state.psi, grid)
        for _ in 1:sp.n_steps
            split_step!(ws)
        end
        N1 = total_norm(ws.state.psi, grid)

        @test abs(N1 - N0) / N0 < 1e-10
    end

    @testset "Convergence order (Strang splitting)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        interactions = InteractionParams(1.0, 0.1)

        T = 0.1
        errors = Float64[]
        dts = [0.01, 0.005, 0.0025]

        for dt in dts
            n_steps = round(Int, T / dt)
            sp = SimParams(; dt, n_steps, imaginary_time=false, save_every=n_steps)
            psi0 = init_psi(grid, sys; state=:polar)
            ws = make_workspace(;
                grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0,
            )
            E0 = total_energy(ws)
            for _ in 1:n_steps
                split_step!(ws)
            end
            E1 = total_energy(ws)
            push!(errors, abs(E1 - E0))
        end

        ratio1 = errors[1] / errors[2]
        ratio2 = errors[2] / errors[3]
        @test ratio1 > 3.0
        @test ratio2 > 3.0
    end

    @testset "Yoshida energy conservation (1D)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        interactions = InteractionParams(1.0, 0.1)
        sp = SimParams(; dt=0.01, n_steps=1, imaginary_time=false, save_every=1)
        psi0 = init_psi(grid, sys; state=:polar)
        ws = make_workspace(; grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0)

        E0 = total_energy(ws)
        n_comp = sys.n_components
        for _ in 1:100
            SpinorBEC._yoshida_core!(ws, 0.01, n_comp)
            ws.state.t += 0.01
        end
        E1 = total_energy(ws)

        @test abs(E1 - E0) / abs(E0) < 1e-4
    end

    @testset "Yoshida norm conservation (1D)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        interactions = InteractionParams(0.5, 0.05)
        sp = SimParams(; dt=0.01, n_steps=1, imaginary_time=false, save_every=1)
        psi0 = init_psi(grid, sys; state=:uniform)
        ws = make_workspace(; grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0)

        N0 = total_norm(ws.state.psi, grid)
        n_comp = sys.n_components
        for _ in 1:200
            SpinorBEC._yoshida_core!(ws, 0.01, n_comp)
        end
        N1 = total_norm(ws.state.psi, grid)

        @test abs(N1 - N0) / N0 < 1e-10
    end

    @testset "Yoshida convergence order > Strang (1D)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        interactions = InteractionParams(1.0, 0.1)
        n_comp = sys.n_components

        T = 0.1
        strang_errors = Float64[]
        yoshida_errors = Float64[]
        dts = [0.02, 0.01, 0.005]

        for dt in dts
            n_steps = round(Int, T / dt)

            # Strang
            sp = SimParams(; dt, n_steps, imaginary_time=false, save_every=n_steps)
            psi0 = init_psi(grid, sys; state=:polar)
            ws = make_workspace(; grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0)
            E0_s = total_energy(ws)
            for _ in 1:n_steps; split_step!(ws); end
            push!(strang_errors, abs(total_energy(ws) - E0_s))

            # Yoshida
            psi0y = init_psi(grid, sys; state=:polar)
            wsy = make_workspace(; grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0y)
            E0_y = total_energy(wsy)
            for _ in 1:n_steps
                SpinorBEC._yoshida_core!(wsy, dt, n_comp)
                wsy.state.t += dt
            end
            push!(yoshida_errors, abs(total_energy(wsy) - E0_y))
        end

        # Yoshida should have better convergence than Strang
        strang_ratio = strang_errors[1] / strang_errors[2]
        yoshida_ratio = yoshida_errors[1] / yoshida_errors[2]
        @test yoshida_ratio > strang_ratio * 1.5
    end

    @testset "Yoshida O(dt^4) convergence (1D)" begin
        config = GridConfig(128, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        interactions = InteractionParams(1.0, 0.1)
        n_comp = sys.n_components

        T = 0.1
        dts = [0.02, 0.01, 0.005]
        errors = Float64[]

        for dt in dts
            n_steps = round(Int, T / dt)
            sp = SimParams(; dt, n_steps, imaginary_time=false, save_every=n_steps)
            psi0 = init_psi(grid, sys; state=:polar)
            ws = make_workspace(; grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0)
            E0 = total_energy(ws)
            for _ in 1:n_steps
                SpinorBEC._yoshida_core!(ws, dt, n_comp)
                ws.state.t += dt
            end
            push!(errors, abs(total_energy(ws) - E0))
        end

        # dt halves → error should drop by ~2^4=16 asymptotically; require > 6
        ratio_1 = errors[1] / errors[2]
        ratio_2 = errors[2] / errors[3]
        @test ratio_1 > 6
        @test ratio_2 > 6
    end

    @testset "Yoshida adaptive simulation (1D)" begin
        config = GridConfig(64, 20.0)
        grid = make_grid(config)
        sys = SpinSystem(1)
        interactions = InteractionParams(1.0, 0.0)
        sp = SimParams(; dt=0.01, n_steps=1)
        psi0 = init_psi(grid, sys; state=:ferromagnetic)
        ws = make_workspace(; grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0,
                            zeeman=ZeemanParams(0.1, 0.0))

        adaptive = AdaptiveDtParams(; dt_init=0.005, dt_min=1e-4, dt_max=0.05, tol=0.01)
        result = run_simulation_yoshida!(ws; adaptive, t_end=0.5, save_interval=0.1)

        @test result.n_accepted > 0
        @test length(result.result.times) >= 2
        N0 = result.result.norms[1]
        N_end = result.result.norms[end]
        @test abs(N_end - N0) / N0 < 1e-6
    end
end
