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
end
