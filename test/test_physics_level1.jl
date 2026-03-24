using Test
using SpinorBEC

@testset "Level 1: Conservation Laws" begin

    @testset "norm conservation (machine precision)" begin
        gc = GridConfig((128,), (20.0,))
        grid = make_grid(gc)
        sys = SpinSystem(1)
        psi0 = init_psi(grid, sys; state=:uniform)

        sp = SimParams(; dt=0.002, n_steps=2000, imaginary_time=false, save_every=2000)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            zeeman=ZeemanParams(0.0, 0.1),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            psi_init=psi0,
        )

        N0 = total_norm(ws.state.psi, grid)
        for _ in 1:sp.n_steps
            split_step!(ws)
        end
        N1 = total_norm(ws.state.psi, grid)

        @test abs(N1 - N0) / N0 < 1e-12
    end

    @testset "energy conservation (Strang splitting O(dt²))" begin
        gc = GridConfig((128,), (20.0,))
        grid = make_grid(gc)
        sys = SpinSystem(1)
        psi0 = init_psi(grid, sys; state=:polar)

        sp = SimParams(; dt=0.002, n_steps=2000, imaginary_time=false, save_every=2000)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            zeeman=ZeemanParams(0.0, 0.1),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            psi_init=psi0,
        )

        E0 = total_energy(ws)
        for _ in 1:sp.n_steps
            split_step!(ws)
        end
        E1 = total_energy(ws)

        @test abs(E1 - E0) / abs(E0) < 0.001
    end

    @testset "magnetization conservation (p=0)" begin
        # With p=0, the Hamiltonian has U(1) symmetry around z-axis
        # → magnetization ⟨Fz⟩ is conserved
        gc = GridConfig((128,), (20.0,))
        grid = make_grid(gc)
        sys = SpinSystem(1)

        # Start from ferromagnetic state: M ≈ 1
        psi0 = init_psi(grid, sys; state=:ferromagnetic)

        sp = SimParams(; dt=0.002, n_steps=2000, imaginary_time=false, save_every=2000)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            zeeman=ZeemanParams(0.0, 0.1),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            psi_init=psi0,
        )

        M0 = magnetization(ws.state.psi, ws.grid, sys)
        for _ in 1:sp.n_steps
            split_step!(ws)
        end
        M1 = magnetization(ws.state.psi, ws.grid, sys)

        @test abs(M1 - M0) < 1e-10
    end

end
