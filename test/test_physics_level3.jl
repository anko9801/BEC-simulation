using Test
using SpinorBEC
using LinearAlgebra

@testset "Level 3: Spin Dynamics" begin

    @testset "Larmor precession: full system with intermediate checkpoints" begin
        # Equatorial spin coherent state |Fx=+1⟩ = (1/2, 1/√2, 1/2)
        # Under linear Zeeman p: ⟨Fx⟩(t) = cos(pt), ⟨Fy⟩(t) = -sin(pt)
        p = 1.0
        dt = 0.005
        L = 10.0

        gc = GridConfig((32,), (L,))
        grid = make_grid(gc)
        atom = AtomSpecies("test_spin1", 1.0, 1, 0.0, 0.0)
        sm = spin_matrices(1)
        dV = cell_volume(grid)

        ζ = ComplexF64[0.5, 1 / √2, 0.5]
        psi = zeros(ComplexF64, 32, 3)
        for i in 1:32, m in 1:3
            psi[i, m] = ζ[m] / √L
        end

        # Record ⟨Fx⟩ at multiple checkpoints
        checkpoints = [π / 4, π / 2, π, 3π / 2, 2π]
        checkpoint_steps = [round(Int, t / dt) for t in checkpoints]

        sp = SimParams(; dt, n_steps=checkpoint_steps[end], imaginary_time=false, save_every=checkpoint_steps[end])
        ws = make_workspace(;
            grid, atom,
            interactions=InteractionParams(0.0, 0.0),
            zeeman=ZeemanParams(p, 0.0),
            potential=NoPotential(),
            sim_params=sp,
            psi_init=psi,
        )

        step = 0
        cp_idx = 1
        for s in 1:checkpoint_steps[end]
            split_step!(ws)
            step += 1
            if cp_idx <= length(checkpoints) && step == checkpoint_steps[cp_idx]
                t = checkpoints[cp_idx]
                fx, fy, fz = spin_density_vector(ws.state.psi, sm, 1)
                int_fx = sum(fx) * dV
                int_fy = sum(fy) * dV

                @test abs(int_fx - cos(p * t)) < 5e-3
                @test abs(int_fy - (-sin(p * t))) < 5e-3
                cp_idx += 1
            end
        end
    end

    @testset "magnetization conservation during spin dynamics (q≠0)" begin
        # With p=0, q≠0, c₁≠0: spin mixing occurs but ⟨Fz⟩ is conserved
        gc = GridConfig((64,), (15.0,))
        grid = make_grid(gc)
        sys = SpinSystem(1)

        # Initial state with nonzero magnetization
        psi0 = init_psi(grid, sys; state=:ferromagnetic)

        sp = SimParams(; dt=0.005, n_steps=1000, imaginary_time=false, save_every=100)
        ws = make_workspace(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            zeeman=ZeemanParams(0.0, 0.5),
            potential=HarmonicTrap(1.0),
            sim_params=sp,
            psi_init=psi0,
        )

        M0 = magnetization(ws.state.psi, ws.grid, sys)

        # Check at every save point
        for step in 1:sp.n_steps
            split_step!(ws)
            if step % sp.save_every == 0
                M = magnetization(ws.state.psi, ws.grid, sys)
                @test abs(M - M0) < 1e-8
            end
        end
    end

    @testset "spinor phase: ferromagnetic ground state (c₁<0)" begin
        # Rb87 (c₁<0): ground state should be ferromagnetic (m=+1 dominated)
        gc = GridConfig((128,), (20.0,))
        grid = make_grid(gc)

        result = find_ground_state(;
            grid, atom=Rb87,
            interactions=InteractionParams(10.0, -0.5),
            potential=HarmonicTrap(1.0),
            dt=0.005, n_steps=5000,
            initial_state=:ferromagnetic,
        )

        psi = result.workspace.state.psi
        dV = cell_volume(grid)
        n1 = sum(abs2.(component_density(psi, 1, 1))) * dV
        n2 = sum(abs2.(component_density(psi, 1, 2))) * dV
        n3 = sum(abs2.(component_density(psi, 1, 3))) * dV
        n_total = n1 + n2 + n3

        @test n1 / n_total > 0.95
    end

    @testset "spinor phase: antiferromagnetic ground state (c₁>0)" begin
        # Na23 (c₁>0): ground state should be polar (m=0 dominated)
        gc = GridConfig((128,), (20.0,))
        grid = make_grid(gc)

        result = find_ground_state(;
            grid, atom=Na23,
            interactions=InteractionParams(10.0, 0.5),
            potential=HarmonicTrap(1.0),
            dt=0.005, n_steps=5000,
            initial_state=:polar,
        )

        psi = result.workspace.state.psi
        dV = cell_volume(grid)
        n1 = sum(abs2.(component_density(psi, 1, 1))) * dV
        n2 = sum(abs2.(component_density(psi, 1, 2))) * dV
        n3 = sum(abs2.(component_density(psi, 1, 3))) * dV
        n_total = n1 + n2 + n3

        @test n2 / n_total > 0.95
    end

end
