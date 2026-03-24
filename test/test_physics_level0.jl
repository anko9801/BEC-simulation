using Test
using SpinorBEC
using LinearAlgebra

@testset "Level 0: Propagator Unit Tests" begin

    @testset "kinetic propagator: free Gaussian spreading" begin
        # ℏ = m = 1 (dimensionless units)
        # ψ(x,0) ∝ exp(-x²/(2σ₀²)), density variance ⟨x²⟩ = σ₀²/2
        # Free evolution: ⟨x²⟩(T) = σ₀²/2 + T²/(2σ₀²)
        σ₀ = 2.0
        T = 3.0
        dt = 0.01
        n_steps = round(Int, T / dt)

        gc = GridConfig((256,), (40.0,))
        grid = make_grid(gc)

        atom = AtomSpecies("scalar", 1.0, 1, 0.0, 0.0)
        psi = zeros(ComplexF64, 256, 3)
        for i in 1:256
            psi[i, 2] = exp(-grid.x[1][i]^2 / (2 * σ₀^2))
        end
        dV = cell_volume(grid)
        psi ./= sqrt(sum(abs2, psi) * dV)

        sp = SimParams(; dt, n_steps, imaginary_time=false, save_every=n_steps)
        ws = make_workspace(;
            grid, atom,
            interactions=InteractionParams(0.0, 0.0),
            potential=NoPotential(),
            sim_params=sp,
            psi_init=psi,
        )

        for _ in 1:n_steps
            split_step!(ws)
        end

        n_density = component_density(ws.state.psi, 1, 2)
        x = grid.x[1]
        N_total = sum(n_density) * dV
        x_mean = sum(x .* n_density) * dV / N_total
        x2_var = sum((x .- x_mean) .^ 2 .* n_density) * dV / N_total

        σ²_exact = σ₀^2 / 2 + T^2 / (2 * σ₀^2)
        @test abs(x2_var - σ²_exact) / σ²_exact < 1e-3
    end

    @testset "Zeeman propagator: Larmor precession" begin
        # Linear Zeeman p with spin coherent state |Fx=+1⟩ = (1/2, 1/√2, 1/2)
        # ⟨Fx⟩(t) = cos(pt), ⟨Fy⟩(t) = -sin(pt), ⟨Fz⟩ = 0
        p = 1.0
        dt = 0.005
        L = 10.0

        gc = GridConfig((32,), (L,))
        grid = make_grid(gc)
        atom = AtomSpecies("test_spin1", 1.0, 1, 0.0, 0.0)
        sm = spin_matrices(1)

        ζ = ComplexF64[0.5, 1 / √2, 0.5]
        psi = zeros(ComplexF64, 32, 3)
        for i in 1:32, m in 1:3
            psi[i, m] = ζ[m] / √L
        end

        dV = cell_volume(grid)

        # Check at t = π/2 (quarter period) and t = 2π (full period)
        for (T, expected_fx) in [(π / 2, 0.0), (2π, 1.0)]
            test_psi = copy(psi)
            n_steps = round(Int, T / dt)
            sp = SimParams(; dt, n_steps, imaginary_time=false, save_every=n_steps)
            ws = make_workspace(;
                grid, atom,
                interactions=InteractionParams(0.0, 0.0),
                zeeman=ZeemanParams(p, 0.0),
                potential=NoPotential(),
                sim_params=sp,
                psi_init=test_psi,
            )

            for _ in 1:n_steps
                split_step!(ws)
            end

            fx, _, fz = spin_density_vector(ws.state.psi, sm, 1)
            integrated_fx = sum(fx) * dV
            integrated_fz = sum(fz) * dV

            @test abs(integrated_fx - expected_fx) < 5e-3
            @test abs(integrated_fz) < 1e-10
        end
    end

    @testset "spin mixing: polar state is eigenstate" begin
        # Polar state (0,1,0) has ⟨F⟩=0, so H_spin=0 for any c₁
        gc = GridConfig((64,), (10.0,))
        grid = make_grid(gc)
        atom = AtomSpecies("test", 1.0, 1, 0.0, 0.0)

        psi = zeros(ComplexF64, 64, 3)
        for i in 1:64
            psi[i, 2] = exp(-grid.x[1][i]^2 / 4.0)
        end
        dV = cell_volume(grid)
        psi ./= sqrt(sum(abs2, psi) * dV)

        n_before = component_density(psi, 1, 2) |> copy
        sm = spin_matrices(1)

        apply_spin_mixing_step!(psi, sm, 10.0, 1.0, 1; imaginary_time=false)

        n_after = component_density(psi, 1, 2)
        @test maximum(abs.(n_after .- n_before)) < 1e-14
    end

    @testset "spin mixing: ferromagnetic state density preserved" begin
        # State (1,0,0) has ⟨Fz⟩=1, H_spin = c₁ Fz
        # |Fz=+1⟩ is eigenstate of Fz, so U|+1⟩ = exp(-ic₁dt)|+1⟩ — only a global phase
        gc = GridConfig((64,), (10.0,))
        grid = make_grid(gc)

        psi = zeros(ComplexF64, 64, 3)
        for i in 1:64
            psi[i, 1] = exp(-grid.x[1][i]^2 / 4.0)
        end
        dV = cell_volume(grid)
        psi ./= sqrt(sum(abs2, psi) * dV)

        n_before = component_density(psi, 1, 1) |> copy
        sm = spin_matrices(1)

        apply_spin_mixing_step!(psi, sm, -0.5, 1.0, 1; imaginary_time=false)

        n_after = component_density(psi, 1, 1)
        @test maximum(abs.(n_after .- n_before)) < 1e-14

        n_m0 = component_density(psi, 1, 2)
        @test maximum(n_m0) < 1e-28
    end

end
