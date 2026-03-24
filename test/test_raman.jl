using Test
using SpinorBEC
using LinearAlgebra

@testset "Raman Coupling" begin

    @testset "RamanCoupling construction" begin
        rc = RamanCoupling{1}(1.0, 0.5, (2π / 0.8e-6,))
        @test rc.Omega_R == 1.0
        @test rc.delta == 0.5
    end

    @testset "zero Rabi frequency: no effect" begin
        grid = make_grid(GridConfig(64, 10.0))
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        psi = zeros(ComplexF64, 64, 3)
        psi[:, 2] .= 1.0 / sqrt(64 * cell_volume(grid))

        psi_before = copy(psi)
        rc = RamanCoupling{1}(0.0, 0.0, (0.0,))

        apply_raman_step!(psi, sm, rc, grid, 0.01)
        @test psi ≈ psi_before atol = 1e-14
    end

    @testset "Rabi oscillation: spin-1, k_eff=0" begin
        grid = make_grid(GridConfig(32, 10.0))
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        dV = cell_volume(grid)

        psi = zeros(ComplexF64, 32, 3)
        psi[:, 1] .= 1.0 / sqrt(32 * dV)

        rc = RamanCoupling{1}(1.0, 0.0, (0.0,))

        n_steps = 1000
        dt = 0.01
        for _ in 1:n_steps
            apply_raman_step!(psi, sm, rc, grid, dt)
        end

        pop_m1 = sum(abs2, psi[:, 1]) * dV
        pop_0 = sum(abs2, psi[:, 2]) * dV
        pop_p1 = sum(abs2, psi[:, 3]) * dV
        total = pop_m1 + pop_0 + pop_p1

        @test total ≈ 1.0 rtol = 1e-6
    end

    @testset "Raman preserves norm" begin
        grid = make_grid(GridConfig(32, 10.0))
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        dV = cell_volume(grid)

        psi = zeros(ComplexF64, 32, 3)
        psi[:, 2] .= 1.0 / sqrt(32 * dV)

        rc = RamanCoupling{1}(2.0, 0.3, (1.0,))

        norm_before = sum(abs2, psi) * dV
        for _ in 1:100
            apply_raman_step!(psi, sm, rc, grid, 0.01)
        end
        norm_after = sum(abs2, psi) * dV

        @test norm_after ≈ norm_before rtol = 1e-10
    end

    @testset "nonzero k_eff gives different result from k_eff=0" begin
        grid = make_grid(GridConfig(64, 20.0))
        sys = SpinSystem(1)
        sm = spin_matrices(1)
        dV = cell_volume(grid)

        psi0 = zeros(ComplexF64, 64, 3)
        psi0[:, 2] .= 1.0 / sqrt(64 * dV)

        psi_k0 = copy(psi0)
        rc_k0 = RamanCoupling{1}(2.0, 0.0, (0.0,))
        for _ in 1:100
            apply_raman_step!(psi_k0, sm, rc_k0, grid, 0.01)
        end

        psi_k = copy(psi0)
        k_eff = 2π / 5.0
        rc_k = RamanCoupling{1}(2.0, 0.0, (k_eff,))
        for _ in 1:100
            apply_raman_step!(psi_k, sm, rc_k, grid, 0.01)
        end

        @test !isapprox(psi_k, psi_k0, atol=1e-6)

        phase_k = angle.(psi_k[:, 1])
        phase_variation = maximum(phase_k) - minimum(phase_k)
        @test phase_variation > 0.1
    end

    @testset "workspace with Raman coupling" begin
        grid = make_grid(GridConfig(32, 10.0))
        atom = Rb87
        interactions = compute_interaction_params(atom)
        rc = RamanCoupling{1}(0.5, 0.0, (0.0,))

        sp = SimParams(; dt=0.001, n_steps=10)
        ws = make_workspace(; grid, atom, interactions, sim_params=sp, raman=rc)
        @test ws.raman !== nothing
        @test ws.raman.Omega_R == 0.5

        for _ in 1:10
            split_step!(ws)
        end

        norm = total_norm(ws.state.psi, grid)
        @test norm ≈ 1.0 rtol = 0.01
    end

end
