#!/usr/bin/env julia
"""
DDIテスト：1ステップでエネルギー変化を確認
"""

using Printf

include("EuFlowerPhaseCPU.jl")
using .EuFlowerPhaseCPU

# 設定
const GRID_SIZE = 64
const BOX_SIZE = 12.0
const BOX_SIZE_Z = 16.0
const LAMBDA_Z = 0.5

# テスト関数
function test_energy_decrease(use_ddi::Bool, init_mode::Symbol)
    println("\n" * "="^50)
    println("Test: DDI=$(use_ddi), init=$(init_mode)")
    println("="^50)

    g = EuFlowerPhaseCPU.Grid(GRID_SIZE, GRID_SIZE, GRID_SIZE÷2, BOX_SIZE,
                              Lz=BOX_SIZE_Z, λz=LAMBDA_Z)

    eps_dd = use_ddi ? 0.44 : 0.0
    p = EuFlowerPhaseCPU.Params(N=1.5e4, g1=-0.005, λz=LAMBDA_Z, a_s=135.0, ε_dd=eps_dd)

    println("g0=$(round(p.g0, digits=2)), gdd=$(round(p.gdd, digits=2))")

    ψ = EuFlowerPhaseCPU.Spinor(g)

    if init_mode == :m0
        EuFlowerPhaseCPU.init_zero_m!(ψ, p, g)
        println("Initialized with m=0 only")
    elseif init_mode == :ferro
        EuFlowerPhaseCPU.init_ferromagnetic!(ψ, p, g)
        println("Initialized with m=+6 (ferromagnetic)")
    else
        EuFlowerPhaseCPU.init_random_perturbation!(ψ, p, g, amp=0.02)
        println("Initialized with random perturbation")
    end

    # m成分の確認
    println("\nComponent populations:")
    for m in -6:6
        idx = EuFlowerPhaseCPU.m_to_idx(m)
        pop = sum(abs2.(ψ.ψ[:,:,:,idx])) * g.dV
        if pop > 1e-6
            @printf("  m=%+d: %.4f\n", m, pop / p.N)
        end
    end

    # DDI係数の確認
    println("\nDDI factors (1 - 3m/F):")
    for m in [-6, 0, 6]
        factor = 1.0 - 3.0 * m / 6.0
        @printf("  m=%+d: factor=%.2f\n", m, factor)
    end

    # エネルギー計算
    E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
    E_before = E_kin + E_trap + E_int + E_spin + E_dd

    @printf("\nBefore: E_kin=%.4f, E_trap=%.4f, E_int=%.4f, E_dd=%.4f, Total=%.4f\n",
            E_kin/p.N, E_trap/p.N, E_int/p.N, E_dd/p.N, E_before/p.N)

    # 1ステップ実行
    EuFlowerPhaseCPU.evolve_imag!(ψ, p, g, dt=1e-5, nsteps=1)

    E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
    E_after = E_kin + E_trap + E_int + E_spin + E_dd

    @printf("After:  E_kin=%.4f, E_trap=%.4f, E_int=%.4f, E_dd=%.4f, Total=%.4f\n",
            E_kin/p.N, E_trap/p.N, E_int/p.N, E_dd/p.N, E_after/p.N)

    dE = (E_after - E_before) / p.N
    @printf("\nΔE/N = %.6f  (%s)\n", dE, dE < 0 ? "✓ DECREASING" : "✗ INCREASING")

    return dE
end

# テスト実行
println("DDI Energy Decrease Test")
println("========================")

# m=0 のみ、DDI無効
test_energy_decrease(false, :m0)

# m=0 のみ、DDI有効
test_energy_decrease(true, :m0)

# m=+6 (強磁性)、DDI無効
test_energy_decrease(false, :ferro)

# m=+6 (強磁性)、DDI有効
test_energy_decrease(true, :ferro)

println("\nDone!")
