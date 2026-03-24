#!/usr/bin/env julia
"""
DDIテスト：長時間発展
"""

using Printf

include("EuFlowerPhaseCPU.jl")
using .EuFlowerPhaseCPU

# 設定
const GRID_SIZE = 64
const BOX_SIZE = 12.0
const BOX_SIZE_Z = 16.0
const LAMBDA_Z = 0.5
const DT = 5e-5
const NSTEPS = 3000

function test_long_evolution(use_ddi::Bool, init_mode::Symbol)
    println("\n" * "="^60)
    println("Long Test: DDI=$(use_ddi), init=$(init_mode)")
    println("="^60)

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

    println("\ndt=$DT, steps=$NSTEPS")

    function callback(step, ψ, p, g)
        if step % 500 == 0
            E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
            E_total = E_kin + E_trap + E_int + E_spin + E_dd
            @printf("  Step %5d: E_kin=%.3f E_trap=%.3f E_int=%.3f E_dd=%.3f | Total=%.4f\n",
                    step, E_kin/p.N, E_trap/p.N, E_int/p.N, E_dd/p.N, E_total/p.N)
        end
    end

    t_start = time()
    EuFlowerPhaseCPU.evolve_imag!(ψ, p, g, dt=DT, nsteps=NSTEPS,
                                  callback=callback, callback_interval=100)
    t_elapsed = time() - t_start

    E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
    E_final = E_kin + E_trap + E_int + E_spin + E_dd
    @printf("\nFinal: E/N = %.4f, Time: %.1f s\n", E_final/p.N, t_elapsed)
end

# テスト実行
println("DDI Long Evolution Test")
println("=======================")

# m=0、DDI有効
test_long_evolution(true, :m0)

# m=+6 (強磁性)、DDI有効
test_long_evolution(true, :ferro)

# m=+6 (強磁性)、DDI無効
test_long_evolution(false, :ferro)

println("\nAll tests done!")
