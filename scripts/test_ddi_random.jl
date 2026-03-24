#!/usr/bin/env julia
"""
DDIテスト：ランダム初期化から長時間発展
全成分を均等に初期化して、DDI存在下でどのm状態に収束するかを調べる
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
const NSTEPS = 5000

function test_random_init()
    println("DDI Random Init Test")
    println("====================")
    println()

    g = EuFlowerPhaseCPU.Grid(GRID_SIZE, GRID_SIZE, GRID_SIZE÷2, BOX_SIZE,
                              Lz=BOX_SIZE_Z, λz=LAMBDA_Z)

    p = EuFlowerPhaseCPU.Params(N=1.5e4, g1=-0.005, λz=LAMBDA_Z, a_s=135.0, ε_dd=0.44)

    println("g0=$(round(p.g0, digits=2)), gdd=$(round(p.gdd, digits=2))")

    ψ = EuFlowerPhaseCPU.Spinor(g)

    # ランダム初期化（全13成分に乱数を与える）
    EuFlowerPhaseCPU.init_random_perturbation!(ψ, p, g, amp=0.5)
    println("Initialized with random perturbation (amp=0.5)")
    println()

    println("dt=$DT, steps=$NSTEPS")
    println()

    # 各成分のpopulationを計算する関数
    function print_populations(ψ, g)
        pops = Float64[]
        total = 0.0
        for idx in 1:13
            pop = sum(abs2.(ψ.ψ[:,:,:,idx])) * g.dV
            push!(pops, pop)
            total += pop
        end
        pops ./= total

        print("  Populations: ")
        for (idx, pop) in enumerate(pops)
            m = idx - 7  # m = -6 to +6
            if pop > 0.01
                @printf("m=%+d:%.1f%% ", m, pop*100)
            end
        end
        println()
    end

    function callback(step, ψ, p, g)
        if step % 500 == 0
            E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
            E_total = E_kin + E_trap + E_int + E_spin + E_dd
            Mx, My, Mz = EuFlowerPhaseCPU.magnetization(ψ, g)
            @printf("Step %5d: E_kin=%.3f E_trap=%.3f E_int=%.3f E_dd=%.3f | Total=%.4f | Mz/N=%.2f\n",
                    step, E_kin/p.N, E_trap/p.N, E_int/p.N, E_dd/p.N, E_total/p.N, Mz/p.N)
            print_populations(ψ, g)
        end
    end

    # 初期状態
    println("Initial state:")
    E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
    E_total = E_kin + E_trap + E_int + E_spin + E_dd
    Mx, My, Mz = EuFlowerPhaseCPU.magnetization(ψ, g)
    @printf("  E_kin=%.3f E_trap=%.3f E_int=%.3f E_dd=%.3f | Total=%.4f | Mz/N=%.2f\n",
            E_kin/p.N, E_trap/p.N, E_int/p.N, E_dd/p.N, E_total/p.N, Mz/p.N)
    print_populations(ψ, g)
    println()

    t_start = time()
    EuFlowerPhaseCPU.evolve_imag!(ψ, p, g, dt=DT, nsteps=NSTEPS,
                                  callback=callback, callback_interval=100)
    t_elapsed = time() - t_start

    println()
    println("Final state:")
    E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
    E_final = E_kin + E_trap + E_int + E_spin + E_dd
    Mx, My, Mz = EuFlowerPhaseCPU.magnetization(ψ, g)
    @printf("  E/N = %.4f, Mz/N = %.4f, Time: %.1f s\n", E_final/p.N, Mz/p.N, t_elapsed)
    print_populations(ψ, g)
end

test_random_init()
