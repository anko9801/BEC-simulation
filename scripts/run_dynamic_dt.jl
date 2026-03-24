#!/usr/bin/env julia
"""
動的dtでエネルギー最小化 + 3Dプロット用データ出力
"""

using Printf

include("EuFlowerPhaseCPU.jl")
using .EuFlowerPhaseCPU

# 設定
const GRID_SIZE = 64
const BOX_SIZE = 12.0
const BOX_SIZE_Z = 16.0
const LAMBDA_Z = 0.5

function main()
    println("Dynamic dt Simulation")
    println("="^40)

    g = EuFlowerPhaseCPU.Grid(GRID_SIZE, GRID_SIZE, GRID_SIZE÷2, BOX_SIZE,
                              Lz=BOX_SIZE_Z, λz=LAMBDA_Z)
    p = EuFlowerPhaseCPU.Params(N=1.5e4, g1=-0.005, λz=LAMBDA_Z, a_s=135.0, ε_dd=0.44)

    ψ = EuFlowerPhaseCPU.Spinor(g)
    EuFlowerPhaseCPU.init_random_perturbation!(ψ, p, g, amp=0.02)

    # 動的dt制御
    dt = 1e-4
    dt_min = 1e-6
    dt_max = 5e-4
    tolerance = 1e-6

    E_prev = EuFlowerPhaseCPU.energy(ψ, p, g) / p.N
    @printf("Initial E/N = %.6f\n\n", E_prev)

    history = []
    total_steps = 0

    for iter in 1:200
        EuFlowerPhaseCPU.evolve_imag!(ψ, p, g, dt=dt, nsteps=100)
        total_steps += 100

        E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
        E_curr = (E_kin + E_trap + E_int + E_spin + E_dd) / p.N
        _, _, Mz = EuFlowerPhaseCPU.magnetization(ψ, g)

        dE = E_curr - E_prev
        push!(history, (steps=total_steps, E=E_curr, E_dd=E_dd/p.N, Mz=Mz/p.N, dt=dt))

        if iter % 20 == 1
            @printf("Step %5d: E/N=%.6f, dE=%.2e, dt=%.1e\n", total_steps, E_curr, dE, dt)
        end

        if abs(dE) < tolerance
            println("\nConverged! |dE| < $tolerance")
            break
        end

        # dt調整
        if dE > 0
            dt = max(dt * 0.5, dt_min)
        elseif abs(dE) < tolerance * 10
            dt = min(dt * 1.2, dt_max)
        end

        E_prev = E_curr
    end

    # 最終状態表示
    E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
    E_final = (E_kin + E_trap + E_int + E_spin + E_dd) / p.N
    _, _, Mz = EuFlowerPhaseCPU.magnetization(ψ, g)

    println("\n" * "="^40)
    @printf("Final: E/N=%.6f, E_dd/N=%.4f, Mz/N=%.4f\n", E_final, E_dd/p.N, Mz/p.N)
    println("Total steps: $total_steps")

    # データ保存
    println("\nSaving plot data...")
    density = sum(abs2.(ψ.ψ), dims=4)[:,:,:,1]

    open("density_xy.dat", "w") do f
        z_mid = g.Nz ÷ 2
        for j in 1:g.Ny, i in 1:g.Nx
            println(f, "$(g.x[i]) $(g.y[j]) $(density[i,j,z_mid])")
        end
    end

    open("density_xz.dat", "w") do f
        y_mid = g.Ny ÷ 2
        for k in 1:g.Nz, i in 1:g.Nx
            println(f, "$(g.x[i]) $(g.z[k]) $(density[i,y_mid,k])")
        end
    end

    open("energy_history.dat", "w") do f
        for h in history
            println(f, "$(h.steps) $(h.E) $(h.E_dd) $(h.Mz)")
        end
    end

    println("Saved: density_xy.dat, density_xz.dat, energy_history.dat")
end

main()
