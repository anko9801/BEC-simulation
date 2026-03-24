#!/usr/bin/env julia
"""
    phase_diagram_gpu.jl

GPU加速版 相図探索スクリプト
RTX 3070Ti で高速に相図をスキャン

実行: julia phase_diagram_gpu.jl
"""

using Printf
using DelimitedFiles

# GPU利用可能かチェック
const USE_GPU = try
    using CUDA
    CUDA.functional()
catch
    false
end

if USE_GPU
    println("Using GPU: $(CUDA.name(CUDA.device()))")
    include("EuFlowerPhaseGPU.jl")
    using .EuFlowerPhaseGPU
    const Module = EuFlowerPhaseGPU
else
    println("GPU not available, using CPU with $(Threads.nthreads()) threads")
    include("EuFlowerPhaseCPU.jl")
    using .EuFlowerPhaseCPU
    const Module = EuFlowerPhaseCPU
end

# ============================================================
# 設定
# ============================================================

# グリッドサイズ（GPU なら大きめ、CPU なら控えめ）
const GRID_SIZE = USE_GPU ? 48 : 32
const BOX_SIZE = 10.0

# 物理パラメータ（固定）
const N_ATOMS = 1.5e4
const A_S = 135.0
const LAMBDA_Z = 0.5

# シミュレーション設定
const DT = 5e-5
const NSTEPS = USE_GPU ? 2500 : 1500

# ============================================================
# 単一パラメータでのシミュレーション
# ============================================================

function run_single(g1::Float64, ε_dd::Float64; verbose::Bool=false)
    if USE_GPU
        g = EuFlowerPhaseGPU.Grid(GRID_SIZE, GRID_SIZE, GRID_SIZE÷2, BOX_SIZE, λz=LAMBDA_Z)
        p = EuFlowerPhaseGPU.Params(N=N_ATOMS, g1=g1, λz=LAMBDA_Z, a_s=A_S, ε_dd=ε_dd)
        ψ = EuFlowerPhaseGPU.SpinorGPU(g)
        EuFlowerPhaseGPU.init_random_perturbation!(ψ, p, g, amp=0.02)
        
        CUDA.@sync EuFlowerPhaseGPU.evolve_imag!(ψ, p, g, dt=DT, nsteps=NSTEPS)
        
        Lz = EuFlowerPhaseGPU.angular_momentum(ψ, g) / p.N
        _, _, Mz = EuFlowerPhaseGPU.magnetization(ψ, g)
        E = EuFlowerPhaseGPU.energy(ψ, p, g) / p.N
        pops = EuFlowerPhaseGPU.component_populations(ψ, g) ./ p.N
    else
        g = EuFlowerPhaseCPU.Grid(GRID_SIZE, GRID_SIZE, GRID_SIZE÷2, BOX_SIZE, λz=LAMBDA_Z)
        p = EuFlowerPhaseCPU.Params(N=N_ATOMS, g1=g1, λz=LAMBDA_Z, a_s=A_S, ε_dd=ε_dd)
        ψ = EuFlowerPhaseCPU.Spinor(g)
        EuFlowerPhaseCPU.init_random_perturbation!(ψ, p, g, amp=0.02)
        
        EuFlowerPhaseCPU.evolve_imag!(ψ, p, g, dt=DT, nsteps=NSTEPS)
        
        Lz = EuFlowerPhaseCPU.angular_momentum(ψ, g) / p.N
        _, _, Mz = EuFlowerPhaseCPU.magnetization(ψ, g)
        E = EuFlowerPhaseCPU.energy(ψ, p, g) / p.N
        pops = EuFlowerPhaseCPU.component_populations(ψ, g) ./ p.N
    end
    
    return (Lz=Lz, Mz=Mz/N_ATOMS, E=E, pops=pops)
end

# ============================================================
# 1D スキャン
# ============================================================

function scan_g1(; ε_dd::Float64=0.44)
    println("\n" * "="^70)
    println("1D Phase Scan: g1 (ε_dd = $ε_dd fixed)")
    println("="^70)
    
    g1_range = range(0.0, -0.025, length=21) |> collect
    
    results = []
    
    for (i, g1) in enumerate(g1_range)
        @printf("\r[%2d/%2d] g1 = %+.4f ... ", i, length(g1_range), g1)
        
        t = @elapsed r = run_single(g1, ε_dd)
        
        push!(results, (g1=g1, ε_dd=ε_dd, Lz=r.Lz, Mz=r.Mz, E=r.E, time=t))
        
        @printf("Lz/N = %.4f (%.1fs)\n", r.Lz, t)
    end
    
    return results
end

function scan_edd(; g1::Float64=-0.005)
    println("\n" * "="^70)
    println("1D Phase Scan: ε_dd (g1 = $g1 fixed)")
    println("="^70)
    
    edd_range = range(0.1, 0.8, length=15) |> collect
    
    results = []
    
    for (i, ε_dd) in enumerate(edd_range)
        @printf("\r[%2d/%2d] ε_dd = %.3f ... ", i, length(edd_range), ε_dd)
        
        t = @elapsed r = run_single(g1, ε_dd)
        
        push!(results, (g1=g1, ε_dd=ε_dd, Lz=r.Lz, Mz=r.Mz, E=r.E, time=t))
        
        @printf("Lz/N = %.4f (%.1fs)\n", r.Lz, t)
    end
    
    return results
end

# ============================================================
# 2D スキャン
# ============================================================

function scan_2d()
    println("\n" * "="^70)
    println("2D Phase Scan: g1 × ε_dd")
    println("="^70)
    
    g1_range = range(0.0, -0.02, length=11) |> collect
    edd_range = range(0.2, 0.7, length=11) |> collect
    
    total = length(g1_range) * length(edd_range)
    println("Total simulations: $total")
    println("Estimated time: ~$(round(total * 3 / 60, digits=1)) min (GPU)")
    println()
    
    Lz_matrix = zeros(length(g1_range), length(edd_range))
    Mz_matrix = zeros(length(g1_range), length(edd_range))
    E_matrix = zeros(length(g1_range), length(edd_range))
    
    count = 0
    t_total = 0.0
    
    for (i, g1) in enumerate(g1_range)
        for (j, ε_dd) in enumerate(edd_range)
            count += 1
            
            t = @elapsed r = run_single(g1, ε_dd)
            t_total += t
            
            Lz_matrix[i, j] = r.Lz
            Mz_matrix[i, j] = r.Mz
            E_matrix[i, j] = r.E
            
            eta = t_total / count * (total - count) / 60
            @printf("\r[%3d/%3d] g1=%+.3f, ε_dd=%.2f: Lz=%.3f (ETA: %.1f min)   ", 
                    count, total, g1, ε_dd, r.Lz, eta)
        end
    end
    println()
    
    return g1_range, edd_range, Lz_matrix, Mz_matrix, E_matrix
end

# ============================================================
# 結果表示・保存
# ============================================================

function print_1d_results(results)
    println("\n" * "-"^70)
    println("Results Summary")
    println("-"^70)
    
    @printf("| %8s | %8s | %8s | %8s | %12s |\n", 
            "g1", "ε_dd", "Lz/N", "Mz/N", "Phase")
    println("|" * "-"^10 * "|" * "-"^10 * "|" * "-"^10 * "|" * "-"^10 * "|" * "-"^14 * "|")
    
    for r in results
        phase = identify_phase(r.Lz, r.Mz)
        @printf("| %+8.4f | %8.3f | %8.4f | %8.4f | %-12s |\n",
                r.g1, r.ε_dd, r.Lz, r.Mz, phase)
    end
end

function identify_phase(Lz, Mz)
    if Lz > 1.0 && Mz > 5.0
        return "Flower/CSV"
    elseif Lz > 0.3
        return "Partial FL"
    elseif Lz > 0.05
        return "Transition"
    elseif Mz > 5.5
        return "Uniform FM"
    else
        return "Other"
    end
end

function save_results(filename, results)
    open(filename, "w") do io
        println(io, "g1,eps_dd,Lz_per_N,Mz_per_N,E_per_N,phase")
        for r in results
            phase = identify_phase(r.Lz, r.Mz)
            @printf(io, "%.6f,%.6f,%.6f,%.6f,%.6f,%s\n",
                    r.g1, r.ε_dd, r.Lz, r.Mz, r.E, phase)
        end
    end
    println("Saved: $filename")
end

function save_2d_results(prefix, g1_range, edd_range, Lz_mat, Mz_mat, E_mat)
    writedlm("$(prefix)_g1.txt", g1_range)
    writedlm("$(prefix)_edd.txt", edd_range)
    writedlm("$(prefix)_Lz.txt", Lz_mat)
    writedlm("$(prefix)_Mz.txt", Mz_mat)
    writedlm("$(prefix)_E.txt", E_mat)
    println("Saved 2D results with prefix: $prefix")
end

function print_2d_ascii(g1_range, edd_range, Lz_matrix)
    println("\n2D Phase Diagram (Lz/N)")
    println("Symbols: ·(<0.1) ○(0.1-0.5) ●(0.5-1) ★(>1)")
    println()
    
    print("g1\\ε_dd ")
    for ε in edd_range[1:2:end]
        @printf("%5.2f ", ε)
    end
    println()
    
    for (i, g1) in enumerate(g1_range)
        @printf("%+.3f: ", g1)
        for j in 1:2:length(edd_range)
            Lz = Lz_matrix[i, j]
            if Lz < 0.1
                print("  ·   ")
            elseif Lz < 0.5
                print("  ○   ")
            elseif Lz < 1.0
                print("  ●   ")
            else
                print("  ★   ")
            end
        end
        println()
    end
end

# ============================================================
# メイン
# ============================================================

function main()
    println("#"^70)
    println("# Eu BEC (F=6) Phase Diagram Exploration")
    println("# Grid: $(GRID_SIZE)³, Steps: $NSTEPS")
    println("#"^70)
    
    # 1D スキャン: g1
    results_g1 = scan_g1(ε_dd=0.44)
    print_1d_results(results_g1)
    save_results("phase_g1_scan.csv", results_g1)
    
    # 1D スキャン: ε_dd
    results_edd = scan_edd(g1=-0.005)
    print_1d_results(results_edd)
    save_results("phase_edd_scan.csv", results_edd)
    
    # 2D スキャン（オプション）
    println("\n" * "="^70)
    print("Run 2D scan? This may take ~10-30 minutes. [y/N]: ")
    
    # 自動実行の場合
    run_2d = false
    
    if run_2d
        g1_r, edd_r, Lz_m, Mz_m, E_m = scan_2d()
        save_2d_results("phase_2d", g1_r, edd_r, Lz_m, Mz_m, E_m)
        print_2d_ascii(g1_r, edd_r, Lz_m)
    else
        println("Skipped.")
    end
    
    # サマリー
    println("\n" * "="^70)
    println("SUMMARY")
    println("="^70)
    println()
    println("Key findings from g1 scan (ε_dd = 0.44):")
    
    flower_threshold = 0.0
    for r in results_g1
        if r.Lz > 0.5
            flower_threshold = r.g1
            break
        end
    end
    
    if flower_threshold < 0
        println("  - Flower phase appears for g1 < $(flower_threshold)")
    else
        println("  - No clear Flower phase transition observed")
    end
    
    println()
    println("Experimental implications:")
    println("  - a_s = $A_S a_B, ε_dd = 0.44 (from your experiment)")
    println("  - If Flower/CSV phase is observed: g1 < 0 (ferromagnetic)")
    println("  - If uniform FM only: g1 ≈ 0 or g1 > 0")
    println()
    println("Next steps:")
    println("  1. Compare simulated spin dynamics with experiment")
    println("  2. Measure Lz via expansion imaging or Bragg spectroscopy")
    println("  3. Refine g1 estimate from spin-mixing rates")
    
    println("\n" * "="^70)
    println("Phase diagram scan completed!")
    println("="^70)
end

main()
