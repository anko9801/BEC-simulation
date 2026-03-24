#!/usr/bin/env julia
"""
    example_parallel.jl

Eu BEC シミュレーション - 並列版の使用例

実行方法:
- GPU版: julia example_parallel.jl gpu
- CPU版: julia -t 24 example_parallel.jl cpu
- 比較:  julia -t 24 example_parallel.jl benchmark

必要パッケージ:
    using Pkg
    Pkg.add(["CUDA", "FFTW"])
"""

using Printf
using Statistics

# ============================================================
# バックエンド選択
# ============================================================

const USE_GPU = length(ARGS) > 0 && lowercase(ARGS[1]) == "gpu"
const RUN_BENCHMARK = length(ARGS) > 0 && lowercase(ARGS[1]) == "benchmark"

GPU_AVAILABLE = false

if USE_GPU || RUN_BENCHMARK
    println("Loading GPU module...")
    try
        using CUDA
        include("EuFlowerPhaseGPU.jl")
        using .EuFlowerPhaseGPU
        global GPU_AVAILABLE = true
        println("  GPU module loaded successfully")

        # GPU情報表示
        println("  GPU: $(CUDA.name(CUDA.device()))")
        println("  VRAM: $(round(CUDA.total_memory() / 1024^3, digits=1)) GB")
    catch e
        println("  GPU not available: $e")
    end
end

if !USE_GPU || RUN_BENCHMARK
    println("Loading CPU module...")
    include("EuFlowerPhaseCPU.jl")
    using .EuFlowerPhaseCPU
    println("  CPU threads: $(Threads.nthreads())")
end

# ============================================================
# シミュレーション設定
# ============================================================

# グリッドサイズ（GPU メモリに合わせて調整）
# RTX 3070Ti (8GB) なら 96^3 程度まで可能
const GRID_SIZE = RUN_BENCHMARK ? 48 : 64
const BOX_SIZE = 12.0
const BOX_SIZE_Z = 16.0  # z方向は大きく（λz=0.5でトラップが弱いため）

# 物理パラメータ
const N_ATOMS = 1.5e4
const A_S = 135.0
const EPS_DD = 0.44  # DDI復活
const LAMBDA_Z = 0.5
const G1_VALUE = -0.005  # 強磁性

# シミュレーション設定
const DT = 5e-5
const NSTEPS = RUN_BENCHMARK ? 500 : 3000

# ============================================================
# GPU版シミュレーション（GPUが利用可能な場合のみ定義）
# ============================================================

if GPU_AVAILABLE
    @eval function run_gpu_simulation()
        println("\n" * "="^60)
        println("GPU Simulation (RTX 3070Ti)")
        println("="^60)

        # グリッド
        g = EuFlowerPhaseGPU.Grid($GRID_SIZE, $GRID_SIZE, $GRID_SIZE÷2, $BOX_SIZE,
                                  Lz=$BOX_SIZE_Z, λz=$LAMBDA_Z)
        println("Grid: $(g.Nx)×$(g.Ny)×$(g.Nz), Box: $(g.Lx)×$(g.Ly)×$(g.Lz)")

        # パラメータ
        p = EuFlowerPhaseGPU.Params(N=$N_ATOMS, g1=$G1_VALUE, λz=$LAMBDA_Z, a_s=$A_S, ε_dd=$EPS_DD)
        println("g0=$(round(p.g0, digits=2)), g1=$(p.g1), gdd=$(round(p.gdd, digits=2))")

        # 波動関数
        println("\nInitializing spinor on GPU...")
        ψ = EuFlowerPhaseGPU.SpinorGPU(g)
        EuFlowerPhaseGPU.init_random_perturbation!(ψ, p, g, amp=0.02)

        # GPU warmup
        println("Warming up GPU...")
        CUDA.@sync EuFlowerPhaseGPU.evolve_imag!(ψ, p, g, dt=$DT, nsteps=10)

        # 初期状態
        Lz0 = EuFlowerPhaseGPU.angular_momentum(ψ, g) / p.N
        E0 = EuFlowerPhaseGPU.energy(ψ, p, g) / p.N
        @printf("Initial: E/N = %.6f, Lz/N = %.6f\n", E0, Lz0)

        # 時間発展
        println("\nRunning imaginary time evolution...")
        println("  dt = $($DT), steps = $($NSTEPS)")

        t_start = time()

        function gpu_callback(step, ψ, p, g)
            if step % 500 == 0
                E = EuFlowerPhaseGPU.energy(ψ, p, g)
                Lz = EuFlowerPhaseGPU.angular_momentum(ψ, g)
                @printf("  Step %4d: E/N = %.6f, Lz/N = %.6f\n", step, E/p.N, Lz/p.N)
            end
        end

        CUDA.@sync EuFlowerPhaseGPU.evolve_imag!(ψ, p, g, dt=$DT, nsteps=$NSTEPS,
                                                  callback=gpu_callback, callback_interval=100)

        t_elapsed = time() - t_start

        # 結果
        Lz = EuFlowerPhaseGPU.angular_momentum(ψ, g) / p.N
        E = EuFlowerPhaseGPU.energy(ψ, p, g) / p.N
        _, _, Mz = EuFlowerPhaseGPU.magnetization(ψ, g)

        println("\n[Final State]")
        @printf("  E/N  = %.6f\n", E)
        @printf("  Lz/N = %.6f ℏ\n", Lz)
        @printf("  Mz/N = %.6f\n", Mz / p.N)
        @printf("\n  Time: %.2f s (%.1f steps/s)\n", t_elapsed, $NSTEPS/t_elapsed)

        return t_elapsed, Lz, E
    end
end

# ============================================================
# CPU版シミュレーション
# ============================================================

function run_cpu_simulation()
    println("\n" * "="^60)
    println("CPU Simulation ($(Threads.nthreads()) threads)")
    println("="^60)

    # グリッド
    g = EuFlowerPhaseCPU.Grid(GRID_SIZE, GRID_SIZE, GRID_SIZE÷2, BOX_SIZE,
                              Lz=BOX_SIZE_Z, λz=LAMBDA_Z)
    println("Grid: $(g.Nx)×$(g.Ny)×$(g.Nz), Box: $(g.Lx)×$(g.Ly)×$(g.Lz)")

    # パラメータ
    p = EuFlowerPhaseCPU.Params(N=N_ATOMS, g1=G1_VALUE, λz=LAMBDA_Z, a_s=A_S, ε_dd=EPS_DD)
    println("g0=$(round(p.g0, digits=2)), g1=$(p.g1), gdd=$(round(p.gdd, digits=2))")

    # 波動関数
    println("\nInitializing spinor...")
    ψ = EuFlowerPhaseCPU.Spinor(g)
    EuFlowerPhaseCPU.init_random_perturbation!(ψ, p, g, amp=0.02)

    # 初期状態
    Lz0 = EuFlowerPhaseCPU.angular_momentum(ψ, g) / p.N
    E0 = EuFlowerPhaseCPU.energy(ψ, p, g) / p.N
    @printf("Initial: E/N = %.6f, Lz/N = %.6f\n", E0, Lz0)

    # 時間発展
    println("\nRunning imaginary time evolution...")
    println("  dt = $DT, steps = $NSTEPS")

    t_start = time()

    function cpu_callback(step, ψ, p, g)
        if step % 500 == 0
            E_kin, E_trap, E_int, E_spin, E_dd = EuFlowerPhaseCPU.energy_components(ψ, p, g)
            E_total = E_kin + E_trap + E_int + E_spin + E_dd
            Lz = EuFlowerPhaseCPU.angular_momentum(ψ, g)
            @printf("  Step %5d: E_kin=%.3f E_trap=%.3f E_int=%.3f E_spin=%.6f E_dd=%.3f | Total=%.4f | Lz/N=%.6f\n",
                    step, E_kin/p.N, E_trap/p.N, E_int/p.N, E_spin/p.N, E_dd/p.N, E_total/p.N, Lz/p.N)
        end
    end

    EuFlowerPhaseCPU.evolve_imag!(ψ, p, g, dt=DT, nsteps=NSTEPS,
                                  callback=cpu_callback, callback_interval=100)

    t_elapsed = time() - t_start

    # 結果
    Lz = EuFlowerPhaseCPU.angular_momentum(ψ, g) / p.N
    E = EuFlowerPhaseCPU.energy(ψ, p, g) / p.N
    _, _, Mz = EuFlowerPhaseCPU.magnetization(ψ, g)

    println("\n[Final State]")
    @printf("  E/N  = %.6f\n", E)
    @printf("  Lz/N = %.6f ℏ\n", Lz)
    @printf("  Mz/N = %.6f\n", Mz / p.N)
    @printf("\n  Time: %.2f s (%.1f steps/s)\n", t_elapsed, NSTEPS/t_elapsed)

    return t_elapsed, Lz, E
end

# ============================================================
# ベンチマーク
# ============================================================

function run_benchmark()
    println("\n" * "#"^60)
    println("# BENCHMARK: GPU vs CPU ($NSTEPS steps)")
    println("#"^60)

    results = Dict{String, NamedTuple}()

    # CPU
    println("\n--- CPU Benchmark ---")
    t_cpu, Lz_cpu, E_cpu = run_cpu_simulation()
    results["CPU"] = (time=t_cpu, Lz=Lz_cpu, E=E_cpu)

    # GPU
    if GPU_AVAILABLE
        println("\n--- GPU Benchmark ---")
        t_gpu, Lz_gpu, E_gpu = run_gpu_simulation()
        results["GPU"] = (time=t_gpu, Lz=Lz_gpu, E=E_gpu)
    end

    # 結果比較
    println("\n" * "="^60)
    println("BENCHMARK RESULTS")
    println("="^60)
    println("Grid: $(GRID_SIZE)×$(GRID_SIZE)×$(GRID_SIZE÷2), Steps: $NSTEPS")
    println()

    @printf("| %-10s | %10s | %10s | %10s |\n", "Backend", "Time (s)", "Steps/s", "Speedup")
    println("|" * "-"^12 * "|" * "-"^12 * "|" * "-"^12 * "|" * "-"^12 * "|")

    t_ref = results["CPU"].time

    for (name, r) in sort(collect(results), by=x->x[2].time)
        speedup = t_ref / r.time
        @printf("| %-10s | %10.2f | %10.1f | %10.2fx |\n",
                name, r.time, NSTEPS/r.time, speedup)
    end

    println()
    println("[Physical Results]")
    for (name, r) in results
        @printf("  %s: E/N = %.6f, Lz/N = %.6f\n", name, r.E, r.Lz)
    end
end

# ============================================================
# メイン
# ============================================================

function main()
    println("="^60)
    println("Eu BEC (F=6) Parallel Simulation")
    println("="^60)
    println("Backend: $(USE_GPU ? "GPU" : (RUN_BENCHMARK ? "Benchmark" : "CPU"))")

    if RUN_BENCHMARK
        run_benchmark()
    elseif USE_GPU && GPU_AVAILABLE
        run_gpu_simulation()
    else
        run_cpu_simulation()
    end

    println("\nDone!")
end

main()
