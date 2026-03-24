#!/usr/bin/env julia
"""
    example_eu.jl

¹⁵¹Eu BEC (F=6) のFlower相シミュレーション例

実験パラメータ:
- 原子数: N = 1.5 × 10⁴
- 散乱長: a_s = 135 a_B
- 双極子パラメータ: ε_dd = 0.44
- トラップ: oblate (λz = 0.5)
"""

using Printf

# モジュール読み込み
include("EuFlowerPhase.jl")
using .EuFlowerPhase

# ============================================================
# パラメータ設定
# ============================================================

println("="^60)
println("¹⁵¹Eu BEC (F=6) Flower Phase Simulation")
println("="^60)

# グリッド設定
# 計算コストを考慮して控えめなサイズ
Nx, Ny, Nz = 48, 48, 24
Lx = 12.0  # a_ho単位

println("\n[Grid]")
println("  Nx × Ny × Nz = $Nx × $Ny × $Nz")
println("  Box size: $Lx × $Lx × $(Lx/2) a_ho")

g = Grid(Nx, Ny, Nz, Lx)

# 物理パラメータ
# 実験値に基づく
N_atoms = 1.5e4
a_s = 135.0      # Bohr radii
ε_dd = 0.44
λz = 0.5         # oblate trap

# トラップ周波数から調和振動子長を推定
# ω_⊥ = 2π × 100 Hz, M = 151 u として
# a_ho ≈ 850 nm
a_ho_nm = 850.0

# スピン相互作用 g1 の値（未知なので変数として）
# g1 < 0: 強磁性 (Flower相が期待される)
# g1 = 0: SU(13)対称 (DDIのみが対称性を破る)
# g1 > 0: 反強磁性
g1_values = [0.0, -0.001, -0.005, -0.01]

println("\n[Physical Parameters]")
println("  N = $N_atoms")
println("  a_s = $a_s a_B")
println("  ε_dd = $ε_dd")
println("  λz = $λz (oblate trap)")
println("  a_ho ≈ $a_ho_nm nm")

# ============================================================
# シミュレーション
# ============================================================

function run_simulation(g1_val::Float64; nsteps_imag::Int=5000)
    println("\n" * "="^60)
    println("g1 = $g1_val")
    println("="^60)
    
    # パラメータ構造体
    p = Params(N=N_atoms, g1=g1_val, λz=λz, 
               a_s=a_s, ε_dd=ε_dd, a_ho_nm=a_ho_nm)
    
    println("\n[Computed Parameters]")
    println("  g0 = $(p.g0)")
    println("  gdd = $(p.gdd)")
    println("  gdd/g0 = $(p.gdd/p.g0)")
    
    # 波動関数初期化
    ψ = Spinor(g)
    
    # Flower相への遷移を促すため、摂動を加えた初期状態
    init_random_perturbation!(ψ, p, g, amp=0.02)
    
    # 初期状態の確認
    println("\n[Initial State]")
    pops = component_populations(ψ, g)
    Lz = angular_momentum(ψ, g)
    Mx, My, Mz = magnetization(ψ, g)
    E = energy(ψ, p, g)
    
    @printf("  E/N = %.6f\n", E / p.N)
    @printf("  Lz/N = %.6f\n", Lz / p.N)
    @printf("  Mz/N = %.6f (max: 6)\n", Mz / p.N)
    
    # 虚時間発展
    println("\n[Imaginary Time Evolution]")
    println("  dt = 5e-5, steps = $nsteps_imag")
    
    step_log = Int[]
    E_log = Float64[]
    Lz_log = Float64[]
    
    function my_callback(step, ψ, p, g)
        E = energy(ψ, p, g)
        Lz = angular_momentum(ψ, g)
        push!(step_log, step)
        push!(E_log, E / p.N)
        push!(Lz_log, Lz / p.N)
        
        if step % 1000 == 0
            @printf("  Step %5d: E/N = %.6f, Lz/N = %.6f\n", step, E/p.N, Lz/p.N)
        end
    end
    
    evolve_imag!(ψ, p, g, dt=5e-5, nsteps=nsteps_imag, 
                 callback=my_callback, callback_interval=100)
    
    # 最終状態
    println("\n[Final State]")
    pops = component_populations(ψ, g)
    Lz = angular_momentum(ψ, g)
    Mx, My, Mz = magnetization(ψ, g)
    E_kin, E_trap, E_int, E_spin, E_dd = energy_components(ψ, p, g)
    E_total = E_kin + E_trap + E_int + E_spin + E_dd
    
    @printf("  E_total/N = %.6f\n", E_total / p.N)
    @printf("  E_kin/N   = %.6f\n", E_kin / p.N)
    @printf("  E_trap/N  = %.6f\n", E_trap / p.N)
    @printf("  E_int/N   = %.6f\n", E_int / p.N)
    @printf("  E_spin/N  = %.6f\n", E_spin / p.N)
    @printf("  E_dd/N    = %.6f\n", E_dd / p.N)
    println()
    @printf("  Lz/N = %.6f ℏ\n", Lz / p.N)
    @printf("  Mz/N = %.6f (expect ~6 for FM)\n", Mz / p.N)
    
    println("\n[Component Populations]")
    for m in 6:-1:-6
        idx = EuFlowerPhase.m_to_idx(m)
        pop_frac = pops[idx] / p.N
        if pop_frac > 1e-4
            @printf("  m = %+2d: %.6f (%.2f%%)\n", m, pop_frac, pop_frac*100)
        end
    end
    
    # 相の判定
    println("\n[Phase Identification]")
    Lz_per_N = Lz / p.N
    if Lz_per_N > 0.5
        println("  → Likely FLOWER or CSV phase (Lz/N > 0.5)")
    elseif Lz_per_N > 0.1
        println("  → Transitional region")
    else
        println("  → Likely UNIFORM FM or PCV phase (Lz/N < 0.1)")
    end
    
    return ψ, p, step_log, E_log, Lz_log
end

# ============================================================
# メイン実行
# ============================================================

# まず g1 = 0 (SU(13)対称、DDIのみ) で実行
println("\n" * "#"^60)
println("# Case 1: g1 = 0 (DDI only)")
println("#"^60)

ψ_ddi, p_ddi, _, _, _ = run_simulation(0.0, nsteps_imag=3000)

# g1 < 0 (強磁性) で実行
println("\n" * "#"^60)
println("# Case 2: g1 = -0.005 (Ferromagnetic)")
println("#"^60)

ψ_fm, p_fm, _, _, _ = run_simulation(-0.005, nsteps_imag=3000)

# ============================================================
# 結果サマリー
# ============================================================

println("\n" * "="^60)
println("SUMMARY")
println("="^60)

println("\n| g1     | Lz/N   | Phase       |")
println("|--------|--------|-------------|")

for g1_val in [0.0, -0.005]
    p_test = Params(N=N_atoms, g1=g1_val, λz=λz,
                    a_s=a_s, ε_dd=ε_dd, a_ho_nm=a_ho_nm)
    ψ_test = Spinor(g)
    init_random_perturbation!(ψ_test, p_test, g, amp=0.02)
    evolve_imag!(ψ_test, p_test, g, dt=5e-5, nsteps=2000)
    
    Lz = angular_momentum(ψ_test, g) / p_test.N
    
    phase = Lz > 0.5 ? "Flower/CSV" : (Lz > 0.1 ? "Transition" : "Uniform FM")
    @printf("| %+.3f | %.4f | %-11s |\n", g1_val, Lz, phase)
end

println("\n[Notes]")
println("- Flower phase expected when: R_TF > ξ_dd and c1 < 0")
println("- For εdd = 0.44, DDI effects should be significant")
println("- g1 (spin interaction) is currently unknown for Eu")
println("- Experimental measurement of spin dynamics could determine g1")

println("\n" * "="^60)
println("Simulation completed!")
println("="^60)

