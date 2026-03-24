```
using Plots
using FFTW
using PlotsGRBackendFontJaEmoji
using ProgressMeter
# CUDA対応
try
    using CUDA
    has_cuda = CUDA.has_cuda()
catch
    has_cuda = false
end

# GPUを使うかどうかのフラグ
USE_GPU = has_cuda # trueでGPU, falseでCPU

if USE_GPU
    println("[INFO] CUDA(GPU)モードで実行します")
    ArrayType = CUDA.CuArray
    fft_func = CUDA.CUFFT.fft
    ifft_func = CUDA.CUFFT.ifft
else
    println("[INFO] CPUモードで実行します")
    ArrayType = Array
    fft_func = fft
    ifft_func = ifft
end

gr()

# --- 物理定数とシミュレーションパラメータ ---
ħ = 1.054571817e-34  # ディラック定数 (J·s)
m = 87 * 1.660539e-27 # ルビジウム87原子の質量 (kg)
a_s = 5.2e-9          # s波散乱長 (m)
N_atoms = 1000        # 原子数

# トラップパラメータ
ω = 2π * 100.0       # トラップ周波数 (rad/s)

# 1Dにおける相互作用の強さ (g = 2ħωa_s)
g = 2 * ħ * ω * a_s

# 数値計算グリッド
L = 50e-6             # 計算領域の長さ (m)
Nx = 256              # グリッドの分割数
dx = L / Nx           # 空間刻み
x = range(-L / 2, L / 2, length = Nx) |> collect |> ArrayType

# 運動量空間のグリッド
dk = 2π / L
k = (fftfreq(Nx, 1 / dx) * 2π) |> ArrayType

# 時間ステップ
dt = 1e-7             # 実時間発展のステップ (s)
dτ = dt / 5           # 虚数時間発展のステップ (s)
total_steps_imag = 2000 # 虚数時間発展のステップ数
total_steps_real = 1000 # 実時間発展のステップ数

# --- ポテンシャルと運動エネルギー演算子 ---
# 調和トラップポテンシャル
V = (0.5 * m * ω^2 * x .^ 2) |> ArrayType

# 運動エネルギー演算子 (フーリエ空間)
T_op = exp.(-im * (ħ * k .^ 2 / (2 * m)) * dt / ħ) |> ArrayType # 実時間用
T_op_imag = exp.(-(ħ^2 * k .^ 2 / (2 * m)) * dτ / ħ) |> ArrayType # 虚数時間用

# --- 波動関数の初期化 ---
# 初期状態としてガウス関数を設定
ψ = exp.(-x .^ 2 / (2 * (1e-6)^2))
ψ = sqrt(N_atoms / sum(abs2.(ψ) * dx)) .* ψ # 原子数で規格化
ψ = ArrayType(ψ)

# --- Part 1: 基底状態の計算 (虚数時間発展) ---
println("虚数時間発展による基底状態の計算を開始...")

let
    ψ_local = copy(ψ)
    prog = Progress(total_steps_imag, 1, "基底状態計算中...")
    anim_imag = @animate for i in 1:total_steps_imag
        # ポテンシャル項 (半ステップ)
        V_eff = V .+ g .* abs2.(ψ_local)
        ψ_local .*= exp.(-V_eff * dτ / (2 * ħ))

        # 運動量項 (1ステップ)
        ψ_k = fft_func(ψ_local)
        ψ_k .*= T_op_imag
        ψ_local = ifft_func(ψ_k)

        # ポテンシャル項 (半ステップ)
        V_eff = V .+ g .* abs2.(ψ_local)
        ψ_local .*= exp.(-V_eff * dτ / (2 * ħ))

        # 原子数で再規格化
        ψ_local .*= sqrt(N_atoms / sum(abs2.(ψ_local) * dx))

        # プロット（プロット用にCPUに戻す）
        plot(
            Array(x) * 1e6,
            Array(abs2.(ψ_local)) / N_atoms,
            title = "虚数時間発展 ステップ: $i",
            xlabel = "位置 (μm)",
            ylabel = "原子密度 (規格化)",
            ylim = (0, maximum(Array(abs2.(ψ_local)) / N_atoms) * 1.2),
            legend = false,
        )
        next!(prog)
    end
    gif(anim_imag, "bec_ground_state.gif", fps = 30)
end
println("基底状態の計算完了。")

# --- Part 2: ダイナミクスの計算 (実時間発展) ---
# 例として、トラップを少し広げてダイナミクスを観察する
println("実時間発展によるダイナミクスのシミュレーションを開始...")
V_dynamic = 0.5 * m * (0.8 * ω)^2 * x .^ 2 # トラップを少し緩める

# 基底状態を初期値とする
ψ_real = copy(ψ)

prog = Progress(total_steps_real, 1, "ダイナミクス計算中...")
anim_real = @animate for i in 1:total_steps_real
    # ポテンシャル項 (半ステップ)
    V_eff = V_dynamic .+ g .* abs2.(ψ_real)
    ψ_real .*= exp.(-im * V_eff * dt / (2 * ħ))

    # 運動量項 (1ステップ)
    ψ_k = fft_func(ψ_real)
    ψ_k .*= T_op
    ψ_real = ifft_func(ψ_k)

    # ポテンシャル項 (半ステップ)
    V_eff = V_dynamic .+ g .* abs2.(ψ_real)
    ψ_real .*= exp.(-im * V_eff * dt / (2 * ħ))

    # プロット（プロット用にCPUに戻す）
    plot(
        Array(x) * 1e6,
        Array(abs2.(ψ_real)) / N_atoms,
        title = "実時間発展 時間: $(round(i*dt*1e3, digits=2)) ms",
        xlabel = "位置 (μm)",
        ylabel = "原子密度 (規格化)",
        ylim = (0, maximum(Array(abs2.(ψ)) / N_atoms) * 1.2),
        legend = false,
    )
    next!(prog)
end

gif(anim_real, "bec_dynamics.gif", fps = 30)
println("シミュレーション完了。")
```