# SpinorBEC.jl

A Julia package for simulating spinor Bose-Einstein condensates using the split-step Fourier method. Supports spin-F GPE in 1D/2D/3D with contact interactions, Zeeman effects, dipole-dipole interactions, and configurable external potentials.

## Features

- **Arbitrary spin F**: spin matrices constructed via angular momentum algebra using StaticArrays
- **N-dimensional**: 1D, 2D, 3D with a single generic code path (`CartesianIndices`)
- **Split-step Fourier**: Strang splitting with nested sub-steps for potential, spin-mixing, and DDI
- **Ground state search**: imaginary time propagation with convergence detection
- **Real-time dynamics**: multi-phase sequences with time-dependent Zeeman ramps
- **Potentials**: harmonic trap, gravity, crossed dipole trap (Gaussian beams), and composites
- **DDI**: dipole-dipole interaction via k-space convolution of the Q tensor
- **YAML experiments**: declarative configuration for reproducible multi-phase simulations
- **Predefined atoms**: Rb87 (F=1, ferromagnetic), Na23 (F=1, antiferromagnetic), Eu151 (F=6, dipolar)

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/BEC-simulation")
```

## Quick Start

### Julia API

```julia
using SpinorBEC

grid = make_grid(GridConfig((64,), (20.0,)))
atom = Rb87
interactions = InteractionParams(10.0, -0.5)
potential = HarmonicTrap((1.0,))

result = find_ground_state(;
    grid, atom, interactions, potential,
    dt=0.005, n_steps=5000, tol=1e-10,
    initial_state=:polar,
)

println("Converged: $(result.converged), E = $(result.energy)")
```

### YAML Experiment

```yaml
experiment:
  name: "Rb87 quench dynamics"
  system:
    atom: Rb87
    grid:
      n_points: [256]
      box_size: [30.0]
    interactions:
      c0: 10.0
      c1: -0.5

  ground_state:
    dt: 0.005
    n_steps: 10000
    tol: 1.0e-10
    initial_state: ferromagnetic
    zeeman: { p: 0.0, q: 0.1 }
    potential:
      type: harmonic
      omega: [1.0]

  sequence:
    - name: ramp_field
      duration: 5.0
      dt: 0.001
      save_every: 100
      zeeman:
        p: 0.0
        q: { from: 0.1, to: -0.5 }

    - name: free_expansion
      duration: 2.0
      dt: 0.0005
      save_every: 50
      zeeman: { p: 0.0, q: 0.0 }
      potential:
        type: none
```

```julia
using SpinorBEC
config = load_experiment("experiment.yaml")
result = run_experiment(config)
```

### Composite Potentials

Multiple potentials can be combined using YAML list syntax:

```yaml
potential:
  - type: harmonic
    omega: [1.0, 1.0, 1.0]
  - type: gravity
    g: 9.81
    axis: 3
```

## Available Potentials

| Type | YAML `type` | Parameters |
|------|-------------|------------|
| Harmonic trap | `harmonic` | `omega: [w_x, w_y, ...]` |
| No potential | `none` | (none) |
| Gravity | `gravity` | `g` (default 9.81), `axis` (default: last) |
| Crossed dipole | `crossed_dipole` | `polarizability`, `beams: [...]` |
| Composite | list syntax | automatically sums components |

## 機能詳細

### 物理モデル

スピンF Gross-Pitaevskii方程式（GPE）を平均場近似で解きます。

**ハミルトニアン:**

```
H = Σ_m ∫ ψ_m* [ -ℏ²∇²/(2M) + V(r) - p·m + q·m² + c0·n(r) + c1·⟨F⟩·F ] ψ_m d³r
```

| 項 | 説明 | 実装 |
|---|---|---|
| 運動エネルギー `-ℏ²∇²/(2M)` | 自由粒子の分散 | FFTでk空間に変換し `k²` を乗算 |
| 外部ポテンシャル `V(r)` | トラップ等の外場 | 実空間で対角演算 |
| 線形ゼーマン `-p·m` | 磁場による準位シフト | 成分ごとに定数シフト |
| 二次ゼーマン `q·m²` | 磁場の二次効果 | 成分ごとに定数シフト |
| 密度相互作用 `c0·n` | スピン非依存の接触相互作用 | 実空間で `c0·|ψ|²` を乗算 |
| スピン相互作用 `c1·⟨F⟩·F` | スピン依存の接触相互作用 | 各格子点で小行列の指数関数 |
| 双極子相互作用 (DDI) | 長距離異方的相互作用 | k空間でQテンソル畳み込み |

**対応次元:** 1D / 2D / 3D（`CartesianIndices`による統一コードパス）

### 相互作用パラメータ

スピン1 BECの場合:
```
c0 = 4πℏ²(a0 + 2a2) / (3M)    密度-密度結合
c1 = 4πℏ²(a2 - a0) / (3M)     スピン-スピン結合
```
- `c1 < 0`: 強磁性（Rb87）
- `c1 > 0`: 反強磁性（Na23）

準低次元系（1D, 2D）では横方向閉じ込めの面積/長さで除算して次元縮約を行います。

DDI結合定数: `C_dd = μ₀·μ² / (4π)`

### 対応原子種

| 原子 | スピンF | 散乱長 | 磁気モーメント | 特徴 |
|---|---|---|---|---|
| ⁸⁷Rb | 1 | a0=101.8 a_B, a2=100.4 a_B | — | 強磁性 (c1 < 0) |
| ²³Na | 1 | a0=50.0 a_B, a2=55.0 a_B | — | 反強磁性 (c1 > 0) |
| ¹⁵¹Eu | 6 | a_s=110.0 a_B | 7 μ_B | 双極子系 |

任意のスピンFに対してスピン行列を角運動量代数から構築可能（`StaticArrays`でスタック割り当て）。

### 数値手法

**Strang分割法（2次精度、対称分割）:**

```
1. 半ポテンシャルステップ (dt/2):
   a. 1/4 対角ポテンシャル（トラップ + ゼーマン + c0·密度）
   b. 1/2 スピン混合（c1 相互作用、行列指数関数）
   c. [DDI サブステップ（有効時）]
   d. 1/4 対角ポテンシャル（密度再計算）
2. 全運動エネルギーステップ (dt):
   FFT → exp(-ik²dt/2) 乗算 → IFFT
3. 半ポテンシャルステップ (dt/2):
   ステップ1の鏡像
```

**虚時間発展（基底状態探索）:**
- `exp(-iHdt)` → `exp(-Hdt)` に置換
- 各ステップ後に波動関数を再規格化
- エネルギー変化量が許容誤差以下で収束判定
- 初期状態: `:polar`（m=0のみ）、`:ferromagnetic`（m=+F のみ）、`:uniform`（均等）

**実時間ダイナミクス:**
- 多フェーズシーケンス（前フェーズの出力を次フェーズの入力に連鎖）
- 時間依存ゼーマンパラメータ（線形ランプ）
- コールバック関数による途中経過の取得

### ポテンシャル

| 種類 | YAML `type` | 数式 | パラメータ |
|---|---|---|---|
| なし | `none` | V = 0 | — |
| 調和トラップ | `harmonic` | V = ½Σ ω_d² x_d² | `omega: [ω_x, ...]` |
| 重力 | `gravity` | V = g·x[axis] | `g`（デフォルト9.81）, `axis` |
| 光双極子トラップ | `crossed_dipole` | V = -α·Σ I_beam | `polarizability`, `beams` |
| 複合 | リスト記法 | V = Σ V_i | 上記の組み合わせ |

光双極子トラップのガウシアンビームは横方向ガウシアンプロファイル `I(r) = I₀·exp(-2r⊥²/w₀²)` をモデル化（Rayleigh長の軸方向変化は省略）。

### 観測量

| 物理量 | 関数 | 説明 |
|---|---|---|
| 全密度 | `total_density(psi, ndim)` | Σ_m \|ψ_m\|² |
| 成分密度 | `component_density(psi, ndim, c)` | \|ψ_c\|² |
| 全ノルム | `total_norm(psi, grid)` | 全密度の空間積分 |
| 磁化 | `magnetization(psi, grid, sys)` | ∫ Σ_m m\|ψ_m\|² dV |
| スピン密度ベクトル | `spin_density_vector(psi, sm, ndim)` | 各空間点での (⟨Fx⟩, ⟨Fy⟩, ⟨Fz⟩) |
| 全エネルギー | `total_energy(ws)` | E_kin + E_trap + E_Zee + E_c0 + E_c1 + E_ddi |

### 入出力

**Julia API:**
```julia
# 基底状態探索
result = find_ground_state(; grid, atom, interactions, potential,
    dt=0.005, n_steps=5000, tol=1e-10, initial_state=:polar)

# 実時間発展
ws = make_workspace(; grid, atom, interactions, ...)
result = run_simulation!(ws; callback=nothing)
```

**YAML実験設定:** 多フェーズシミュレーションを宣言的に定義（上記 Quick Start 参照）

**状態保存:** JLD2形式で波動関数・時刻・グリッドパラメータを保存/読み込み（`save_state` / `load_state`）

### 可視化（弱依存拡張）

| 拡張 | 機能 |
|---|---|
| PlotlyJS | `plot_density`（1D/2D密度）、`plot_spinor`（成分占有数）、`plot_spin_texture`（スピンテクスチャ）、`animate_dynamics`（アニメーション） |
| Makie | 3Dサーフェスプロット、ボリュームレンダリング、リアルタイムアニメーション、インタラクティブスライダー |

### 単位系

`Units`サブモジュールでSI定数（ℏ, AMU, ボーア半径, ボーア磁子, μ₀, k_B）を定義。`DimensionlessScales`で調和振動子基準の無次元化/次元化変換を提供。

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed documentation of the internal design, data flow, and numerical methods.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
