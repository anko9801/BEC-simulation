# SpinorBEC.jl

スピンF ボース・アインシュタイン凝縮体（BEC）のシミュレーションパッケージ。分割ステップフーリエ法により、1D/2D/3D のスピノルGross-Pitaevskii方程式を解く。

## 機能一覧

- **任意スピンF**: 角運動量代数からスピン行列を構築（`StaticArrays`でスタック割り当て）
- **N次元汎用**: 1D / 2D / 3D を `CartesianIndices` による単一コードパスで処理
- **分割ステップフーリエ法**: Strang分割＋ポテンシャル・スピン混合・DDI・Ramanの入れ子サブステップ
- **基底状態探索**: 虚時間発展＋収束判定
- **実時間ダイナミクス**: 多フェーズシーケンス、時間依存ゼーマンランプ、適応時間刻み
- **ポテンシャル**: 調和トラップ、重力、光双極子トラップ（ガウシアンビーム）、レーザービームポテンシャル、複合
- **DDI**: 双極子-双極子相互作用（k空間Qテンソル畳み込み）
- **Raman結合**: 二光子Raman遷移（空間依存、行列指数関数）
- **ガウシアンビーム光学**: 複素ビームパラメータq、ABCD行列伝搬、モード結合
- **Thomas-Fermi初期化**: 化学ポテンシャル二分探索による密度プロファイル
- **トポロジカル観測量**: Berry曲率、超流動渦度、スカーミオン電荷、Majorana星表現
- **YAML実験設定**: 宣言的な再現可能マルチフェーズシミュレーション
- **Unitful対応**: 物理単位付き量の直接入力
- **定義済み原子種**: Rb87（F=1, 強磁性）、Na23（F=1, 反強磁性）、Eu151（F=6, 双極子系）

## インストール

```julia
using Pkg
Pkg.develop(path="path/to/BEC-simulation")
```

## クイックスタート

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

### YAML実験設定

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

## 物理モデル

### ハミルトニアン

```
H = Σ_m ∫ ψ_m* [ -ℏ²∇²/(2M) + V(r) - p·m + q·m² + c0·n(r) + c1·⟨F⟩·F + H_ddi + H_Raman ] ψ_m d³r
```

| 項 | 説明 | 実装 |
|---|---|---|
| 運動エネルギー `-ℏ²∇²/(2M)` | 自由粒子の分散 | FFTでk空間に変換し `k²` を乗算 |
| 外部ポテンシャル `V(r)` | トラップ等の外場 | 実空間で対角演算 |
| 線形ゼーマン `-p·m` | 磁場による準位シフト | 成分ごとに定数シフト |
| 二次ゼーマン `q·m²` | 磁場の二次効果 | 成分ごとに定数シフト |
| 密度相互作用 `c0·n` | スピン非依存の接触相互作用 | 実空間で `c0·\|ψ\|²` を乗算 |
| スピン相互作用 `c1·⟨F⟩·F` | スピン依存の接触相互作用 | 各格子点で小行列の指数関数 |
| DDI | 長距離異方的双極子相互作用 | k空間で `Q_αβ(k) = k̂_αk̂_β - δ_αβ/3` 畳み込み |
| Raman結合 | 二光子遷移 `(Ω_R/2)(e^{ik·r}F₊ + h.c.) + δF_z` | 空間依存行列指数関数 |

### 相互作用パラメータ

スピン1 BECの場合:

```
c0 = 4πℏ²(a0 + 2a2) / (3M)    密度-密度結合
c1 = 4πℏ²(a2 - a0) / (3M)     スピン-スピン結合
```

- `c1 < 0`: 強磁性（Rb87）
- `c1 > 0`: 反強磁性（Na23）

準低次元系（1D, 2D）では横方向閉じ込めで次元縮約。DDI結合定数: `C_dd = μ₀·μ² / (4π)`

### 対応原子種

| 原子 | スピンF | 散乱長 | 磁気モーメント | 特徴 |
|---|---|---|---|---|
| ⁸⁷Rb | 1 | a0=101.8 a_B, a2=100.4 a_B | — | 強磁性 (c1 < 0) |
| ²³Na | 1 | a0=50.0 a_B, a2=55.0 a_B | — | 反強磁性 (c1 > 0) |
| ¹⁵¹Eu | 6 | a_s=110.0 a_B | 7 μ_B | 双極子系 (ε_dd ≈ 0.55) |

任意のスピンFに対してスピン行列を角運動量代数から構築可能。

## 数値手法

### Strang分割法（2次精度、対称分割）

```
1. 半ポテンシャルステップ (dt/2):
   a. 1/4 対角ポテンシャル（トラップ + ゼーマン + c0·密度）
   b. 1/2 スピン混合（c1 相互作用、行列指数関数）
   c. [DDI サブステップ（有効時）]
   d. [Raman サブステップ（有効時）]
   e. 1/4 対角ポテンシャル（密度再計算）
2. 全運動エネルギーステップ (dt):
   FFT → exp(-ik²dt/2) 乗算 → IFFT
3. 半ポテンシャルステップ (dt/2):
   ステップ1の鏡像
```

スピン混合は `c1 ≈ 0` の場合（例: Eu151）自動スキップ。スピン1ではRodrigues公式、高スピンでは固有値分解による行列指数関数。

### 虚時間発展（基底状態探索）

- `exp(-iHdt)` → `exp(-Hdt)` に置換
- 各ステップ後に波動関数を再規格化
- エネルギー変化量が許容誤差以下で収束判定
- 初期状態: `:polar`（m=0）、`:ferromagnetic`（m=+F）、`:uniform`（均等）
- Thomas-Fermi初期化: `init_psi_thomas_fermi` で化学ポテンシャルから密度プロファイルを生成

### 実時間ダイナミクス

- 多フェーズシーケンス（前フェーズの出力を次フェーズの入力に連鎖）
- `TimeDependentZeeman` による時間依存ゼーマンパラメータ（線形ランプ）
- コールバック関数による途中経過の取得
- 適応時間刻み対応

## ポテンシャル

| 種類 | YAML `type` | 数式 | パラメータ |
|---|---|---|---|
| なし | `none` | V = 0 | — |
| 調和トラップ | `harmonic` | V = ½Σ ω_d² x_d² | `omega: [ω_x, ...]` |
| 重力 | `gravity` | V = g·x[axis] | `g`（デフォルト9.81）, `axis` |
| 光双極子トラップ | `crossed_dipole` | V = -α·Σ I_beam | `polarizability`, `beams` |
| レーザービーム | — | ガウシアンビーム強度分布 | `LaserBeamPotential` |
| 複合 | リスト記法 | V = Σ V_i | 上記の組み合わせ |

複合ポテンシャルはYAMLのリスト記法で定義:

```yaml
potential:
  - type: harmonic
    omega: [1.0, 1.0, 1.0]
  - type: gravity
    g: 9.81
    axis: 3
```

## ガウシアンビーム光学

`OpticalBeam` は複素ビームパラメータ q による正確なガウシアンビーム伝搬を実装。

```julia
beam = OpticalBeam(wavelength=1064e-9, power=1.0, waist=50e-6)
propagated = propagate(beam, abcd_free_space(0.1))

w = waist_radius(beam)          # ビームウエスト半径
zR = rayleigh_length(beam)      # レイリー長
I0 = peak_intensity(beam)       # ピーク強度
```

ABCD行列: `abcd_free_space`, `abcd_thin_lens`, `abcd_curved_mirror`, `abcd_flat_mirror`

ファイバー結合効率: `mode_overlap`, `fiber_coupling`

Unitful対応: `OpticalBeam(wavelength=1064u"nm", power=1u"W", waist=50u"μm")`

## 観測量

### 基本量

| 物理量 | 関数 | 説明 |
|---|---|---|
| 全密度 | `total_density(psi, ndim)` | Σ_m \|ψ_m\|² |
| 成分密度 | `component_density(psi, ndim, c)` | \|ψ_c\|² |
| 全ノルム | `total_norm(psi, grid)` | ∫n dV |
| 磁化 | `magnetization(psi, grid, sys)` | ∫ Σ_m m\|ψ_m\|² dV |
| スピン密度ベクトル | `spin_density_vector(psi, sm, ndim)` | 各点の (⟨Fx⟩, ⟨Fy⟩, ⟨Fz⟩) |
| 全エネルギー | `total_energy(ws)` | E_kin + E_trap + E_Zee + E_c0 + E_c1 + E_ddi |
| 成分占有率 | `component_populations(psi, grid, sys)` | 各スピン成分の規格化占有率 |

### 流体力学量

| 物理量 | 関数 | 説明 |
|---|---|---|
| 確率流密度 | `probability_current(psi, grid, plans)` | j(r) = Σ_c Im(ψ_c* ∇ψ_c) |
| 超流動速度 | `superfluid_velocity(psi, grid, plans)` | v = j / n |
| 軌道角運動量 | `orbital_angular_momentum(psi, grid, plans)` | ⟨L_z⟩ = ∫ ψ*(-i)(x∂_y - y∂_x)ψ dV |
| 全角運動量 | `total_angular_momentum(psi, grid, plans, sys)` | J_z = L_z + S_z |
| 超流動渦度 | `superfluid_vorticity(psi, grid, plans)` | ω = ∇ × v_s（2D: スカラー、3D: ベクトル） |

### トポロジカル量

| 物理量 | 関数 | 説明 |
|---|---|---|
| Berry曲率 | `berry_curvature(psi, grid, plans, sm)` | Mermin-Ho関係: Ω = ŝ·(∂_iŝ × ∂_jŝ) |
| スカーミオン電荷 | `spin_texture_charge(psi, grid, plans, sm)` | Q = (1/4πF) ∫ Ω d²r（2Dのみ） |
| Majorana星 | `majorana_stars(spinor, F)` | Majorana多項式の根（2F個の星） |
| 正二十面体秩序 | `icosahedral_order_parameter(psi, sm, ndim)` | Steinhardt Q₆ 結合秩序パラメータ（F≥6） |

## 診断ツール

| 関数 | 説明 |
|---|---|
| `spin_mixing_period(c1, q)` | スピン混合振動周期（無次元） |
| `spin_mixing_period_si(c1, q)` | スピン混合振動周期（SI単位） |
| `quadratic_zeeman_from_field(g_F, B, ΔE_hf)` | 磁場からの二次ゼーマンシフト |
| `healing_length_contact(m, c0, n)` | 接触相互作用のヒーリング長 |
| `healing_length_spin(m, c1, n)` | スピン相互作用のヒーリング長 |
| `healing_length_ddi(m, C_dd, n)` | DDIのヒーリング長 |
| `thomas_fermi_radius(density, x)` | 密度プロファイルからTFR抽出 |
| `phase_diagram_point(...)` | 相図上の座標 (R_TF/ξ_sp, R_TF/ξ_dd) |

## 入出力

```julia
# 基底状態探索
result = find_ground_state(; grid, atom, interactions, potential,
    dt=0.005, n_steps=5000, tol=1e-10, initial_state=:polar)

# 実時間発展
ws = make_workspace(; grid, atom, interactions, ...)
result = run_simulation!(ws; callback=nothing)

# 状態保存/読み込み（JLD2形式）
save_state("checkpoint.jld2", ws)
state = load_state("checkpoint.jld2")
```

## 単位系

`Units` サブモジュールでSI定数を定義:

| 定数 | 記号 |
|---|---|
| ディラック定数 | `Units.HBAR` |
| 原子質量単位 | `Units.AMU` |
| ボーア半径 | `Units.A_BOHR` |
| ボーア磁子 | `Units.MU_BOHR` |
| 真空透磁率 | `Units.MU_0` |
| ボルツマン定数 | `Units.K_B` |

`DimensionlessScales` で調和振動子基準の無次元化/次元化変換を提供。

Unitful.jl による物理単位付き量の直接入力にも対応。

## 可視化（弱依存拡張）

| 拡張 | 機能 |
|---|---|
| PlotlyJS | `plot_density`, `plot_spinor`, `plot_spin_texture`, `animate_dynamics` |
| Makie | 3Dサーフェス、ボリュームレンダリング、リアルタイムアニメーション |

## アーキテクチャ

詳細は [docs/architecture.md](docs/architecture.md) を参照。

## テスト

```bash
julia --project=. -e 'using Pkg; Pkg.test()'  # 全4734テスト
```
