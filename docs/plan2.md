了解です。それなら設計の優先順位が根本的に変わります。BEC物理の正しさを最優先にするとき、最も重要な洞察は一つです。

**テストを書かないと何が壊れているかわからない。コードを書く前にテストを書く。**

物理シミュレーションのテストは通常のソフトウェアテストと性質が違います。「正しい出力」が解析解として事前に存在するからです。その構造から考えます。

---

## 物理的正しさの検証ピラミッドLevel 0から順番に書き、上が通らないうちは上に進まない。これが鉄則です。

---

## Level 0: 各伝播子の単体テスト

各伝播子を「他から切り離して」テストします。これが最も発見効率が高い。

### 0a. 運動伝播子 — 自由ガウシアン拡散

相互作用なし・ポテンシャルなしの1Dでは、ガウシアン波束の幅が時間とともに広がる厳密解があります。

```
σ(t)² = σ₀² + (ℏt / m σ₀)²
```

```julia
@testset "kinetic propagator: free Gaussian spreading" begin
    # 単位系: ℏ=1, m=1, ω=1 → a_ho=1
    σ₀ = 1.0      # 初期ウェスト（グリッド単位）
    T  = 5.0      # 伝播時間
    dt = 0.001

    gc = GridConfig{1}(n_points=(512,), box_size=(40.0,))
    grid = Grid(gc)
    sys  = SpinSystem(0)              # スカラーBEC（スピン0）

    psi = zeros(ComplexF64, 512, 1)
    @. psi[:,1] = exp(-grid.x[1]^2 / (2σ₀^2)) |> normalize

    sp  = SimParams(dt=dt, n_steps=round(Int, T/dt), imaginary_time=false)
    ws  = make_workspace(grid, sys, ..., psi_init=psi, sim_params=sp)

    # 自由伝播（ポテンシャルなし、相互作用なし）
    run_simulation!(ws)

    n = total_density(ws.psi, 1)[:, 1]
    # 数値的なσを計算: σ² = ∫x²n dx / N - (∫x n dx / N)²
    N   = sum(n) * grid.dx[1]
    x̄   = sum(grid.x[1] .* n) * grid.dx[1] / N
    σ²  = sum((grid.x[1] .- x̄).^2 .* n) * grid.dx[1] / N

    σ_exact = sqrt(σ₀^2 + (T/σ₀)^2)   # ℏ=m=1
    @test abs(sqrt(σ²) - σ_exact) / σ_exact < 1e-4
end
```

このテストが通らないなら運動伝播子かFFTの正規化に問題があります。

### 0b. ゼーマン伝播子 — Larmor歳差

線形ゼーマン効果 p のみ存在するとき、空間的に一様なスピン-1凝縮体は角速度 p/ℏ で歳差運動します。

初期状態 `ψ = (1, 0, 0)^T`（m=+1 状態）に対し：

```
⟨Fz⟩ = 1  （時間不変）
⟨Fx⟩(t) + i⟨Fy⟩(t) = exp(−ipt/ℏ) × (⟨Fx⟩₀ + i⟨Fy⟩₀)
```

しかしm=+1状態は ⟨Fx⟩=⟨Fy⟩=0 なのでこれでは面白くありません。より良いテスト初期状態：

```julia
# 「赤道上」のスピンコヒーレント状態 |θ=π/2, φ=0⟩
# スピン-1の場合: ζ = (1/2, 1/√2, 1/2)^T
ζ_init = [0.5, 1/√2, 0.5] |> normalize

@testset "Zeeman propagator: Larmor precession" begin
    p = 0.5     # 線形ゼーマン係数
    T = 2π/p   # 一周期

    # ... ワークスペースを作る（相互作用なし、ポテンシャルなし）...

    times, Fx, Fy, Fz = [], [], [], []
    for t in 0.0 : dt : T
        push!(times, t)
        Fv = spin_density_integrated(ws)   # ⟨F⟩の体積積分
        push!(Fx, Fv[1]); push!(Fy, Fv[2]); push!(Fz, Fv[3])
        split_step!(ws, ZeemanOnly=true)
    end

    # ⟨Fz⟩は保存されるべき
    @test std(Fz) < 1e-10

    # ⟨Fx⟩は cos(pt) で振動
    @test maximum(abs.(Fx .- cos.(p .* times) .* Fx[1])) < 1e-6
end
```

### 0c. スピン混合伝播子 — 均質BECの固有状態

c₁ > 0（Na23: 反強磁性）の場合、極性状態 `ψ = (0, 1, 0)^T`（m=0のみ）は `c₁(F·F)` ハミルトニアンの固有状態です。従って時間発展しても密度分布が変わらないはずです。

```julia
@testset "spin mixing: polar state is eigenstate (antiferromagnetic)" begin
    # Na23パラメータ: c1 > 0
    ψ_polar = zeros(ComplexF64, N_grid, 3)
    ψ_polar[:, 2] .= sqrt(n₀)   # m=0成分のみ

    n_before = component_density(ψ_polar, 1, 2)
    apply_spin_mixing!(ws, dt=1.0)
    n_after  = component_density(ws.psi, 1, 2)

    @test maximum(abs.(n_after .- n_before)) < 1e-14
end
```

逆に `c₁ < 0`（Rb87: 強磁性）の場合、強磁性状態 `ψ = (1, 0, 0)^T` が固有状態になります。

---

## Level 1: 保存則テスト

これが「通っているはずなのに通らない」と最も危険なケースです。

```julia
@testset "conservation laws: real-time evolution" begin
    ws = make_workspace(
        grid  = Grid(GridConfig{2}(n_points=(64,64), box_size=(20.0,20.0))),
        atom  = Rb87,
        potential = HarmonicTrap{2}(omega=(1.0, 1.0)),
        interactions = InteractionParams(c0=50.0, c1=-0.5),
        zeeman = ZeemanParams(p=0.0, q=0.1),
        sim_params = SimParams(dt=0.002, n_steps=5000)
    )

    N₀ = total_norm(ws.psi, ws.grid)
    E₀ = total_energy(ws)
    M₀ = magnetization(ws.psi, ws.grid, ws.system)

    run_simulation!(ws)

    N₁ = total_norm(ws.psi, ws.grid)
    E₁ = total_energy(ws)
    M₁ = magnetization(ws.psi, ws.grid, ws.system)

    # 規格化保存: 機械精度（10^-14 程度）
    @test abs(N₁ - N₀) / N₀ < 1e-12

    # エネルギー保存: Strang分割はO(dt²)なので長時間で少しずつ誤差が溜まる
    # 5000ステップ後に0.1%以内であれば合格
    @test abs(E₁ - E₀) / abs(E₀) < 0.001

    # 磁化保存: p=0のとき磁化は対称性から保存される
    @test abs(M₁ - M₀) < 1e-10
end
```

このテストで最も注意すべき点を以下に示します。

**規格化保存が 1e-12 以下にならない場合の原因一覧：**

| 症状             | 原因                                     | 修正                           |
| ---------------- | ---------------------------------------- | ------------------------------ |
| ドリフトが O(dt) | 実時間伝播でも再規格化している           | 虚時間と実時間の判定分岐を確認 |
| 急に発散         | スピン混合の行列指数関数がユニタリでない | `_exp_i_hermitian` の検証      |
| ゆっくりドリフト | FFTの前後で規格化係数が合っていない      | Parsevalの定理を確認           |

---

## Level 2: Thomas-Fermi との比較

スカラーBECの最も重要なベンチマーク。相互作用が強い極限（TF極限）では密度プロファイルが解析的に求まります。

### 1D調和トラップでの化学ポテンシャル

```
μ_TF = ℏω/2 × (3Ng/√2 / (ℏω a_ho))^(2/3)
```

ここで g = c₀ (1D)、a_ho = √(ℏ/mω)。無次元化（ℏ=m=ω=1）では：

```
μ_TF = (3N g/√2)^(2/3) / 2
```

密度プロファイル：

```
n_TF(x) = max(0, (μ_TF - x²/2) / g)
```

```julia
@testset "ground state: Thomas-Fermi profile (1D, scalar)" begin
    g  = 200.0   # c₀: TF極限に入るのに十分大きい
    N  = 1.0     # 規格化（∫|ψ|²dx = 1）

    ws = make_workspace(
        grid = Grid(GridConfig{1}(n_points=(512,), box_size=(30.0,))),
        atom = ScalarAtom(mass=1.0),
        potential = HarmonicTrap{1}(omega=(1.0,)),
        interactions = InteractionParams(c0=g),
        sim_params = SimParams(dt=0.01, n_steps=5000, imaginary_time=true)
    )
    result = find_ground_state(ws, tol=1e-10)

    μ_TF = (3N*g / sqrt(2))^(2/3) / 2
    x    = ws.grid.x[1]
    n_TF = max.(0.0, (μ_TF .- x.^2 ./ 2) ./ g)
    n_TF ./= sum(n_TF) * ws.grid.dx[1]   # 規格化

    n_num = total_density(result.workspace.psi, 1)[:,1]

    # L1ノルムで5%以内
    err = sum(abs.(n_num .- n_TF)) * ws.grid.dx[1]
    @test err < 0.05

    # 化学ポテンシャルの比較
    @test abs(result.energy - μ_TF) / μ_TF < 0.02
end
```

---

## Level 3: スピン動力学の解析解

スピン-1の最も重要な動力学ベンチマークは「スピン混合振動」です。

### 均質BECでのスピン混合振動

一様密度 n₀ のスピン-1 BEC を考えます。初期状態が m=±1 に等確率で分布しているとき、スピン混合により m=0 への遷移が起きます。単一モード近似（SMA）での厳密解は楕円積分で表されますが、小振幅極限では単純です：

初期状態 `ψ ≈ (1/√2, ε, 1/√2)^T × √n₀`（ε ≪ 1）に対し、m=0 成分の密度は：

```
n₀(t) ≈ |ε|² + |ε|² [cos(Ω_SM t) - 1]
```

ここで `Ω_SM = 2|c₁| n₀ / ℏ` がスピン混合振動周波数です。

```julia
@testset "spin dynamics: spin mixing oscillations (homogeneous, small amplitude)" begin
    n₀  = 1.0
    c₁  = -0.5     # 強磁性 (Rb87)
    ε   = 0.05     # 小振幅

    # 1D均質系（ポテンシャルなし、周期的境界条件）
    ψ_init = zeros(ComplexF64, 64, 3)
    ψ_init[:, 1] .= sqrt(n₀/2)
    ψ_init[:, 2] .= ε * sqrt(n₀)
    ψ_init[:, 3] .= sqrt(n₀/2)
    # 規格化

    Ω_SM  = 2 * abs(c₁) * n₀
    T_SM  = 2π / Ω_SM    # 一周期

    # T_SM の2周期ぶん実時間発展
    n_m0_series = []
    times = 0.0:dt:(2T_SM)
    for t in times
        push!(n_m0_series, mean(component_density(ws.psi, 1, 2)))
        split_step!(ws)
    end

    # 振動周波数の確認（FFTで周波数を取得）
    freq_numerical = dominant_frequency(n_m0_series, dt)
    @test abs(freq_numerical - Ω_SM / (2π)) / (Ω_SM / (2π)) < 0.02
end
```

---

## 現在のコードで最も深刻な問題: DDI の k=0 特異点

アーキテクチャに書かれているDDIカーネルを見ると：

```
Q_ab(k) = k_a k_b / k² - δ_ab / 3
```

k=0 では `k_a k_b / k²` が 0/0 不定形です。これをどう処理するかがコードに書いていないのが気になります。

**物理的に正しい値は 0 です。** 理由：等方的な系ではDDI効果が平均的にゼロになる。これはDDIカーネルのトレースが

```
∑_a Q_aa(k) = 1 - 3×(1/3) = 0
```

から来ます。k=0 での値は系の等方性から `Q_ab(0) = 0` が唯一の正しい選択です。

修正コード：

```julia
function build_ddi_kernel(grid::Grid{N}, params::DDIParams{N}) where N
    Q = zeros(Float64, size(grid.k_squared)..., 3, 3)  # 6成分だが概念的に3×3

    for idx in CartesianIndices(grid.k_squared)
        k2 = grid.k_squared[idx]

        if k2 < 1e-30   # k ≈ 0 の場合
            # Q_ab(0) = 0  （等方性の要請）
            continue      # zerosで初期化済みなのでskip
        end

        kx = grid.k[1][idx[1]]
        ky = (N >= 2) ? grid.k[2][idx[2]] : 0.0
        kz = (N >= 3) ? grid.k[3][idx[3]] : 0.0
        kvec = (kx, ky, kz)

        for a in 1:3, b in 1:3
            Q[idx, a, b] = kvec[a] * kvec[b] / k2 - (a == b ? 1/3 : 0.0)
        end
    end
    return Q
end
```

この修正なしでは虚時間伝播が発散する危険性があります。

---

## 実装すべきテストファイルの全体構造

```
test/
  runtests.jl

  # Level 0: 単体テスト
  test_propagators/
    test_kinetic.jl       ← 自由ガウシアン拡散
    test_potential.jl     ← 調和振動子固有状態
    test_zeeman.jl        ← Larmor歳差
    test_spin_mixing.jl   ← 固有状態の安定性
    test_ddi.jl           ← k=0処理、トレース条件

  # Level 1: 保存則
  test_conservation/
    test_norm.jl          ← 機械精度での規格化
    test_energy.jl        ← エネルギー保存 (dt依存性)
    test_magnetization.jl ← 磁化保存

  # Level 2: 基底状態
  test_ground_state/
    test_scalar_1d.jl     ← 1D Thomas-Fermi
    test_scalar_2d.jl     ← 2D Thomas-Fermi
    test_spinor_rb87.jl   ← Rb87基底状態（強磁性相）
    test_spinor_na23.jl   ← Na23基底状態（反強磁性相）

  # Level 3: 動力学
  test_dynamics/
    test_free_expansion.jl   ← 解放後の自由膨張
    test_spin_mixing.jl      ← スピン混合振動周波数
    test_larmor.jl           ← Larmor歳差（全系）
    test_breathing_mode.jl   ← 呼吸振動モード
```

この構造で `runtests.jl` を走らせたとき、全テストが通っている状態を「BEC物理が正しい」の定義にします。

---

## 開始順序の提案

今すぐ始めるなら、この順番が最も効率的です。

1. まず `test_kinetic.jl` と `test_zeeman.jl` だけ書く（1時間）
2. 走らせて通るか確認
3. 通ったら `test_conservation/test_norm.jl` を書く
4. 通ったら DDI の k=0 を修正して `test_ddi.jl` を書く
5. 全 Level 0-1 が通ったら Thomas-Fermi 比較に進む

どのテストを最初に実装するか、あるいはDDIの k=0 修正から始めるか、どちらに進みますか？
