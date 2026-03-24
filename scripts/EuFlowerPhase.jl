"""
    EuFlowerPhase.jl

F=6 Europium BEC のFlower相シミュレーション（無次元化版）

# 物理的背景
- 同位体: ¹⁵¹Eu (I=5/2, F=6 超微細状態)
- 電子状態: ⁸S₇/₂ (純粋スピン系)
- 磁気モーメント: μ ≈ 7μ_B
- 散乱長: a_s = 135 a_B (実験値)
- 双極子パラメータ: ε_dd = 0.44

# 単位系
- 長さ: a_ho = √(ℏ/Mω_⊥)
- エネルギー: ℏω_⊥
- 時間: 1/ω_⊥

# 参考文献
- Kawaguchi & Ueda, Phys. Rep. 520, 253 (2012)
"""
module EuFlowerPhase

using FFTW
using LinearAlgebra

export Params, Grid, Spinor
export init_ferromagnetic!, init_flower!, init_random_perturbation!
export evolve_imag!, evolve_real!
export density, spin_density, angular_momentum, magnetization
export energy, energy_components, component_populations
export m_to_idx, idx_to_m

# ============================================================
# 定数とパラメータ
# ============================================================

"""スピン量子数"""
const SPIN_F = 6
const NUM_COMPONENTS = 2 * SPIN_F + 1  # = 13

"""
    Params

シミュレーションパラメータ（無次元化）

# フィールド
- `N`: 原子数
- `g0`: 密度相互作用 (c₀ に対応)
- `g1`: スピン相互作用 (c₁ に対応、負で強磁性)
- `gdd`: 双極子相互作用強度
- `λz`: トラップアスペクト比 ω_z/ω_⊥
- `ε_dd`: 双極子パラメータ a_dd/a_s (参考値)
"""
struct Params
    N::Float64      # 原子数
    g0::Float64     # 密度相互作用 (4πa_s/a_ho * N)
    g1::Float64     # スピン相互作用 (負で強磁性)
    gdd::Float64    # 双極子相互作用
    λz::Float64     # アスペクト比

    function Params(; N=1.5e4, g0=nothing, g1=0.0, gdd=nothing, 
                    λz=0.5, a_s=135.0, ε_dd=0.44, a_ho_nm=850.0)
        # a_s in units of Bohr radius, a_ho_nm in nm
        a_B_nm = 0.0529  # Bohr radius in nm
        
        if g0 === nothing
            # g0 = 4π * a_s / a_ho * N
            g0 = 4π * (a_s * a_B_nm / a_ho_nm) * N
        end
        
        if gdd === nothing
            # gdd = ε_dd * g0
            gdd = ε_dd * g0
        end
        
        new(N, g0, g1, gdd, λz)
    end
end

"""
    Grid

空間グリッド（3D）

# フィールド
- `Nx, Ny, Nz`: グリッド点数
- `Lx, Ly, Lz`: ボックスサイズ（a_ho単位）
- `dx, dy, dz`: グリッド間隔
- `x, y, z`: 座標配列
- `kx, ky, kz`: 波数配列
- `k2`: |k|²
"""
struct Grid
    Nx::Int
    Ny::Int
    Nz::Int
    Lx::Float64
    Ly::Float64
    Lz::Float64
    dx::Float64
    dy::Float64
    dz::Float64
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
    k2::Array{Float64,3}
    dV::Float64

    function Grid(Nx::Int, Ny::Int, Nz::Int, Lx::Float64; 
                  Ly::Float64=Lx, Lz::Float64=Lx/2)
        dx = Lx / Nx
        dy = Ly / Ny
        dz = Lz / Nz

        x = range(-Lx/2, Lx/2 - dx, length=Nx) |> collect
        y = range(-Ly/2, Ly/2 - dy, length=Ny) |> collect
        z = range(-Lz/2, Lz/2 - dz, length=Nz) |> collect

        kx = fftfreq(Nx, 2π/dx) |> collect
        ky = fftfreq(Ny, 2π/dy) |> collect
        kz = fftfreq(Nz, 2π/dz) |> collect

        k2 = zeros(Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            k2[i,j,k] = kx[i]^2 + ky[j]^2 + kz[k]^2
        end

        dV = dx * dy * dz
        new(Nx, Ny, Nz, Lx, Ly, Lz, dx, dy, dz, x, y, z, kx, ky, kz, k2, dV)
    end
end

# 簡易コンストラクタ
Grid(N::Int, L::Float64) = Grid(N, N, N÷2, L)

"""
    Spinor

F=6 スピノル波動関数（13成分）

# フィールド
- `ψ`: 波動関数配列 [Nx, Ny, Nz, 13]
- `ψk`: フーリエ変換用バッファ
"""
mutable struct Spinor
    ψ::Array{ComplexF64,4}   # [Nx, Ny, Nz, 13]
    ψk::Array{ComplexF64,4}  # FFT buffer

    function Spinor(g::Grid)
        ψ = zeros(ComplexF64, g.Nx, g.Ny, g.Nz, NUM_COMPONENTS)
        ψk = zeros(ComplexF64, g.Nx, g.Ny, g.Nz, NUM_COMPONENTS)
        new(ψ, ψk)
    end
end

# ============================================================
# スピン行列 (F=6, 13×13)
# ============================================================

"""
    spin_matrices()

F=6 のスピン行列 (f_x, f_y, f_z) を返す
インデックス: m = +6, +5, ..., -6 → idx = 1, 2, ..., 13
"""
function spin_matrices()
    f = SPIN_F
    dim = NUM_COMPONENTS
    
    fz = zeros(dim, dim)
    fp = zeros(dim, dim)  # f_+
    fm = zeros(dim, dim)  # f_-
    
    for idx in 1:dim
        m = f - (idx - 1)  # m = 6, 5, 4, ..., -6
        fz[idx, idx] = m
        
        # f_+ |m⟩ = √(f(f+1) - m(m+1)) |m+1⟩
        if idx > 1
            m_next = m + 1
            fp[idx-1, idx] = sqrt(f*(f+1) - m*m_next)
        end
        
        # f_- |m⟩ = √(f(f+1) - m(m-1)) |m-1⟩
        if idx < dim
            m_prev = m - 1
            fm[idx+1, idx] = sqrt(f*(f+1) - m*m_prev)
        end
    end
    
    fx = (fp + fm) / 2
    fy = (fp - fm) / (2im)
    
    return fx, fy, fz
end

# グローバルに一度だけ計算
const FX, FY, FZ = spin_matrices()

"""m値からインデックスを取得"""
m_to_idx(m::Int) = SPIN_F - m + 1

"""インデックスからm値を取得"""
idx_to_m(idx::Int) = SPIN_F - (idx - 1)

# ============================================================
# 初期化関数
# ============================================================

"""
    init_ferromagnetic!(ψ::Spinor, p::Params, g::Grid)

強磁性初期状態 (m = +6 に完全偏極)
Thomas-Fermi分布を使用
"""
function init_ferromagnetic!(ψ::Spinor, p::Params, g::Grid)
    ψ.ψ .= 0
    
    # Thomas-Fermi 化学ポテンシャルの推定
    # μ ≈ (15 N g0 / (16π))^(2/5) * (ω̄)^(2/5) (球対称近似)
    ω_bar = (p.λz)^(1/3)  # geometric mean ratio
    μ_TF = 0.5 * (15 * p.g0 / (4π))^0.4
    
    idx_p6 = m_to_idx(6)  # m = +6 のインデックス
    
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        r2 = g.x[i]^2 + g.y[j]^2 + (p.λz * g.z[k])^2
        V_trap = 0.5 * r2
        
        n_local = max(0.0, μ_TF - V_trap) / p.g0
        ψ.ψ[i,j,k,idx_p6] = sqrt(n_local)
    end
    
    normalize!(ψ, p, g)
end

"""
    init_flower!(ψ::Spinor, p::Params, g::Grid; Jz::Int=6, seed_amp::Float64=0.1)

Flower相アンザッツで初期化
ψ_m(r,φ,z) = exp(i(Jz-m)φ) η_m(r,z)
"""
function init_flower!(ψ::Spinor, p::Params, g::Grid; Jz::Int=SPIN_F, seed_amp::Float64=0.1)
    # まず強磁性状態で初期化
    init_ferromagnetic!(ψ, p, g)
    
    # 各成分に位相巻き数を付与
    for m in -SPIN_F:SPIN_F
        idx = m_to_idx(m)
        winding = Jz - m
        
        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            φ = atan(g.y[j], g.x[i])
            r = sqrt(g.x[i]^2 + g.y[j]^2)
            
            if winding != 0
                # 渦の核を滑らかにするファクター
                r_core = 1.0  # 渦芯サイズ（a_ho単位）
                core_factor = r^abs(winding) / (r^abs(winding) + r_core^abs(winding))
                
                # m ≠ +6 成分には小さな種を入れる
                if m != SPIN_F
                    ψ.ψ[i,j,k,idx] = seed_amp * sqrt(abs(ψ.ψ[i,j,k,m_to_idx(SPIN_F)])^2) * 
                                     core_factor * exp(im * winding * φ)
                else
                    ψ.ψ[i,j,k,idx] *= exp(im * winding * φ)
                end
            end
        end
    end
    
    normalize!(ψ, p, g)
end

"""
    init_random_perturbation!(ψ::Spinor, p::Params, g::Grid; amp::Float64=0.01)

強磁性状態にランダムな摂動を加える
対称性を破ってFlower相への緩和を促す
"""
function init_random_perturbation!(ψ::Spinor, p::Params, g::Grid; amp::Float64=0.01)
    init_ferromagnetic!(ψ, p, g)
    
    # m ≠ +6 成分に小さなノイズを加える
    for m in -SPIN_F:(SPIN_F-1)
        idx = m_to_idx(m)
        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            n_local = abs2(ψ.ψ[i,j,k,m_to_idx(SPIN_F)])
            if n_local > 1e-10
                ψ.ψ[i,j,k,idx] = amp * sqrt(n_local) * (randn() + im*randn()) / sqrt(2)
            end
        end
    end
    
    normalize!(ψ, p, g)
end

"""規格化"""
function normalize!(ψ::Spinor, p::Params, g::Grid)
    norm2 = 0.0
    for idx in 1:NUM_COMPONENTS
        norm2 += sum(abs2.(ψ.ψ[:,:,:,idx])) * g.dV
    end
    ψ.ψ .*= sqrt(p.N / norm2)
end

# ============================================================
# 物理量計算
# ============================================================

"""全密度 n(r) = Σ_m |ψ_m|²"""
function density(ψ::Spinor, g::Grid)
    n = zeros(g.Nx, g.Ny, g.Nz)
    for idx in 1:NUM_COMPONENTS
        n .+= abs2.(ψ.ψ[:,:,:,idx])
    end
    return n
end

"""
    spin_density(ψ::Spinor, g::Grid)

スピン密度ベクトル F = (Fx, Fy, Fz)
F_ν = Σ_{m,m'} ψ_m* (f_ν)_{mm'} ψ_{m'}
"""
function spin_density(ψ::Spinor, g::Grid)
    Fx = zeros(ComplexF64, g.Nx, g.Ny, g.Nz)
    Fy = zeros(ComplexF64, g.Nx, g.Ny, g.Nz)
    Fz = zeros(ComplexF64, g.Nx, g.Ny, g.Nz)
    
    for idx2 in 1:NUM_COMPONENTS, idx1 in 1:NUM_COMPONENTS
        @views begin
            Fx .+= conj.(ψ.ψ[:,:,:,idx1]) .* FX[idx1,idx2] .* ψ.ψ[:,:,:,idx2]
            Fy .+= conj.(ψ.ψ[:,:,:,idx1]) .* FY[idx1,idx2] .* ψ.ψ[:,:,:,idx2]
            Fz .+= conj.(ψ.ψ[:,:,:,idx1]) .* FZ[idx1,idx2] .* ψ.ψ[:,:,:,idx2]
        end
    end
    
    return real.(Fx), real.(Fy), real.(Fz)
end

"""
    magnetization(ψ::Spinor, g::Grid)

全磁化 ⟨F⟩ = ∫ F(r) dr
"""
function magnetization(ψ::Spinor, g::Grid)
    Fx, Fy, Fz = spin_density(ψ, g)
    return sum(Fx)*g.dV, sum(Fy)*g.dV, sum(Fz)*g.dV
end

"""
    angular_momentum(ψ::Spinor, g::Grid)

軌道角運動量 Lz = ∫ Σ_m (Jz - m) |ψ_m|² dr
Flower相では Jz = 6
"""
function angular_momentum(ψ::Spinor, g::Grid; Jz::Int=SPIN_F)
    Lz = 0.0
    for m in -SPIN_F:SPIN_F
        idx = m_to_idx(m)
        winding = Jz - m
        Lz += winding * sum(abs2.(ψ.ψ[:,:,:,idx])) * g.dV
    end
    return Lz
end

"""成分ごとの粒子数"""
function component_populations(ψ::Spinor, g::Grid)
    pops = zeros(NUM_COMPONENTS)
    for idx in 1:NUM_COMPONENTS
        pops[idx] = sum(abs2.(ψ.ψ[:,:,:,idx])) * g.dV
    end
    return pops
end

# ============================================================
# エネルギー計算
# ============================================================

"""
    energy(ψ::Spinor, p::Params, g::Grid)

全エネルギーを計算
"""
function energy(ψ::Spinor, p::Params, g::Grid)
    E_kin, E_trap, E_int, E_spin, E_dd = energy_components(ψ, p, g)
    return E_kin + E_trap + E_int + E_spin + E_dd
end

"""
    energy_components(ψ::Spinor, p::Params, g::Grid)

エネルギーの各成分を返す
"""
function energy_components(ψ::Spinor, p::Params, g::Grid)
    n = density(ψ, g)
    Fx, Fy, Fz = spin_density(ψ, g)
    F2 = Fx.^2 .+ Fy.^2 .+ Fz.^2
    
    # 運動エネルギー
    E_kin = 0.0
    for idx in 1:NUM_COMPONENTS
        ψk = fft(ψ.ψ[:,:,:,idx])
        E_kin += 0.5 * sum(g.k2 .* abs2.(ψk)) * g.dV / (g.Nx * g.Ny * g.Nz)
    end
    
    # トラップポテンシャル
    E_trap = 0.0
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        V = 0.5 * (g.x[i]^2 + g.y[j]^2 + (p.λz * g.z[k])^2)
        E_trap += V * n[i,j,k]
    end
    E_trap *= g.dV
    
    # 密度相互作用
    E_int = 0.5 * p.g0 / p.N * sum(n.^2) * g.dV
    
    # スピン相互作用
    E_spin = 0.5 * p.g1 / p.N * sum(F2) * g.dV
    
    # 双極子相互作用（簡略化：平均場近似）
    E_dd = compute_ddi_energy(ψ, p, g)
    
    return E_kin, E_trap, E_int, E_spin, E_dd
end

"""DDIエネルギー（フーリエ空間で計算）"""
function compute_ddi_energy(ψ::Spinor, p::Params, g::Grid)
    if abs(p.gdd) < 1e-10
        return 0.0
    end
    
    n = density(ψ, g)
    _, _, Fz = spin_density(ψ, g)
    
    # 有効密度: n_eff = n - 3*Fz/f (軸方向に揃った双極子の場合)
    n_eff = n .- 3.0 .* Fz ./ SPIN_F
    
    nk = fft(n_eff)
    
    # DDIカーネル: Q(k) = (4π/3) * c_dd * (3cos²θ_k - 1)
    E_dd = 0.0
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        k2_local = g.k2[i,j,k]
        if k2_local > 1e-10
            cos2_theta = g.kz[k]^2 / k2_local
            Qk = (3*cos2_theta - 1)
            E_dd += Qk * abs2(nk[i,j,k])
        end
    end
    
    E_dd *= 0.5 * p.gdd / p.N * g.dV / (g.Nx * g.Ny * g.Nz)
    return real(E_dd)
end

# ============================================================
# 時間発展
# ============================================================

"""
    evolve_imag!(ψ::Spinor, p::Params, g::Grid; dt, nsteps, callback)

虚時間発展による基底状態探索
Split-step Fourier法
"""
function evolve_imag!(ψ::Spinor, p::Params, g::Grid; 
                      dt::Float64=1e-4, nsteps::Int=10000,
                      callback=nothing, callback_interval::Int=100)
    
    # 運動エネルギー演算子 exp(-dt * k²/2)
    exp_K = exp.(-dt .* g.k2 ./ 2)
    
    # 作業配列
    H_local = zeros(ComplexF64, NUM_COMPONENTS)
    
    for step in 1:nsteps
        # 1. 運動エネルギー項（半ステップ）
        for idx in 1:NUM_COMPONENTS
            ψ.ψk[:,:,:,idx] = fft(ψ.ψ[:,:,:,idx])
            ψ.ψk[:,:,:,idx] .*= exp_K
            ψ.ψ[:,:,:,idx] = ifft(ψ.ψk[:,:,:,idx])
        end
        
        # 2. ポテンシャル項（フルステップ）
        apply_potential_step_imag!(ψ, p, g, dt)
        
        # 3. 運動エネルギー項（半ステップ）
        for idx in 1:NUM_COMPONENTS
            ψ.ψk[:,:,:,idx] = fft(ψ.ψ[:,:,:,idx])
            ψ.ψk[:,:,:,idx] .*= exp_K
            ψ.ψ[:,:,:,idx] = ifft(ψ.ψk[:,:,:,idx])
        end
        
        # 規格化
        normalize!(ψ, p, g)
        
        # コールバック
        if callback !== nothing && step % callback_interval == 0
            callback(step, ψ, p, g)
        end
    end
end

"""
    evolve_real!(ψ::Spinor, p::Params, g::Grid; dt, nsteps, callback)

実時間発展（ダイナミクス）
"""
function evolve_real!(ψ::Spinor, p::Params, g::Grid;
                      dt::Float64=1e-4, nsteps::Int=1000,
                      callback=nothing, callback_interval::Int=100)
    
    # 運動エネルギー演算子 exp(-i*dt * k²/2)
    exp_K = exp.(-im * dt .* g.k2 ./ 2)
    
    for step in 1:nsteps
        # Split-step
        for idx in 1:NUM_COMPONENTS
            ψ.ψk[:,:,:,idx] = fft(ψ.ψ[:,:,:,idx])
            ψ.ψk[:,:,:,idx] .*= exp_K
            ψ.ψ[:,:,:,idx] = ifft(ψ.ψk[:,:,:,idx])
        end
        
        apply_potential_step_real!(ψ, p, g, dt)
        
        for idx in 1:NUM_COMPONENTS
            ψ.ψk[:,:,:,idx] = fft(ψ.ψ[:,:,:,idx])
            ψ.ψk[:,:,:,idx] .*= exp_K
            ψ.ψ[:,:,:,idx] = ifft(ψ.ψk[:,:,:,idx])
        end
        
        if callback !== nothing && step % callback_interval == 0
            callback(step, ψ, p, g)
        end
    end
end

"""ポテンシャル項の適用（虚時間）"""
function apply_potential_step_imag!(ψ::Spinor, p::Params, g::Grid, dt::Float64)
    n = density(ψ, g)
    Fx, Fy, Fz = spin_density(ψ, g)
    Vdd = compute_ddi_potential(ψ, p, g)
    
    H_local = zeros(ComplexF64, NUM_COMPONENTS)
    ψ_new = zeros(ComplexF64, NUM_COMPONENTS)
    
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        # トラップポテンシャル
        V_trap = 0.5 * (g.x[i]^2 + g.y[j]^2 + (p.λz * g.z[k])^2)
        
        # 平均場ポテンシャル
        V_mf = p.g0 / p.N * n[i,j,k]
        
        # 局所ハミルトニアン行列を構築
        for idx1 in 1:NUM_COMPONENTS
            H_local[idx1] = (V_trap + V_mf + Vdd[i,j,k]) * ψ.ψ[i,j,k,idx1]
            
            # スピン相互作用項: c1 * F · f
            for idx2 in 1:NUM_COMPONENTS
                H_local[idx1] += p.g1 / p.N * (
                    Fx[i,j,k] * FX[idx1,idx2] +
                    Fy[i,j,k] * FY[idx1,idx2] +
                    Fz[i,j,k] * FZ[idx1,idx2]
                ) * ψ.ψ[i,j,k,idx2]
            end
        end
        
        # 虚時間発展: ψ_new = exp(-dt*H) ψ ≈ ψ - dt*H*ψ
        for idx in 1:NUM_COMPONENTS
            ψ.ψ[i,j,k,idx] -= dt * H_local[idx]
        end
    end
end

"""ポテンシャル項の適用（実時間）"""
function apply_potential_step_real!(ψ::Spinor, p::Params, g::Grid, dt::Float64)
    n = density(ψ, g)
    Fx, Fy, Fz = spin_density(ψ, g)
    Vdd = compute_ddi_potential(ψ, p, g)
    
    H_local = zeros(ComplexF64, NUM_COMPONENTS)
    
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        V_trap = 0.5 * (g.x[i]^2 + g.y[j]^2 + (p.λz * g.z[k])^2)
        V_mf = p.g0 / p.N * n[i,j,k]
        
        for idx1 in 1:NUM_COMPONENTS
            H_local[idx1] = (V_trap + V_mf + Vdd[i,j,k]) * ψ.ψ[i,j,k,idx1]
            
            for idx2 in 1:NUM_COMPONENTS
                H_local[idx1] += p.g1 / p.N * (
                    Fx[i,j,k] * FX[idx1,idx2] +
                    Fy[i,j,k] * FY[idx1,idx2] +
                    Fz[i,j,k] * FZ[idx1,idx2]
                ) * ψ.ψ[i,j,k,idx2]
            end
        end
        
        # 実時間発展: ψ_new = exp(-i*dt*H) ψ ≈ ψ - i*dt*H*ψ
        for idx in 1:NUM_COMPONENTS
            ψ.ψ[i,j,k,idx] -= im * dt * H_local[idx]
        end
    end
end

"""DDIポテンシャル（フーリエ空間）"""
function compute_ddi_potential(ψ::Spinor, p::Params, g::Grid)
    Vdd = zeros(g.Nx, g.Ny, g.Nz)
    
    if abs(p.gdd) < 1e-10
        return Vdd
    end
    
    n = density(ψ, g)
    _, _, Fz = spin_density(ψ, g)
    
    # 有効密度
    n_eff = n .- 3.0 .* Fz ./ SPIN_F
    nk = fft(n_eff)
    
    # DDIカーネル適用
    Vk = zeros(ComplexF64, g.Nx, g.Ny, g.Nz)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        k2_local = g.k2[i,j,k]
        if k2_local > 1e-10
            cos2_theta = g.kz[k]^2 / k2_local
            Qk = (3*cos2_theta - 1)
            Vk[i,j,k] = p.gdd / p.N * Qk * nk[i,j,k]
        end
    end
    
    Vdd = real.(ifft(Vk))
    return Vdd
end

# ============================================================
# ユーティリティ
# ============================================================

"""状態を保存"""
function save_state(filename::String, ψ::Spinor, p::Params, g::Grid)
    # JLD2などで保存
    # 簡易版：テキストファイルに基本情報のみ
    open(filename, "w") do io
        println(io, "# EuFlowerPhase state")
        println(io, "N = $(p.N)")
        println(io, "g0 = $(p.g0)")
        println(io, "g1 = $(p.g1)")
        println(io, "gdd = $(p.gdd)")
        println(io, "λz = $(p.λz)")
        
        Lz = angular_momentum(ψ, g)
        Mx, My, Mz = magnetization(ψ, g)
        E = energy(ψ, p, g)
        
        println(io, "Lz/N = $(Lz / p.N)")
        println(io, "Mz/N = $(Mz / p.N)")
        println(io, "E/N = $(E / p.N)")
    end
end

"""進捗表示用コールバック"""
function default_callback(step, ψ, p, g)
    E = energy(ψ, p, g)
    Lz = angular_momentum(ψ, g)
    pops = component_populations(ψ, g)
    
    println("Step $step: E/N = $(E/p.N), Lz/N = $(Lz/p.N)")
    println("  Populations: m=+6: $(pops[1]/p.N), m=+5: $(pops[2]/p.N), ...")
end

end # module

