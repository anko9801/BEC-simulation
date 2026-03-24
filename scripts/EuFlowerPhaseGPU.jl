"""
    EuFlowerPhaseGPU.jl

F=6 Europium BEC Flower相シミュレーション - GPU加速版

NVIDIA RTX 3070Ti (8GB VRAM) 向けに最適化
CUDA.jl を使用

# 必要パッケージ
using Pkg
Pkg.add(["CUDA", "FFTW"])
"""
module EuFlowerPhaseGPU

using CUDA
using CUDA.CUFFT
using LinearAlgebra

export Params, Grid, SpinorGPU
export init_ferromagnetic!, init_flower!, init_random_perturbation!
export evolve_imag!, evolve_real!
export density, spin_density, angular_momentum, magnetization
export energy, energy_components, component_populations
export m_to_idx, idx_to_m
export to_cpu, to_gpu

# ============================================================
# 定数
# ============================================================

const SPIN_F = 6
const NUM_COMPONENTS = 2 * SPIN_F + 1  # 13

# ============================================================
# パラメータ構造体
# ============================================================

struct Params
    N::Float64
    g0::Float64
    g1::Float64
    gdd::Float64
    λz::Float64

    function Params(; N=1.5e4, g0=nothing, g1=0.0, gdd=nothing,
                    λz=0.5, a_s=135.0, ε_dd=0.44, a_ho_nm=850.0)
        a_B_nm = 0.0529
        if g0 === nothing
            g0 = 4π * (a_s * a_B_nm / a_ho_nm) * N
        end
        if gdd === nothing
            gdd = ε_dd * g0
        end
        new(N, g0, g1, gdd, λz)
    end
end

# ============================================================
# グリッド構造体（GPU配列を含む）
# ============================================================

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
    dV::Float64
    # CPU配列
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    # GPU配列
    k2_gpu::CuArray{Float32,3}
    kz2_gpu::CuArray{Float32,3}  # kz^2 for DDI
    trap_gpu::CuArray{Float32,3}  # トラップポテンシャル

    function Grid(Nx::Int, Ny::Int, Nz::Int, Lx::Float64;
                  Ly::Float64=Lx, Lz::Float64=Lx/2, λz::Float64=0.5)
        dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
        dV = dx * dy * dz

        x = range(-Lx/2, Lx/2 - dx, length=Nx) |> collect
        y = range(-Ly/2, Ly/2 - dy, length=Ny) |> collect
        z = range(-Lz/2, Lz/2 - dz, length=Nz) |> collect

        # 波数空間（CPU）
        kx = fftfreq(Nx, 2π/dx)
        ky = fftfreq(Ny, 2π/dy)
        kz = fftfreq(Nz, 2π/dz)

        # k² と kz² をGPUに転送
        k2 = zeros(Float32, Nx, Ny, Nz)
        kz2 = zeros(Float32, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            k2[i,j,k] = kx[i]^2 + ky[j]^2 + kz[k]^2
            kz2[i,j,k] = kz[k]^2
        end

        # トラップポテンシャル
        trap = zeros(Float32, Nx, Ny, Nz)
        for k in 1:Nz, j in 1:Ny, i in 1:Nx
            trap[i,j,k] = 0.5f0 * (x[i]^2 + y[j]^2 + (λz * z[k])^2)
        end

        k2_gpu = CuArray(k2)
        kz2_gpu = CuArray(kz2)
        trap_gpu = CuArray(trap)

        new(Nx, Ny, Nz, Lx, Ly, Lz, dx, dy, dz, dV,
            x, y, z, k2_gpu, kz2_gpu, trap_gpu)
    end
end

Grid(N::Int, L::Float64; λz::Float64=0.5) = Grid(N, N, N÷2, L, λz=λz)

# ============================================================
# fftfreq関数
# ============================================================

function fftfreq(n::Int, d::Float64)
    freq = zeros(n)
    for i in 0:n-1
        if i < n÷2 + 1
            freq[i+1] = i / (n * d) * 2π
        else
            freq[i+1] = (i - n) / (n * d) * 2π
        end
    end
    return freq
end

# ============================================================
# スピノル波動関数（GPU）
# ============================================================

mutable struct SpinorGPU
    ψ::CuArray{ComplexF32,4}   # [Nx, Ny, Nz, 13]
    ψk::CuArray{ComplexF32,4}  # FFT buffer
    # 作業配列
    n::CuArray{Float32,3}      # 密度
    Fx::CuArray{Float32,3}     # スピン密度
    Fy::CuArray{Float32,3}
    Fz::CuArray{Float32,3}
    Vdd::CuArray{Float32,3}    # DDIポテンシャル
    # FFTプラン
    fft_plan::CUFFT.cCuFFTPlan
    ifft_plan::AbstractFFTs.ScaledPlan

    function SpinorGPU(g::Grid)
        ψ = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz, NUM_COMPONENTS)
        ψk = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz, NUM_COMPONENTS)
        n = CUDA.zeros(Float32, g.Nx, g.Ny, g.Nz)
        Fx = CUDA.zeros(Float32, g.Nx, g.Ny, g.Nz)
        Fy = CUDA.zeros(Float32, g.Nx, g.Ny, g.Nz)
        Fz = CUDA.zeros(Float32, g.Nx, g.Ny, g.Nz)
        Vdd = CUDA.zeros(Float32, g.Nx, g.Ny, g.Nz)

        # FFTプラン（3D FFT、バッチ処理なし）
        tmp = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz)
        fft_plan = plan_fft!(tmp)
        ifft_plan = plan_ifft!(tmp)

        new(ψ, ψk, n, Fx, Fy, Fz, Vdd, fft_plan, ifft_plan)
    end
end

# CPU ↔ GPU 変換
function to_cpu(ψ_gpu::SpinorGPU)
    return Array(ψ_gpu.ψ)
end

function to_gpu(ψ_cpu::Array{ComplexF64,4}, ψ_gpu::SpinorGPU)
    ψ_gpu.ψ .= CuArray(ComplexF32.(ψ_cpu))
end

# ============================================================
# スピン行列（CPU上で計算、GPUカーネルで使用）
# ============================================================

function spin_matrices()
    f = SPIN_F
    dim = NUM_COMPONENTS

    fz = zeros(Float32, dim, dim)
    fp = zeros(Float32, dim, dim)
    fm = zeros(Float32, dim, dim)

    for idx in 1:dim
        m = f - (idx - 1)
        fz[idx, idx] = m

        if idx > 1
            m_next = m + 1
            fp[idx-1, idx] = sqrt(f*(f+1) - m*m_next)
        end
        if idx < dim
            m_prev = m - 1
            fm[idx+1, idx] = sqrt(f*(f+1) - m*m_prev)
        end
    end

    fx = (fp + fm) / 2
    fy = (fp - fm) / 2  # 虚数単位は後で処理

    return fx, fy, fz, fp, fm
end

const FX_CPU, FY_CPU, FZ_CPU, FP_CPU, FM_CPU = spin_matrices()
const FX_GPU = CuArray(FX_CPU)
const FY_GPU = CuArray(FY_CPU)
const FZ_GPU = CuArray(FZ_CPU)

m_to_idx(m::Int) = SPIN_F - m + 1
idx_to_m(idx::Int) = SPIN_F - (idx - 1)

# ============================================================
# 初期化関数
# ============================================================

function init_ferromagnetic!(ψ::SpinorGPU, p::Params, g::Grid)
    ψ.ψ .= 0

    μ_TF = 0.5f0 * (15 * p.g0 / (4π))^0.4f0
    idx_p6 = m_to_idx(6)

    # GPUカーネル
    function kernel_init_fm!(ψ, trap, μ_TF, g0, idx_p6)
        i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
        k = (blockIdx().z - 1) * blockDim().z + threadIdx().z

        Nx, Ny, Nz, _ = size(ψ)
        if i <= Nx && j <= Ny && k <= Nz
            V = trap[i,j,k]
            n_local = max(0f0, μ_TF - V) / g0
            ψ[i,j,k,idx_p6] = ComplexF32(sqrt(n_local), 0f0)
        end
        return nothing
    end

    threads = (8, 8, 4)
    blocks = (cld(g.Nx, 8), cld(g.Ny, 8), cld(g.Nz, 4))

    @cuda threads=threads blocks=blocks kernel_init_fm!(
        ψ.ψ, g.trap_gpu, Float32(μ_TF), Float32(p.g0), idx_p6)

    normalize!(ψ, p, g)
end

function init_random_perturbation!(ψ::SpinorGPU, p::Params, g::Grid; amp::Float64=0.02)
    init_ferromagnetic!(ψ, p, g)

    # CPU上でノイズを生成してGPUに転送
    ψ_cpu = Array(ψ.ψ)
    n_p6 = abs2.(ψ_cpu[:,:,:,m_to_idx(6)])

    for m in -SPIN_F:(SPIN_F-1)
        idx = m_to_idx(m)
        noise = amp .* sqrt.(n_p6) .* (randn(ComplexF32, g.Nx, g.Ny, g.Nz))
        ψ_cpu[:,:,:,idx] .= noise
    end

    ψ.ψ .= CuArray(ψ_cpu)
    normalize!(ψ, p, g)
end

function init_flower!(ψ::SpinorGPU, p::Params, g::Grid; Jz::Int=SPIN_F, seed_amp::Float64=0.1)
    init_ferromagnetic!(ψ, p, g)

    # CPU上で位相を設定
    ψ_cpu = Array(ψ.ψ)

    for m in -SPIN_F:SPIN_F
        idx = m_to_idx(m)
        winding = Jz - m

        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            φ = atan(g.y[j], g.x[i])
            r = sqrt(g.x[i]^2 + g.y[j]^2)

            if winding != 0
                r_core = 1.0f0
                core_factor = r^abs(winding) / (r^abs(winding) + r_core^abs(winding))

                if m != SPIN_F
                    ψ_cpu[i,j,k,idx] = seed_amp * sqrt(abs2(ψ_cpu[i,j,k,m_to_idx(SPIN_F)])) *
                                       core_factor * exp(im * winding * φ)
                else
                    ψ_cpu[i,j,k,idx] *= exp(im * winding * φ)
                end
            end
        end
    end

    ψ.ψ .= CuArray(ComplexF32.(ψ_cpu))
    normalize!(ψ, p, g)
end

# ============================================================
# 規格化（GPU）
# ============================================================

function normalize!(ψ::SpinorGPU, p::Params, g::Grid)
    norm2 = Float64(sum(abs2.(ψ.ψ))) * g.dV
    ψ.ψ .*= Float32(sqrt(p.N / norm2))
end

# ============================================================
# 物理量計算（GPU）
# ============================================================

"""密度計算カーネル"""
function compute_density!(n::CuArray, ψ::CuArray)
    n .= 0f0
    for idx in 1:NUM_COMPONENTS
        @views n .+= abs2.(ψ[:,:,:,idx])
    end
end

function density(ψ::SpinorGPU, g::Grid)
    compute_density!(ψ.n, ψ.ψ)
    return Array(ψ.n)
end

"""スピン密度計算"""
function compute_spin_density!(Fx, Fy, Fz, ψ)
    Fx .= 0f0
    Fy .= 0f0
    Fz .= 0f0

    # F_z は対角
    for idx in 1:NUM_COMPONENTS
        m = Float32(idx_to_m(idx))
        @views Fz .+= m .* abs2.(ψ[:,:,:,idx])
    end

    # F_x, F_y は非対角（昇降演算子）
    for idx in 1:(NUM_COMPONENTS-1)
        # f_+ の寄与
        coeff = FP_CPU[idx, idx+1]
        if abs(coeff) > 1e-10
            @views begin
                # F_x += Re(ψ*_{idx} ψ_{idx+1}) * coeff
                Fx .+= coeff .* real.(conj.(ψ[:,:,:,idx]) .* ψ[:,:,:,idx+1])
                # F_y += Im(ψ*_{idx} ψ_{idx+1}) * coeff (from f_+ - f_-)
                Fy .+= coeff .* imag.(conj.(ψ[:,:,:,idx]) .* ψ[:,:,:,idx+1])
            end
        end
    end

    # 対称化
    Fx .*= 2f0  # f_+ + f_- の係数
end

function spin_density(ψ::SpinorGPU, g::Grid)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    return Array(ψ.Fx), Array(ψ.Fy), Array(ψ.Fz)
end

"""磁化"""
function magnetization(ψ::SpinorGPU, g::Grid)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    Mx = Float64(sum(ψ.Fx)) * g.dV
    My = Float64(sum(ψ.Fy)) * g.dV
    Mz = Float64(sum(ψ.Fz)) * g.dV
    return Mx, My, Mz
end

"""軌道角運動量"""
function angular_momentum(ψ::SpinorGPU, g::Grid; Jz::Int=SPIN_F)
    Lz = 0.0
    ψ_cpu = Array(ψ.ψ)
    for m in -SPIN_F:SPIN_F
        idx = m_to_idx(m)
        winding = Jz - m
        Lz += winding * sum(abs2.(ψ_cpu[:,:,:,idx])) * g.dV
    end
    return Lz
end

"""成分ごとの粒子数"""
function component_populations(ψ::SpinorGPU, g::Grid)
    pops = zeros(NUM_COMPONENTS)
    ψ_cpu = Array(ψ.ψ)
    for idx in 1:NUM_COMPONENTS
        pops[idx] = sum(abs2.(ψ_cpu[:,:,:,idx])) * g.dV
    end
    return pops
end

# ============================================================
# エネルギー計算
# ============================================================

function energy(ψ::SpinorGPU, p::Params, g::Grid)
    E_kin, E_trap, E_int, E_spin, E_dd = energy_components(ψ, p, g)
    return E_kin + E_trap + E_int + E_spin + E_dd
end

function energy_components(ψ::SpinorGPU, p::Params, g::Grid)
    compute_density!(ψ.n, ψ.ψ)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)

    # 運動エネルギー（GPU FFT）
    E_kin = 0.0
    tmp = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz)
    for idx in 1:NUM_COMPONENTS
        tmp .= ψ.ψ[:,:,:,idx]
        ψ.fft_plan * tmp  # in-place FFT
        E_kin += 0.5 * Float64(sum(g.k2_gpu .* abs2.(tmp))) * g.dV / (g.Nx * g.Ny * g.Nz)
    end

    # トラップエネルギー
    E_trap = Float64(sum(g.trap_gpu .* ψ.n)) * g.dV

    # 密度相互作用
    E_int = 0.5 * p.g0 / p.N * Float64(sum(ψ.n .^ 2)) * g.dV

    # スピン相互作用
    F2 = ψ.Fx .^ 2 .+ ψ.Fy .^ 2 .+ ψ.Fz .^ 2
    E_spin = 0.5 * p.g1 / p.N * Float64(sum(F2)) * g.dV

    # DDI
    E_dd = compute_ddi_energy(ψ, p, g)

    return E_kin, E_trap, E_int, E_spin, E_dd
end

function compute_ddi_energy(ψ::SpinorGPU, p::Params, g::Grid)
    if abs(p.gdd) < 1e-10
        return 0.0
    end

    compute_density!(ψ.n, ψ.ψ)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)

    # 有効密度
    n_eff = ψ.n .- 3f0 .* ψ.Fz ./ Float32(SPIN_F)

    # FFT
    nk = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz)
    nk .= ComplexF32.(n_eff)
    ψ.fft_plan * nk

    # DDI カーネル: (3 kz²/k² - 1)
    E_dd = 0.0
    nk_cpu = Array(nk)
    k2_cpu = Array(g.k2_gpu)
    kz2_cpu = Array(g.kz2_gpu)

    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        if k2_cpu[i,j,k] > 1e-10
            cos2_theta = kz2_cpu[i,j,k] / k2_cpu[i,j,k]
            Qk = 3*cos2_theta - 1
            E_dd += Qk * abs2(nk_cpu[i,j,k])
        end
    end

    E_dd *= 0.5 * p.gdd / p.N * g.dV / (g.Nx * g.Ny * g.Nz)
    return E_dd
end

# ============================================================
# 時間発展（GPU加速）
# ============================================================

"""虚時間発展"""
function evolve_imag!(ψ::SpinorGPU, p::Params, g::Grid;
                      dt::Float64=1e-4, nsteps::Int=10000,
                      callback=nothing, callback_interval::Int=100)

    dt32 = Float32(dt)
    exp_K = exp.(-dt32 .* g.k2_gpu ./ 2f0)

    tmp = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz)

    for step in 1:nsteps
        # 運動エネルギー項（半ステップ）
        for idx in 1:NUM_COMPONENTS
            tmp .= ψ.ψ[:,:,:,idx]
            ψ.fft_plan * tmp
            tmp .*= exp_K
            ψ.ifft_plan * tmp
            ψ.ψ[:,:,:,idx] .= tmp
        end

        # ポテンシャル項
        apply_potential_step_imag!(ψ, p, g, dt32)

        # 運動エネルギー項（半ステップ）
        for idx in 1:NUM_COMPONENTS
            tmp .= ψ.ψ[:,:,:,idx]
            ψ.fft_plan * tmp
            tmp .*= exp_K
            ψ.ifft_plan * tmp
            ψ.ψ[:,:,:,idx] .= tmp
        end

        # 規格化
        normalize!(ψ, p, g)

        # コールバック
        if callback !== nothing && step % callback_interval == 0
            callback(step, ψ, p, g)
        end
    end
end

"""実時間発展"""
function evolve_real!(ψ::SpinorGPU, p::Params, g::Grid;
                      dt::Float64=1e-4, nsteps::Int=1000,
                      callback=nothing, callback_interval::Int=100)

    dt32 = Float32(dt)
    exp_K = exp.(-1im * dt32 .* g.k2_gpu ./ 2f0)

    tmp = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz)

    for step in 1:nsteps
        for idx in 1:NUM_COMPONENTS
            tmp .= ψ.ψ[:,:,:,idx]
            ψ.fft_plan * tmp
            tmp .*= exp_K
            ψ.ifft_plan * tmp
            ψ.ψ[:,:,:,idx] .= tmp
        end

        apply_potential_step_real!(ψ, p, g, dt32)

        for idx in 1:NUM_COMPONENTS
            tmp .= ψ.ψ[:,:,:,idx]
            ψ.fft_plan * tmp
            tmp .*= exp_K
            ψ.ifft_plan * tmp
            ψ.ψ[:,:,:,idx] .= tmp
        end

        if callback !== nothing && step % callback_interval == 0
            callback(step, ψ, p, g)
        end
    end
end

"""ポテンシャル項の適用（虚時間）- GPU カーネル版"""
function apply_potential_step_imag!(ψ::SpinorGPU, p::Params, g::Grid, dt::Float32)
    compute_density!(ψ.n, ψ.ψ)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    compute_ddi_potential!(ψ, p, g)

    g0_N = Float32(p.g0 / p.N)
    g1_N = Float32(p.g1 / p.N)

    # 簡略化版：対角成分のみ（c1 F·f の対角部分）
    for idx in 1:NUM_COMPONENTS
        m = Float32(idx_to_m(idx))

        @views begin
            # V_eff = V_trap + g0*n + g1*Fz*m + Vdd
            V_eff = g.trap_gpu .+ g0_N .* ψ.n .+ g1_N .* ψ.Fz .* m .+ ψ.Vdd

            # 正確な指数関数を使用
            ψ.ψ[:,:,:,idx] .*= exp.((-dt) .* V_eff)
        end
    end

    # 非対角項（スピン混合）- より正確な計算が必要な場合
    if abs(p.g1) > 1e-10
        apply_spin_mixing!(ψ, p, g, dt)
    end
end

"""ポテンシャル項の適用（実時間）"""
function apply_potential_step_real!(ψ::SpinorGPU, p::Params, g::Grid, dt::Float32)
    compute_density!(ψ.n, ψ.ψ)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    compute_ddi_potential!(ψ, p, g)

    g0_N = Float32(p.g0 / p.N)
    g1_N = Float32(p.g1 / p.N)

    for idx in 1:NUM_COMPONENTS
        m = Float32(idx_to_m(idx))

        @views begin
            V_eff = g.trap_gpu .+ g0_N .* ψ.n .+ g1_N .* ψ.Fz .* m .+ ψ.Vdd
            ψ.ψ[:,:,:,idx] .*= exp.(-1im .* dt .* V_eff)
        end
    end

    if abs(p.g1) > 1e-10
        apply_spin_mixing_real!(ψ, p, g, dt)
    end
end

"""スピン混合項（非対角）"""
function apply_spin_mixing!(ψ::SpinorGPU, p::Params, g::Grid, dt::Float32)
    g1_N = Float32(p.g1 / p.N)

    # F_+ と F_- による混合
    # c1 * (F_x * f_x + F_y * f_y) の効果
    # f_± |m⟩ = √(f(f+1)-m(m±1)) |m±1⟩

    ψ_old = copy(ψ.ψ)

    for idx in 1:(NUM_COMPONENTS-1)
        coeff = FP_CPU[idx, idx+1]  # √(f(f+1)-m(m+1))
        if abs(coeff) > 1e-10
            # F_+ による m → m+1 遷移、F_- による m+1 → m 遷移
            @views begin
                # δψ_idx = -dt * g1 * (F_x - i*F_y)/2 * coeff * ψ_{idx+1}
                #        = -dt * g1 * F_- * coeff * ψ_{idx+1} / 2
                mix_term = g1_N * coeff * (ψ.Fx .- 1im .* ψ.Fy) ./ 2f0
                ψ.ψ[:,:,:,idx] .-= dt .* mix_term .* ψ_old[:,:,:,idx+1]

                # δψ_{idx+1} = -dt * g1 * (F_x + i*F_y)/2 * coeff * ψ_idx
                mix_term_conj = g1_N * coeff * (ψ.Fx .+ 1im .* ψ.Fy) ./ 2f0
                ψ.ψ[:,:,:,idx+1] .-= dt .* mix_term_conj .* ψ_old[:,:,:,idx]
            end
        end
    end
end

function apply_spin_mixing_real!(ψ::SpinorGPU, p::Params, g::Grid, dt::Float32)
    g1_N = Float32(p.g1 / p.N)
    ψ_old = copy(ψ.ψ)

    for idx in 1:(NUM_COMPONENTS-1)
        coeff = FP_CPU[idx, idx+1]
        if abs(coeff) > 1e-10
            @views begin
                mix_term = g1_N * coeff * (ψ.Fx .- 1im .* ψ.Fy) ./ 2f0
                ψ.ψ[:,:,:,idx] .-= 1im .* dt .* mix_term .* ψ_old[:,:,:,idx+1]

                mix_term_conj = g1_N * coeff * (ψ.Fx .+ 1im .* ψ.Fy) ./ 2f0
                ψ.ψ[:,:,:,idx+1] .-= 1im .* dt .* mix_term_conj .* ψ_old[:,:,:,idx]
            end
        end
    end
end

"""DDIポテンシャル計算"""
function compute_ddi_potential!(ψ::SpinorGPU, p::Params, g::Grid)
    if abs(p.gdd) < 1e-10
        ψ.Vdd .= 0f0
        return
    end

    # 有効密度
    n_eff = ψ.n .- 3f0 .* ψ.Fz ./ Float32(SPIN_F)

    # FFT
    nk = CUDA.zeros(ComplexF32, g.Nx, g.Ny, g.Nz)
    nk .= ComplexF32.(n_eff)
    ψ.fft_plan * nk

    # DDIカーネル適用
    gdd_N = Float32(p.gdd / p.N)
    @. nk *= gdd_N * (3f0 * g.kz2_gpu / (g.k2_gpu + 1f-10) - 1f0)

    # k=0 を処理
    nk[1,1,1] = 0f0

    # 逆FFT
    ψ.ifft_plan * nk
    ψ.Vdd .= real.(nk)
end

end # module
