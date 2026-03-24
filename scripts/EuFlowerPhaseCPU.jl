"""
    EuFlowerPhaseCPU.jl

F=6 Europium BEC Flower相シミュレーション - CPUマルチスレッド版

Ryzen 5900X (12コア24スレッド) 向けに最適化
Julia起動時に: julia -t 24 または環境変数 JULIA_NUM_THREADS=24

# 必要パッケージ
using Pkg
Pkg.add(["FFTW", "LoopVectorization"])
"""
module EuFlowerPhaseCPU

using FFTW
using LinearAlgebra
using Base.Threads

export Params, Grid, Spinor
export init_ferromagnetic!, init_flower!, init_random_perturbation!
export evolve_imag!, evolve_real!
export density, spin_density, angular_momentum, magnetization
export energy, energy_components, component_populations
export m_to_idx, idx_to_m

# ============================================================
# 定数
# ============================================================

const SPIN_F = 6
const NUM_COMPONENTS = 2 * SPIN_F + 1  # 13

# ============================================================
# パラメータ
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
# グリッド
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
    x::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    kx::Vector{Float64}
    ky::Vector{Float64}
    kz::Vector{Float64}
    k2::Array{Float64,3}
    kz2::Array{Float64,3}
    trap::Array{Float64,3}

    function Grid(Nx::Int, Ny::Int, Nz::Int, Lx::Float64;
                  Ly::Float64=Lx, Lz::Float64=Lx/2, λz::Float64=0.5)
        dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz
        dV = dx * dy * dz

        x = range(-Lx/2, Lx/2 - dx, length=Nx) |> collect
        y = range(-Ly/2, Ly/2 - dy, length=Ny) |> collect
        z = range(-Lz/2, Lz/2 - dz, length=Nz) |> collect

        kx = fftfreq(Nx, 2π/dx) |> collect
        ky = fftfreq(Ny, 2π/dy) |> collect
        kz = fftfreq(Nz, 2π/dz) |> collect

        k2 = zeros(Nx, Ny, Nz)
        kz2 = zeros(Nx, Ny, Nz)
        trap = zeros(Nx, Ny, Nz)

        # 並列で初期化
        @threads for k in 1:Nz
            for j in 1:Ny
                @simd for i in 1:Nx
                    k2[i,j,k] = kx[i]^2 + ky[j]^2 + kz[k]^2
                    kz2[i,j,k] = kz[k]^2
                    trap[i,j,k] = 0.5 * (x[i]^2 + y[j]^2 + (λz * z[k])^2)
                end
            end
        end

        new(Nx, Ny, Nz, Lx, Ly, Lz, dx, dy, dz, dV,
            x, y, z, kx, ky, kz, k2, kz2, trap)
    end
end

Grid(N::Int, L::Float64; λz::Float64=0.5) = Grid(N, N, N÷2, L, λz=λz)

# ============================================================
# スピノル（マルチスレッドFFT対応）
# ============================================================

mutable struct Spinor
    ψ::Array{ComplexF64,4}   # [Nx, Ny, Nz, 13]
    ψk::Array{ComplexF64,4}  # FFT buffer
    # 作業配列
    n::Array{Float64,3}
    Fx::Array{Float64,3}
    Fy::Array{Float64,3}
    Fz::Array{Float64,3}
    Vdd::Array{Float64,3}
    # FFTプラン（各成分用）
    fft_plans::Vector{FFTW.cFFTWPlan}
    ifft_plans::Vector{AbstractFFTs.ScaledPlan}

    function Spinor(g::Grid)
        # FFTWのスレッド数を設定
        FFTW.set_num_threads(Threads.nthreads())

        ψ = zeros(ComplexF64, g.Nx, g.Ny, g.Nz, NUM_COMPONENTS)
        ψk = zeros(ComplexF64, g.Nx, g.Ny, g.Nz, NUM_COMPONENTS)
        n = zeros(g.Nx, g.Ny, g.Nz)
        Fx = zeros(g.Nx, g.Ny, g.Nz)
        Fy = zeros(g.Nx, g.Ny, g.Nz)
        Fz = zeros(g.Nx, g.Ny, g.Nz)
        Vdd = zeros(g.Nx, g.Ny, g.Nz)

        # 各スピン成分用のFFTプランを作成
        fft_plans = Vector{FFTW.cFFTWPlan}(undef, NUM_COMPONENTS)
        ifft_plans = Vector{AbstractFFTs.ScaledPlan}(undef, NUM_COMPONENTS)

        for idx in 1:NUM_COMPONENTS
            tmp = zeros(ComplexF64, g.Nx, g.Ny, g.Nz)
            fft_plans[idx] = plan_fft!(tmp, flags=FFTW.MEASURE)
            ifft_plans[idx] = plan_ifft!(tmp, flags=FFTW.MEASURE)
        end

        new(ψ, ψk, n, Fx, Fy, Fz, Vdd, fft_plans, ifft_plans)
    end
end

# ============================================================
# スピン行列
# ============================================================

function spin_matrices()
    f = SPIN_F
    dim = NUM_COMPONENTS

    fz = zeros(dim, dim)
    fp = zeros(dim, dim)
    fm = zeros(dim, dim)

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
    fy = (fp - fm) / (2im)

    return fx, fy, fz, fp, fm
end

const FX, FY, FZ, FP, FM = spin_matrices()

m_to_idx(m::Int) = SPIN_F - m + 1
idx_to_m(idx::Int) = SPIN_F - (idx - 1)

# ============================================================
# 初期化（並列化）
# ============================================================

function init_ferromagnetic!(ψ::Spinor, p::Params, g::Grid)
    ψ.ψ .= 0
    μ_TF = 0.5 * (15 * p.g0 / (4π))^0.4
    idx_p6 = m_to_idx(6)

    @threads for k in 1:g.Nz
        for j in 1:g.Ny
            @simd for i in 1:g.Nx
                V = g.trap[i,j,k]
                n_local = max(0.0, μ_TF - V) / p.g0
                ψ.ψ[i,j,k,idx_p6] = sqrt(n_local)
            end
        end
    end

    normalize!(ψ, p, g)
end

function init_random_perturbation!(ψ::Spinor, p::Params, g::Grid; amp::Float64=0.02)
    init_ferromagnetic!(ψ, p, g)

    n_p6 = abs2.(ψ.ψ[:,:,:,m_to_idx(6)])

    @threads for m in -SPIN_F:(SPIN_F-1)
        idx = m_to_idx(m)
        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            if n_p6[i,j,k] > 1e-10
                ψ.ψ[i,j,k,idx] = amp * sqrt(n_p6[i,j,k]) * (randn() + im*randn()) / sqrt(2)
            end
        end
    end

    normalize!(ψ, p, g)
end

"""m=0成分のみで初期化（DDIテスト用）"""
function init_zero_m!(ψ::Spinor, p::Params, g::Grid)
    fill!(ψ.ψ, 0.0)
    # m=0 は idx=7
    idx_m0 = m_to_idx(0)
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        r2 = g.x[i]^2 + g.y[j]^2 + (p.λz * g.z[k])^2
        ψ.ψ[i,j,k,idx_m0] = exp(-r2/4)
    end
    normalize!(ψ, p, g)
end

function init_flower!(ψ::Spinor, p::Params, g::Grid; Jz::Int=SPIN_F, seed_amp::Float64=0.1)
    init_ferromagnetic!(ψ, p, g)

    @threads for m in -SPIN_F:SPIN_F
        idx = m_to_idx(m)
        winding = Jz - m

        for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
            φ = atan(g.y[j], g.x[i])
            r = sqrt(g.x[i]^2 + g.y[j]^2)

            if winding != 0
                r_core = 1.0
                core_factor = r^abs(winding) / (r^abs(winding) + r_core^abs(winding))

                if m != SPIN_F
                    ψ.ψ[i,j,k,idx] = seed_amp * sqrt(abs2(ψ.ψ[i,j,k,m_to_idx(SPIN_F)])) *
                                     core_factor * exp(im * winding * φ)
                else
                    ψ.ψ[i,j,k,idx] *= exp(im * winding * φ)
                end
            end
        end
    end

    normalize!(ψ, p, g)
end

# ============================================================
# 規格化（並列リダクション）
# ============================================================

function normalize!(ψ::Spinor, p::Params, g::Grid)
    # 各スレッドで部分和を計算
    partial_sums = zeros(Threads.nthreads())

    @threads for idx in 1:NUM_COMPONENTS
        tid = Threads.threadid()
        partial_sums[tid] += sum(abs2.(ψ.ψ[:,:,:,idx]))
    end

    norm2 = sum(partial_sums) * g.dV
    scale = sqrt(p.N / norm2)

    @threads for idx in 1:NUM_COMPONENTS
        ψ.ψ[:,:,:,idx] .*= scale
    end
end

# ============================================================
# 物理量計算（並列化）
# ============================================================

function compute_density!(n::Array{Float64,3}, ψ::Array{ComplexF64,4})
    fill!(n, 0.0)

    # 各成分の寄与を並列に計算
    partial_n = [zeros(size(n)) for _ in 1:Threads.nthreads()]

    @threads for idx in 1:NUM_COMPONENTS
        tid = Threads.threadid()
        @views partial_n[tid] .+= abs2.(ψ[:,:,:,idx])
    end

    # 合計
    for tid in 1:Threads.nthreads()
        n .+= partial_n[tid]
    end
end

function density(ψ::Spinor, g::Grid)
    compute_density!(ψ.n, ψ.ψ)
    return copy(ψ.n)
end

function compute_spin_density!(Fx, Fy, Fz, ψ)
    fill!(Fx, 0.0)
    fill!(Fy, 0.0)
    fill!(Fz, 0.0)

    Nx, Ny, Nz, _ = size(ψ)

    # Fz（対角成分）
    @threads for k in 1:Nz
        for j in 1:Ny
            @simd for i in 1:Nx
                fz_local = 0.0
                for idx in 1:NUM_COMPONENTS
                    m = idx_to_m(idx)
                    fz_local += m * abs2(ψ[i,j,k,idx])
                end
                Fz[i,j,k] = fz_local
            end
        end
    end

    # Fx, Fy（非対角成分）
    @threads for k in 1:Nz
        for j in 1:Ny
            @simd for i in 1:Nx
                fx_local = 0.0
                fy_local = 0.0
                for idx in 1:(NUM_COMPONENTS-1)
                    coeff = FP[idx, idx+1]
                    if abs(coeff) > 1e-10
                        prod = conj(ψ[i,j,k,idx]) * ψ[i,j,k,idx+1]
                        fx_local += coeff * real(prod)
                        fy_local += coeff * imag(prod)
                    end
                end
                Fx[i,j,k] = 2.0 * fx_local
                Fy[i,j,k] = 2.0 * fy_local
            end
        end
    end
end

function spin_density(ψ::Spinor, g::Grid)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    return copy(ψ.Fx), copy(ψ.Fy), copy(ψ.Fz)
end

function magnetization(ψ::Spinor, g::Grid)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    Mx = sum(ψ.Fx) * g.dV
    My = sum(ψ.Fy) * g.dV
    Mz = sum(ψ.Fz) * g.dV
    return Mx, My, Mz
end

function angular_momentum(ψ::Spinor, g::Grid; Jz::Int=SPIN_F)
    partial_Lz = zeros(Threads.nthreads())

    @threads for m in -SPIN_F:SPIN_F
        tid = Threads.threadid()
        idx = m_to_idx(m)
        winding = Jz - m
        partial_Lz[tid] += winding * sum(abs2.(ψ.ψ[:,:,:,idx]))
    end

    return sum(partial_Lz) * g.dV
end

function component_populations(ψ::Spinor, g::Grid)
    pops = zeros(NUM_COMPONENTS)
    @threads for idx in 1:NUM_COMPONENTS
        pops[idx] = sum(abs2.(ψ.ψ[:,:,:,idx])) * g.dV
    end
    return pops
end

# ============================================================
# エネルギー（並列）
# ============================================================

function energy(ψ::Spinor, p::Params, g::Grid)
    E_kin, E_trap, E_int, E_spin, E_dd = energy_components(ψ, p, g)
    return E_kin + E_trap + E_int + E_spin + E_dd
end

function energy_components(ψ::Spinor, p::Params, g::Grid)
    compute_density!(ψ.n, ψ.ψ)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)

    # 運動エネルギー（並列FFT）
    E_kin_parts = zeros(NUM_COMPONENTS)
    @threads for idx in 1:NUM_COMPONENTS
        ψk_slice = fft(ψ.ψ[:,:,:,idx])
        E_kin_parts[idx] = 0.5 * sum(g.k2 .* abs2.(ψk_slice)) * g.dV / (g.Nx * g.Ny * g.Nz)
    end
    E_kin = sum(E_kin_parts)

    # トラップエネルギー
    E_trap = sum(g.trap .* ψ.n) * g.dV

    # 密度相互作用
    E_int = 0.5 * p.g0 / p.N * sum(ψ.n .^ 2) * g.dV

    # スピン相互作用
    F2 = ψ.Fx .^ 2 .+ ψ.Fy .^ 2 .+ ψ.Fz .^ 2
    E_spin = 0.5 * p.g1 / p.N * sum(F2) * g.dV

    # DDI
    E_dd = compute_ddi_energy(ψ, p, g)

    return E_kin, E_trap, E_int, E_spin, E_dd
end

function compute_ddi_energy(ψ::Spinor, p::Params, g::Grid)
    if abs(p.gdd) < 1e-10
        return 0.0
    end

    n_eff = ψ.n .- 3.0 .* ψ.Fz ./ SPIN_F
    nk = fft(n_eff)

    E_dd = 0.0
    @threads for k in 1:g.Nz
        E_local = 0.0
        for j in 1:g.Ny
            @simd for i in 1:g.Nx
                k2_val = g.k2[i,j,k]
                if k2_val > 1e-10
                    cos2_theta = g.kz2[i,j,k] / k2_val
                    Qk = 3*cos2_theta - 1
                    E_local += Qk * abs2(nk[i,j,k])
                end
            end
        end
        # アトミックな加算が必要だが、簡略化
    end

    # シリアル版にフォールバック
    E_dd = 0.0
    for k in 1:g.Nz, j in 1:g.Ny, i in 1:g.Nx
        k2_val = g.k2[i,j,k]
        if k2_val > 1e-10
            cos2_theta = g.kz2[i,j,k] / k2_val
            Qk = 3*cos2_theta - 1
            E_dd += Qk * abs2(nk[i,j,k])
        end
    end

    E_dd *= 0.5 * p.gdd / p.N * g.dV / (g.Nx * g.Ny * g.Nz)
    return real(E_dd)
end

# ============================================================
# 時間発展（並列）
# ============================================================

"""虚時間発展（マルチスレッド）"""
function evolve_imag!(ψ::Spinor, p::Params, g::Grid;
                      dt::Float64=1e-4, nsteps::Int=10000,
                      callback=nothing, callback_interval::Int=100)

    exp_K = exp.(-dt .* g.k2 ./ 2)

    # 各スレッド用の一時配列
    tmp_arrays = [zeros(ComplexF64, g.Nx, g.Ny, g.Nz) for _ in 1:NUM_COMPONENTS]

    for step in 1:nsteps
        # 運動エネルギー項（半ステップ）- 並列FFT
        @threads for idx in 1:NUM_COMPONENTS
            @views begin
                tmp_arrays[idx] .= ψ.ψ[:,:,:,idx]
                ψ.fft_plans[idx] * tmp_arrays[idx]
                tmp_arrays[idx] .*= exp_K
                ψ.ifft_plans[idx] * tmp_arrays[idx]
                ψ.ψ[:,:,:,idx] .= tmp_arrays[idx]
            end
        end

        # ポテンシャル項
        apply_potential_step_imag!(ψ, p, g, dt)

        # 運動エネルギー項（半ステップ）
        @threads for idx in 1:NUM_COMPONENTS
            @views begin
                tmp_arrays[idx] .= ψ.ψ[:,:,:,idx]
                ψ.fft_plans[idx] * tmp_arrays[idx]
                tmp_arrays[idx] .*= exp_K
                ψ.ifft_plans[idx] * tmp_arrays[idx]
                ψ.ψ[:,:,:,idx] .= tmp_arrays[idx]
            end
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
function evolve_real!(ψ::Spinor, p::Params, g::Grid;
                      dt::Float64=1e-4, nsteps::Int=1000,
                      callback=nothing, callback_interval::Int=100)

    exp_K = exp.(-im * dt .* g.k2 ./ 2)
    tmp_arrays = [zeros(ComplexF64, g.Nx, g.Ny, g.Nz) for _ in 1:NUM_COMPONENTS]

    for step in 1:nsteps
        @threads for idx in 1:NUM_COMPONENTS
            @views begin
                tmp_arrays[idx] .= ψ.ψ[:,:,:,idx]
                ψ.fft_plans[idx] * tmp_arrays[idx]
                tmp_arrays[idx] .*= exp_K
                ψ.ifft_plans[idx] * tmp_arrays[idx]
                ψ.ψ[:,:,:,idx] .= tmp_arrays[idx]
            end
        end

        apply_potential_step_real!(ψ, p, g, dt)

        @threads for idx in 1:NUM_COMPONENTS
            @views begin
                tmp_arrays[idx] .= ψ.ψ[:,:,:,idx]
                ψ.fft_plans[idx] * tmp_arrays[idx]
                tmp_arrays[idx] .*= exp_K
                ψ.ifft_plans[idx] * tmp_arrays[idx]
                ψ.ψ[:,:,:,idx] .= tmp_arrays[idx]
            end
        end

        if callback !== nothing && step % callback_interval == 0
            callback(step, ψ, p, g)
        end
    end
end

"""ポテンシャル項（虚時間、並列）"""
function apply_potential_step_imag!(ψ::Spinor, p::Params, g::Grid, dt::Float64)
    compute_density!(ψ.n, ψ.ψ)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    compute_ddi_potential!(ψ, p, g)

    g0_N = p.g0 / p.N
    g1_N = p.g1 / p.N

    # 対角成分（並列）
    @threads for idx in 1:NUM_COMPONENTS
        m = idx_to_m(idx)
        # ddi_factor removed - using uniform DDI for all m components
        @views begin
            for k in 1:g.Nz, j in 1:g.Ny
                @simd for i in 1:g.Nx
                    V_eff = g.trap[i,j,k] + g0_N * ψ.n[i,j,k] +
                            g1_N * ψ.Fz[i,j,k] * m + ψ.Vdd[i,j,k]
                    ψ.ψ[i,j,k,idx] *= exp(-dt * V_eff)
                end
            end
        end
    end

    # スピン混合（非対角）
    if abs(p.g1) > 1e-10
        apply_spin_mixing!(ψ, p, g, dt)
    end
end

"""ポテンシャル項（実時間、並列）"""
function apply_potential_step_real!(ψ::Spinor, p::Params, g::Grid, dt::Float64)
    compute_density!(ψ.n, ψ.ψ)
    compute_spin_density!(ψ.Fx, ψ.Fy, ψ.Fz, ψ.ψ)
    compute_ddi_potential!(ψ, p, g)

    g0_N = p.g0 / p.N
    g1_N = p.g1 / p.N

    @threads for idx in 1:NUM_COMPONENTS
        m = idx_to_m(idx)
        # ddi_factor removed - using uniform DDI for all m components
        @views begin
            for k in 1:g.Nz, j in 1:g.Ny
                @simd for i in 1:g.Nx
                    V_eff = g.trap[i,j,k] + g0_N * ψ.n[i,j,k] +
                            g1_N * ψ.Fz[i,j,k] * m + ψ.Vdd[i,j,k]
                    ψ.ψ[i,j,k,idx] *= exp(-im * dt * V_eff)
                end
            end
        end
    end

    if abs(p.g1) > 1e-10
        apply_spin_mixing_real!(ψ, p, g, dt)
    end
end

"""スピン混合"""
function apply_spin_mixing!(ψ::Spinor, p::Params, g::Grid, dt::Float64)
    g1_N = p.g1 / p.N
    ψ_old = copy(ψ.ψ)

    @threads for idx in 1:(NUM_COMPONENTS-1)
        coeff = real(FP[idx, idx+1])
        if abs(coeff) > 1e-10
            @views begin
                for k in 1:g.Nz, j in 1:g.Ny
                    @simd for i in 1:g.Nx
                        Fm = (ψ.Fx[i,j,k] - im * ψ.Fy[i,j,k]) / 2
                        Fp = (ψ.Fx[i,j,k] + im * ψ.Fy[i,j,k]) / 2

                        ψ.ψ[i,j,k,idx] -= dt * g1_N * coeff * Fm * ψ_old[i,j,k,idx+1]
                        ψ.ψ[i,j,k,idx+1] -= dt * g1_N * coeff * Fp * ψ_old[i,j,k,idx]
                    end
                end
            end
        end
    end
end

function apply_spin_mixing_real!(ψ::Spinor, p::Params, g::Grid, dt::Float64)
    g1_N = p.g1 / p.N
    ψ_old = copy(ψ.ψ)

    @threads for idx in 1:(NUM_COMPONENTS-1)
        coeff = real(FP[idx, idx+1])
        if abs(coeff) > 1e-10
            @views begin
                for k in 1:g.Nz, j in 1:g.Ny
                    @simd for i in 1:g.Nx
                        Fm = (ψ.Fx[i,j,k] - im * ψ.Fy[i,j,k]) / 2
                        Fp = (ψ.Fx[i,j,k] + im * ψ.Fy[i,j,k]) / 2

                        ψ.ψ[i,j,k,idx] -= im * dt * g1_N * coeff * Fm * ψ_old[i,j,k,idx+1]
                        ψ.ψ[i,j,k,idx+1] -= im * dt * g1_N * coeff * Fp * ψ_old[i,j,k,idx]
                    end
                end
            end
        end
    end
end

"""DDIポテンシャル"""
function compute_ddi_potential!(ψ::Spinor, p::Params, g::Grid)
    # DDIポテンシャル Φ_dd を計算（密度 n のみ使用）
    # V_dd,m = (gdd/N) * (1 - 3m/F) * Φ_dd はポテンシャル適用時に処理
    if abs(p.gdd) < 1e-10
        fill!(ψ.Vdd, 0.0)
        return
    end

    # 密度のFFT
    nk = fft(ψ.n)

    gdd_N = p.gdd / p.N

    # k空間でDDIカーネルを適用
    @threads for k in 1:g.Nz
        for j in 1:g.Ny
            @simd for i in 1:g.Nx
                k2_val = g.k2[i,j,k]
                if k2_val > 1e-10
                    cos2_theta = g.kz2[i,j,k] / k2_val
                    nk[i,j,k] *= gdd_N * (3*cos2_theta - 1)
                else
                    nk[i,j,k] = 0.0
                end
            end
        end
    end

    # Φ_dd = IFFT(...)
    ψ.Vdd .= real.(ifft(nk))
end

end # module
