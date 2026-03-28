using StaticArrays
using LinearAlgebra
using FFTW

# --- Grid Configuration ---

struct GridConfig{N}
    n_points::NTuple{N,Int}
    box_size::NTuple{N,Float64}

    function GridConfig{N}(n_points::NTuple{N,Int}, box_size::NTuple{N,Float64}) where {N}
        all(n -> n > 0 && iseven(n), n_points) || throw(ArgumentError("n_points must be positive even integers"))
        all(L -> L > 0, box_size) || throw(ArgumentError("box_size must be positive"))
        new{N}(n_points, box_size)
    end
end

GridConfig(n_points::NTuple{N,Int}, box_size::NTuple{N,Float64}) where {N} = GridConfig{N}(n_points, box_size)
GridConfig(n_points::Int, box_size::Float64) = GridConfig{1}((n_points,), (box_size,))

spatial_dims(::GridConfig{N}) where {N} = N

# --- Spatial Grid ---

struct Grid{N}
    config::GridConfig{N}
    x::NTuple{N,Vector{Float64}}
    dx::NTuple{N,Float64}
    k::NTuple{N,Vector{Float64}}
    dk::NTuple{N,Float64}
    k_squared::Array{Float64,N}
end

# --- Spin System ---

struct SpinSystem
    F::Int
    n_components::Int
    m_values::Vector{Int}
end

function SpinSystem(F::Int)
    F >= 0 || throw(ArgumentError("F must be non-negative"))
    n = 2F + 1
    SpinSystem(F, n, collect(F:-1:-F))
end

# --- Spin Matrices ---

struct SpinMatrices{D,M<:SMatrix}
    Fx::M
    Fy::M
    Fz::M
    Fp::M
    Fm::M
    F_dot_F::M
    system::SpinSystem
    Fy_eigvecs::Matrix{ComplexF64}
    Fy_eigvecs_adj::Matrix{ComplexF64}
    Fy_eigvals::SVector{D,Float64}
end

# --- Atom Species ---

"""
    AtomSpecies

Atomic species for spinor BEC simulation.

# Fields
- `name`: human-readable name (e.g. "87Rb")
- `mass`: atomic mass in kg
- `F`: total spin quantum number
- `a0`: F=1: F_tot=0 scattering length (m). F>1: mean s-wave scattering length a_s (m)
         when channel-resolved data is unavailable (e.g. Eu151). Use `a_s` for the
         unambiguous mean scattering length regardless of F.
- `a2`: F_tot=2 scattering length (m). Zero if unknown.
- `a_s`: mean s-wave scattering length (m). For F=1: (a0+2a2)/3. For F>1: same as `a0`.
- `mu_mag`: magnetic dipole moment (J/T). Zero for non-dipolar atoms.
- `g_F`: Landé g-factor
- `scattering_lengths`: Dict{Int,Float64} mapping total spin S => a_S (m).
                         Empty when channel-resolved data is unavailable.
- `Delta_E_hf`: hyperfine splitting (J). Zero if unknown/not applicable.
"""
struct AtomSpecies
    name::String
    mass::Float64
    F::Int
    a0::Float64
    a2::Float64
    a_s::Float64
    mu_mag::Float64
    g_F::Float64
    scattering_lengths::Dict{Int,Float64}
    Delta_E_hf::Float64

    function AtomSpecies(name, mass, F, a0, a2, mu_mag, g_F, scattering_lengths;
                         Delta_E_hf::Float64=0.0)
        a_s = F == 1 ? (a0 + 2a2) / 3 : a0
        new(name, mass, F, a0, a2, a_s, mu_mag, g_F, scattering_lengths, Delta_E_hf)
    end

    function AtomSpecies(name, mass, F, a0, a2, mu_mag, g_F::Real;
                         Delta_E_hf::Float64=0.0)
        sl = if F == 1 && (a0 != 0.0 || a2 != 0.0)
            Dict{Int,Float64}(0 => a0, 2 => a2)
        else
            Dict{Int,Float64}()
        end
        a_s = F == 1 ? (a0 + 2a2) / 3 : a0
        new(name, mass, F, a0, a2, a_s, mu_mag, Float64(g_F), sl, Delta_E_hf)
    end

    function AtomSpecies(name, mass, F, a0, a2, mu_mag, scattering_lengths::Dict;
                         Delta_E_hf::Float64=0.0)
        a_s = F == 1 ? (a0 + 2a2) / 3 : a0
        new(name, mass, F, a0, a2, a_s, mu_mag, 0.0, scattering_lengths, Delta_E_hf)
    end

    function AtomSpecies(name, mass, F, a0, a2, mu_mag; Delta_E_hf::Float64=0.0)
        sl = if F == 1 && (a0 != 0.0 || a2 != 0.0)
            Dict{Int,Float64}(0 => a0, 2 => a2)
        else
            Dict{Int,Float64}()
        end
        a_s = F == 1 ? (a0 + 2a2) / 3 : a0
        new(name, mass, F, a0, a2, a_s, mu_mag, 0.0, sl, Delta_E_hf)
    end
end

AtomSpecies(name, mass, F, a0, a2) = AtomSpecies(name, mass, F, a0, a2, 0.0)

# --- Interaction Parameters ---

"""
    InteractionParams(c0, c1, [c_lhy], [c_extra])

Contact interaction parameters. `c0` is the density coupling, `c1` the spin coupling.

`c_extra` stores higher-rank couplings: `c_extra[n-1]` = cₙ for n ≥ 2.
Access via `get_cn(ip, n)`. When any even-rank c_extra entry with k ≥ 4 is nonzero,
`make_workspace` builds a `TensorInteractionCache` and zeros c0/c1 (all contact
interactions are then handled by the tensor step).
"""
struct InteractionParams
    c0::Float64
    c1::Float64
    c_lhy::Float64
    c_extra::Vector{Float64}

    InteractionParams(c0::Float64, c1::Float64) = new(c0, c1, 0.0, Float64[])
    InteractionParams(c0::Float64, c1::Float64, c_extra::Vector{Float64}) = new(c0, c1, 0.0, c_extra)
    InteractionParams(c0::Float64, c1::Float64, c_lhy::Float64) = new(c0, c1, c_lhy, Float64[])
    InteractionParams(c0::Float64, c1::Float64, c_lhy::Float64, c_extra::Vector{Float64}) = new(c0, c1, c_lhy, c_extra)
end

function get_cn(ip::InteractionParams, n::Int)
    n == 0 && return ip.c0
    n == 1 && return ip.c1
    idx = n - 1
    idx <= length(ip.c_extra) ? ip.c_extra[idx] : 0.0
end

# --- Zeeman Parameters ---

struct ZeemanParams
    p::Float64      # linear Zeeman (energy)
    q::Float64      # quadratic Zeeman (energy)
end

ZeemanParams() = ZeemanParams(0.0, 0.0)

struct TimeDependentZeeman
    B_func::Function  # t -> ZeemanParams
end

# --- Raman Coupling ---

struct RamanCoupling{N}
    Omega_R::Float64          # Rabi frequency
    delta::Float64            # two-photon detuning
    k_eff::NTuple{N,Float64}  # effective wave vector (difference of two beams)
end

# --- Potential ---

abstract type AbstractPotential end

struct HarmonicTrap{N} <: AbstractPotential
    omega::NTuple{N,Float64}
end

HarmonicTrap(omega::Float64) = HarmonicTrap((omega,))
HarmonicTrap(ox::Float64, oy::Float64) = HarmonicTrap((ox, oy))
HarmonicTrap(ox::Float64, oy::Float64, oz::Float64) = HarmonicTrap((ox, oy, oz))

struct NoPotential <: AbstractPotential end

struct GravityPotential{N} <: AbstractPotential
    g::Float64
    axis::Int

    function GravityPotential{N}(g::Float64, axis::Int) where {N}
        1 <= axis <= N || throw(ArgumentError("axis must be between 1 and $N"))
        new{N}(g, axis)
    end
end

GravityPotential(g::Float64, axis::Int, ndim::Int) = GravityPotential{ndim}(g, axis)

struct CompositePotential{N} <: AbstractPotential
    components::Vector{AbstractPotential}
end

# --- Simulation Parameters ---

struct SimParams
    dt::Float64
    n_steps::Int
    imaginary_time::Bool
    normalize_every::Int
    save_every::Int
end

function SimParams(;
    dt::Float64,
    n_steps::Int,
    imaginary_time::Bool=false,
    normalize_every::Int=imaginary_time ? 1 : 0,
    save_every::Int=max(1, n_steps ÷ 100),
)
    dt > 0 || throw(ArgumentError("dt must be positive"))
    n_steps > 0 || throw(ArgumentError("n_steps must be positive"))
    SimParams(dt, n_steps, imaginary_time, normalize_every, save_every)
end

# --- Simulation State (mutable) ---

mutable struct SimState{N,A<:AbstractArray}
    psi::A              # wavefunction: spatial dims... × n_components
    fft_buf::Array{ComplexF64,N}  # spatial-only buffer for FFT
    t::Float64
    step::Int
end

# --- FFT Plans ---

struct FFTPlans{P,IP}
    forward::P
    inverse::IP
end

# --- rFFT Plans (for DDI on real-valued spin density) ---

struct RFFTPlans{N,RP,IRP}
    forward::RP
    inverse::IRP
    rk_shape::NTuple{N,Int}
end

# --- DDI ---

struct DDIParams{N}
    C_dd::Float64
    Q_xx::Array{Float64,N}
    Q_xy::Array{Float64,N}
    Q_xz::Array{Float64,N}
    Q_yy::Array{Float64,N}
    Q_yz::Array{Float64,N}
    Q_zz::Array{Float64,N}
end

struct DDIBuffers{N,RP,IRP}
    rfft_plans::RFFTPlans{N,RP,IRP}
    Fx_r::Array{Float64,N}
    Fy_r::Array{Float64,N}
    Fz_r::Array{Float64,N}
    Fx_rk::Array{ComplexF64,N}
    Fy_rk::Array{ComplexF64,N}
    Fz_rk::Array{ComplexF64,N}
    Phi_x_rk::Array{ComplexF64,N}
    Phi_y_rk::Array{ComplexF64,N}
    Phi_z_rk::Array{ComplexF64,N}
    Phi_x::Array{Float64,N}
    Phi_y::Array{Float64,N}
    Phi_z::Array{Float64,N}
end

# --- Loss Parameters ---

struct LossParams
    gamma_dr::Float64
    L3::Float64
end

LossParams(gamma_dr::Float64) = LossParams(gamma_dr, 0.0)

# --- DDI Padded Context ---

struct DDIPaddedContext{N,RP,IRP}
    padded_shape::NTuple{N,Int}
    rfft_plans::RFFTPlans{N,RP,IRP}
    Q_xx::Array{Float64,N}
    Q_xy::Array{Float64,N}
    Q_xz::Array{Float64,N}
    Q_yy::Array{Float64,N}
    Q_yz::Array{Float64,N}
    Q_zz::Array{Float64,N}
    Fx_pad::Array{Float64,N}
    Fy_pad::Array{Float64,N}
    Fz_pad::Array{Float64,N}
    Fx_pad_rk::Array{ComplexF64,N}
    Fy_pad_rk::Array{ComplexF64,N}
    Fz_pad_rk::Array{ComplexF64,N}
    Phi_x_pad_rk::Array{ComplexF64,N}
    Phi_y_pad_rk::Array{ComplexF64,N}
    Phi_z_pad_rk::Array{ComplexF64,N}
    Phi_x_pad::Array{Float64,N}
    Phi_y_pad::Array{Float64,N}
    Phi_z_pad::Array{Float64,N}
end

# --- Batched Kinetic Cache ---

struct BatchedKineticCache{P,IP}
    forward::P
    inverse::IP
    kinetic_phase_bc::Array{ComplexF64}
end

# --- Tensor Interaction Cache (general-F) ---

struct TensorInteractionCache
    F::Int
    D::Int
    cg_table::Dict{NTuple{4,Int},Float64}
    active_channels::Vector{Int}      # S values (even total spin channels)
    g_values::Vector{Float64}         # corresponding g_S coupling constants
end

# --- Adaptive Time Stepping ---

struct AdaptiveDtParams
    dt_init::Float64
    dt_min::Float64
    dt_max::Float64
    tol::Float64

    function AdaptiveDtParams(; dt_init::Float64=0.001, dt_min::Float64=1e-5,
                               dt_max::Float64=0.01, tol::Float64=1e-3)
        dt_init > 0 || throw(ArgumentError("dt_init must be positive"))
        dt_min > 0 || throw(ArgumentError("dt_min must be positive"))
        dt_max >= dt_min || throw(ArgumentError("dt_max must be >= dt_min"))
        tol > 0 || throw(ArgumentError("tol must be positive"))
        new(dt_init, dt_min, dt_max, tol)
    end
end

# --- Workspace ---

struct Workspace{N,A,P,IP,SM<:SpinMatrices,ZEE,DDI,DDIB,RAM,LOSS,DDIP,BK,TC}
    state::SimState{N,A}
    fft_plans::FFTPlans{P,IP}
    kinetic_phase::Array{ComplexF64,N}
    potential_values::Array{Float64,N}
    density_buf::Array{Float64,N}
    spin_matrices::SM
    grid::Grid{N}
    atom::AtomSpecies
    interactions::InteractionParams
    zeeman::ZEE
    potential::AbstractPotential
    sim_params::SimParams
    ddi::DDI
    ddi_bufs::DDIB
    raman::RAM
    loss::LOSS
    ddi_padded::DDIP
    batched_kinetic::BK
    tensor_cache::TC
end
