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
end

# --- Atom Species ---

struct AtomSpecies
    name::String
    mass::Float64       # kg
    F::Int
    a0::Float64         # m (F_tot=0 scattering length)
    a2::Float64         # m (F_tot=2 scattering length)
    mu_mag::Float64     # J/T (magnetic dipole moment, 0.0 for non-dipolar)
end

AtomSpecies(name, mass, F, a0, a2) = AtomSpecies(name, mass, F, a0, a2, 0.0)

# --- Interaction Parameters ---

struct InteractionParams
    c0::Float64
    c1::Float64
    c_extra::Vector{Float64}

    InteractionParams(c0::Float64, c1::Float64) = new(c0, c1, Float64[])
    InteractionParams(c0::Float64, c1::Float64, c_extra::Vector{Float64}) = new(c0, c1, c_extra)
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

struct DDIBuffers{N}
    Fx_r::Array{Float64,N}
    Fy_r::Array{Float64,N}
    Fz_r::Array{Float64,N}
    Fk::Array{ComplexF64,N}
    Phi_x::Array{ComplexF64,N}
    Phi_y::Array{ComplexF64,N}
    Phi_z::Array{ComplexF64,N}
end

# --- Workspace ---

struct Workspace{N,A,P,IP}
    state::SimState{N,A}
    fft_plans::FFTPlans{P,IP}
    kinetic_phase::Array{ComplexF64,N}
    potential_values::Array{Float64,N}
    spin_matrices::SpinMatrices
    grid::Grid{N}
    atom::AtomSpecies
    interactions::InteractionParams
    zeeman::Union{ZeemanParams,TimeDependentZeeman}
    potential::AbstractPotential
    sim_params::SimParams
    ddi::Union{Nothing,DDIParams{N}}
    ddi_bufs::Union{Nothing,DDIBuffers{N}}
    raman::Union{Nothing,RamanCoupling{N}}
end
