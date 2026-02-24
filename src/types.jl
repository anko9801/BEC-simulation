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
end

# --- Interaction Parameters ---

struct InteractionParams
    c0::Float64
    c1::Float64
end

# --- Zeeman Parameters ---

struct ZeemanParams
    p::Float64      # linear Zeeman (energy)
    q::Float64      # quadratic Zeeman (energy)
end

ZeemanParams() = ZeemanParams(0.0, 0.0)

# --- Potential ---

abstract type AbstractPotential end

struct HarmonicTrap{N} <: AbstractPotential
    omega::NTuple{N,Float64}
end

HarmonicTrap(omega::Float64) = HarmonicTrap((omega,))
HarmonicTrap(ox::Float64, oy::Float64) = HarmonicTrap((ox, oy))

struct NoPotential <: AbstractPotential end

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
    zeeman::ZeemanParams
    potential::AbstractPotential
    sim_params::SimParams
end
