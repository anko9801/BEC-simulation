"""
Gaussian beam with complex beam parameter q.

1/q = 1/R - i M²λ/(πw²)

At the waist (R=∞): q = i z_R where z_R = π w₀² / (M² λ).
"""
struct OpticalBeam
    q::ComplexF64
    wavelength::Float64   # m
    power::Float64        # W
    M2::Float64           # beam quality factor (ideal = 1)
end

function OpticalBeam(; wavelength::Float64, power::Float64, waist::Float64, M2::Float64=1.0)
    z_R = π * waist^2 / (M2 * wavelength)
    OpticalBeam(im * z_R, wavelength, power, M2)
end

function waist_radius(b::OpticalBeam)
    inv_q = 1.0 / b.q
    sqrt(-b.M2 * b.wavelength / (π * imag(inv_q)))
end

function rayleigh_length(b::OpticalBeam)
    π * waist_radius(b)^2 / (b.M2 * b.wavelength)
end

function radius_of_curvature(b::OpticalBeam)
    inv_q = 1.0 / b.q
    r = real(inv_q)
    abs(r) < 1e-30 ? Inf : 1.0 / r
end

function divergence_angle(b::OpticalBeam)
    b.M2 * b.wavelength / (π * waist_radius(b))
end

function peak_intensity(b::OpticalBeam)
    w = waist_radius(b)
    2 * b.power / (π * w^2)
end

"""
Beam intensity at transverse distance r_perp and axial distance z from current beam center.
I(r,z) = I₀ (w₀/w(z))² exp(-2r²/w(z)²)
"""
function beam_intensity(b::OpticalBeam, r_perp::Float64, z::Float64)
    w0 = waist_radius(b)
    z_R = rayleigh_length(b)
    wz = w0 * sqrt(1 + (z / z_R)^2)
    I0 = 2 * b.power / (π * w0^2)
    I0 * (w0 / wz)^2 * exp(-2 * r_perp^2 / wz^2)
end

# --- ABCD Matrix Propagation ---

"""Propagate beam through an ABCD matrix [A B; C D]."""
function propagate(beam::OpticalBeam, M::SMatrix{2,2,Float64})
    q_new = (M[1, 1] * beam.q + M[1, 2]) / (M[2, 1] * beam.q + M[2, 2])
    OpticalBeam(q_new, beam.wavelength, beam.power, beam.M2)
end

function propagate(beam::OpticalBeam, Ms::AbstractVector{<:SMatrix{2,2,Float64}})
    b = beam
    for M in Ms
        b = propagate(b, M)
    end
    b
end

abcd_free_space(d::Float64) = SMatrix{2,2,Float64}(1.0, 0.0, d, 1.0)

abcd_thin_lens(f::Float64) = SMatrix{2,2,Float64}(1.0, -1.0 / f, 0.0, 1.0)

abcd_curved_mirror(R::Float64) = SMatrix{2,2,Float64}(1.0, -2.0 / R, 0.0, 1.0)

abcd_flat_mirror() = SMatrix{2,2,Float64}(1.0, 0.0, 0.0, 1.0)

# --- Mode Overlap (Fiber Coupling) ---

"""
Coupling efficiency between two Gaussian beams (mode overlap).
η = |∫ E₁* E₂ dA|² / (∫|E₁|²dA × ∫|E₂|²dA)

For two Gaussian beams with waists w₁, w₂ and lateral offset Δr:
η = (2w₁w₂/(w₁²+w₂²))² × exp(-2Δr²/(w₁²+w₂²))
"""
function mode_overlap(beam1::OpticalBeam, beam2::OpticalBeam; lateral_offset::Float64=0.0)
    w1 = waist_radius(beam1)
    w2 = waist_radius(beam2)
    w_sum_sq = w1^2 + w2^2
    spatial = (2 * w1 * w2 / w_sum_sq)^2 * exp(-2 * lateral_offset^2 / w_sum_sq)
    spatial
end

"""
Coupling efficiency into a single-mode fiber with given mode field diameter.
"""
function fiber_coupling(beam::OpticalBeam, fiber_mfd::Float64; lateral_offset::Float64=0.0)
    w_fiber = fiber_mfd / 2
    fiber_beam = OpticalBeam(; wavelength=beam.wavelength, power=1.0, waist=w_fiber, M2=1.0)
    mode_overlap(beam, fiber_beam; lateral_offset)
end
