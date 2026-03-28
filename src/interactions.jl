"""
    compute_interaction_params(atom; N_atoms, dims, length_scale)

Compute interaction parameters in SI units from channel-resolved scattering lengths.

For F=1: uses analytic formulas c₀ = 4πℏ²(a₀+2a₂)/(3m), c₁ = 4πℏ²(a₂−a₀)/(3m).
For F≥2: requires `atom.scattering_lengths` dict (S => a_S for all even S channels).

For atoms where individual a_S are unknown (e.g. ¹⁵¹Eu), use
`interaction_params_from_constraint(; c_total, c1_ratio, F)` instead, which
constructs InteractionParams directly from the physical constraint c₀+F²c₁ = c_total.
"""
function compute_interaction_params(atom::AtomSpecies; N_atoms::Int=1, dims::Int=1, length_scale::Float64=1.0)
    if atom.F == 1
        a0, a2, m = atom.a0, atom.a2, atom.mass
        hbar = Units.HBAR

        c0_3d = 4π * hbar^2 * (a0 + 2a2) / (3m)
        c1_3d = 4π * hbar^2 * (a2 - a0) / (3m)

        if dims == 1
            l_perp = length_scale
            c0 = c0_3d / (2π * l_perp^2) * N_atoms
            c1 = c1_3d / (2π * l_perp^2) * N_atoms
        elseif dims == 2
            l_z = length_scale
            c0 = c0_3d / (sqrt(2π) * l_z) * N_atoms
            c1 = c1_3d / (sqrt(2π) * l_z) * N_atoms
        else
            c0 = c0_3d * N_atoms
            c1 = c1_3d * N_atoms
        end

        return InteractionParams(c0, c1)
    end

    if isempty(atom.scattering_lengths)
        @warn "No channel scattering lengths for F=$(atom.F) atom $(atom.name); using c0-only (c1=0)" maxlog=1
        c0 = compute_c0(atom; N_atoms, dims, length_scale)
        return InteractionParams(c0, 0.0)
    end
    compute_interaction_params_general_f(atom; N_atoms, dims, length_scale)
end

"""
    compute_interaction_params_general_f(atom; N_atoms, dims, length_scale)

Compute interaction params for general spin-F with channel-resolved scattering lengths.

Returns `InteractionParams(0.0, 0.0)` — all contact interactions are handled by the
tensor interaction step when scattering lengths are provided. The `g_S` values
are stored in `TensorInteractionCache`, not in `InteractionParams`.
"""
function compute_interaction_params_general_f(atom::AtomSpecies;
    N_atoms::Int=1, dims::Int=1, length_scale::Float64=1.0,
)
    InteractionParams(0.0, 0.0)
end

"""
Density-only contact interaction for general F.
c0 = 4πℏ² a_s / m (no spin channels, just s-wave scattering length a0).
"""
function compute_c0(atom::AtomSpecies; N_atoms::Int=1, dims::Int=1, length_scale::Float64=1.0)
    hbar = Units.HBAR
    c0_3d = 4π * hbar^2 * atom.a_s / atom.mass

    if dims == 1
        c0_3d / (2π * length_scale^2) * N_atoms
    elseif dims == 2
        c0_3d / (sqrt(2π) * length_scale) * N_atoms
    else
        c0_3d * N_atoms
    end
end

"""
DDI coupling constant: c_dd = μ₀ μ².
Returns c_dd in SI (J·m³). Zero for non-dipolar atoms.

Used with k-space kernel Q_αβ(k) = k̂_αk̂_β − δ_αβ/3, which is the Fourier
transform of (δ_αβ − 3r̂_αr̂_β)/(4πr³) — the 1/(4π) is absorbed into Q.
"""
function compute_c_dd(atom::AtomSpecies)
    atom.mu_mag == 0.0 && return 0.0
    Units.MU_0 * atom.mu_mag^2
end

"""
Dipolar length: a_dd = μ₀ μ² m / (12π ℏ²).
"""
function compute_a_dd(atom::AtomSpecies)
    atom.mu_mag == 0.0 && return 0.0
    Units.MU_0 * atom.mu_mag^2 * atom.mass / (12π * Units.HBAR^2)
end

function compute_interaction_params_dimless(atom::AtomSpecies; N_atoms::Int=1, dims::Int=1, omega::Float64=1.0)
    hbar = Units.HBAR
    m = atom.mass
    a_ho = sqrt(hbar / (m * omega))

    params_si = compute_interaction_params(atom; N_atoms, dims, length_scale=a_ho)

    energy_scale = hbar * omega
    InteractionParams(params_si.c0 / energy_scale, params_si.c1 / energy_scale)
end

"""
    _c0c1_to_gS(F, c0, c1) → Dict{Int,Float64}

Convert physical density (c₀) and spin (c₁) couplings to channel couplings g_S:
  g_S = c₀ + c₁(S(S+1) − 2F(F+1))/2

This is the physical relation, NOT the 6j tensor transform. It gives:
- F=1: g₀ = c₀ − 2c₁, g₂ = c₀ + c₁
- General F: g_S for all even S ∈ 0:2:2F
"""
function _c0c1_to_gS(F::Int, c0::Float64, c1::Float64)
    Dict{Int,Float64}(
        S => c0 + c1 * (S * (S + 1) - 2 * F * (F + 1)) / 2
        for S in 0:2:2F
    )
end

"""
    _c_extra_to_delta_gS(F, c_extra) → Dict{Int,Float64}

Convert higher-rank tensor couplings c_k to channel coupling perturbations δg_S
via 6j transform.

Processes even-rank entries k ∈ {2, 4, ..., 2F} from `c_extra`, where
`c_extra[idx]` = c_{idx+1} (i.e. c_extra[1]=c₂, c_extra[3]=c₄, c_extra[5]=c₆).
Odd-rank and zero entries are skipped.
"""
function _c_extra_to_delta_gS(F::Int, c_extra::Vector{Float64})
    c_dict = Dict{Int,Float64}()
    for (idx, val) in enumerate(c_extra)
        k = idx + 1
        abs(val) > 1e-30 && iseven(k) && k <= 2F && (c_dict[k] = val)
    end
    isempty(c_dict) && return Dict{Int,Float64}()
    _cn_to_gS(F, c_dict)
end

"""
    interaction_params_from_constraint(; c_total, c1_ratio, F, c_extra)

Compute c₀, c₁ satisfying the physical constraint c₀ + F²c₁ = c_total.

For atoms where individual scattering lengths a_S are unknown (e.g. ¹⁵¹Eu),
the total contact interaction c_total = 4π(a_s/a_ho)N is known but the
spin-dependent split c₁/c₀ is a free parameter. This function parameterizes
by ratio r = c₁/c₀:

  c₀ = c_total / (1 + F²r)
  c₁ = r × c₀

The optional `c_extra` vector provides higher-rank tensor couplings where
`c_extra[n-1]` = cₙ (same indexing as `InteractionParams`). When any even-rank
entry with k ≥ 4 is nonzero, `make_workspace` activates the tensor interaction
path and zeros c₀/c₁.

Example with c₄ = 50:

    ip = interaction_params_from_constraint(;
        c_total=4689.0, c1_ratio=1/36, F=6,
        c_extra=[0.0, 0.0, 50.0])  # c_extra[3] = c₄

Note: r = -1/F² is singular (c₀ → ∞). For F=6, avoid r ≤ -1/36.
"""
function interaction_params_from_constraint(; c_total::Float64, c1_ratio::Float64=0.0,
                                              F::Int, c_extra::Vector{Float64}=Float64[])
    c0 = c_total / (1.0 + F^2 * c1_ratio)
    c1 = c1_ratio * c0
    InteractionParams(c0, c1, 0.0, c_extra)
end

"""
    compute_c_total(atom; N_atoms, omega_ref)

Total contact interaction c_total = 4π(a_s/a_ho)N in dimensionless units (3D).
"""
function compute_c_total(atom::AtomSpecies; N_atoms::Int, omega_ref::Float64)
    a_ho = sqrt(Units.HBAR / (atom.mass * omega_ref))
    4π * (atom.a_s / a_ho) * N_atoms
end

"""
    compute_c_dd_dimless(atom; N_atoms, omega_ref)

Dimensionless DDI coupling: c_dd = N × μ₀μ² / (ℏω × a_ho³).
"""
function compute_c_dd_dimless(atom::AtomSpecies; N_atoms::Int, omega_ref::Float64)
    a_ho = sqrt(Units.HBAR / (atom.mass * omega_ref))
    N_atoms * compute_c_dd(atom) / (Units.HBAR * omega_ref * a_ho^3)
end

"""
    linear_zeeman_p(atom, B, omega_ref)

Dimensionless linear Zeeman shift: p = g_F × μ_B × B / (ℏ × omega_ref).
"""
function linear_zeeman_p(atom::AtomSpecies, B::Float64, omega_ref::Float64)
    atom.g_F * Units.MU_BOHR * B / (Units.HBAR * omega_ref)
end
