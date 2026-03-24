"""
Compute spin-1 interaction parameters c0 and c1 in SI units.

For spin-1:
  c0 = 4πℏ²(a0 + 2a2) / (3m)
  c1 = 4πℏ²(a2 - a0) / (3m)

c1 < 0 → ferromagnetic (87Rb)
c1 > 0 → antiferromagnetic (23Na)

For general F, this needs extension.
"""
function compute_interaction_params(atom::AtomSpecies; N_atoms::Int=1, dims::Int=1, length_scale::Float64=1.0)
    atom.F == 1 || throw(ArgumentError("Only F=1 supported for interaction params"))

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

    InteractionParams(c0, c1)
end

"""
Density-only contact interaction for general F.
c0 = 4πℏ² a_s / m (no spin channels, just s-wave scattering length a0).
"""
function compute_c0(atom::AtomSpecies; N_atoms::Int=1, dims::Int=1, length_scale::Float64=1.0)
    hbar = Units.HBAR
    c0_3d = 4π * hbar^2 * atom.a0 / atom.mass

    if dims == 1
        c0_3d / (2π * length_scale^2) * N_atoms
    elseif dims == 2
        c0_3d / (sqrt(2π) * length_scale) * N_atoms
    else
        c0_3d * N_atoms
    end
end

"""
DDI coupling constant: C_dd = μ₀ μ² / (4π).
Returns C_dd in SI (J·m³). Zero for non-dipolar atoms.
"""
function compute_c_dd(atom::AtomSpecies)
    atom.mu_mag == 0.0 && return 0.0
    Units.MU_0 * atom.mu_mag^2 / (4π)
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
