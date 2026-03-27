# Matsui et al., Science 391, 384-388 (2026)
# Dimensionless units: ℏ = m = ω_ref = 1, ω_ref = 2π × 110 Hz
using SpinorBEC

const EU_ω_ref    = 2π * 110.0
const EU_N_atoms  = 50_000
const EU_m_Eu     = Eu151.mass
const EU_a_ho     = sqrt(Units.HBAR / (EU_m_Eu * EU_ω_ref))
const EU_t_unit   = 1.0 / EU_ω_ref
const EU_a_s_dl   = Eu151.a0 / EU_a_ho
const EU_c_total  = 4π * EU_a_s_dl * EU_N_atoms
const EU_c0       = EU_c_total
const EU_c_dd     = EU_N_atoms * compute_c_dd(Eu151) / (Units.HBAR * EU_ω_ref * EU_a_ho^3)
const EU_λ_z      = 130.0 / 110.0
const EU_g_F      = Eu151.g_F
const EU_B_weak   = 2.6e-9
const EU_p_weak   = linear_zeeman_p(Eu151, EU_B_weak, EU_ω_ref)
const EU_ε_dd     = compute_a_dd(Eu151) / Eu151.a0

"""
Compute constrained interaction params for Eu151.
c₀ + F²c₁ = c_total, with c₁/c₀ = c1_ratio.
"""
function eu_interaction_params(c1_ratio::Float64)
    interaction_params_from_constraint(; c_total=EU_c_total, c1_ratio, F=6)
end
