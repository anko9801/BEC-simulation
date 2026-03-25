# Matsui et al., Science 391, 384-388 (2026)
# Dimensionless units: ℏ = m = ω_ref = 1, ω_ref = 2π × 110 Hz
using SpinorBEC

const EU_ω_ref    = 2π * 110.0
const EU_N_atoms  = 50_000
const EU_m_Eu     = Eu151.mass
const EU_a_ho     = sqrt(Units.HBAR / (EU_m_Eu * EU_ω_ref))
const EU_t_unit   = 1.0 / EU_ω_ref
const EU_a_s_dl   = Eu151.a0 / EU_a_ho
const EU_c0       = 4π * EU_a_s_dl * EU_N_atoms
const EU_c_dd     = EU_N_atoms * compute_c_dd(Eu151) / (Units.HBAR * EU_ω_ref * EU_a_ho^3)
const EU_λ_z      = 130.0 / 110.0
const EU_g_F      = 7.0 / 6.0
const EU_B_weak   = 2.6e-9
const EU_p_weak   = EU_g_F * Units.MU_BOHR * EU_B_weak / (Units.HBAR * EU_ω_ref)
const EU_ε_dd     = compute_a_dd(Eu151) / Eu151.a0
