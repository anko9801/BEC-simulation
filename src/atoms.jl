const Rb87 = AtomSpecies(
    "87Rb",
    86.909180527 * Units.AMU,
    1,
    101.8 * Units.BOHR_RADIUS,   # a0 (F_tot=0)
    100.4 * Units.BOHR_RADIUS,   # a2 (F_tot=2)
    0.0,                          # non-dipolar
    -0.5,                         # g_F (F=1 ground state)
)

const Na23 = AtomSpecies(
    "23Na",
    22.9897692820 * Units.AMU,
    1,
    50.0 * Units.BOHR_RADIUS,    # a0
    55.0 * Units.BOHR_RADIUS,    # a2
    0.0,                          # non-dipolar
    -0.5,                         # g_F (F=1 ground state)
)

# ¹⁵¹Eu: ⁸S₇/₂ ground state (J=7/2, I=5/2, F=6)
#   g_J = 1.9934 (NIST ASD), g_F = g_J × [F(F+1)+J(J+1)−I(I+1)] / [2F(F+1)] = g_J × 7/12
#   μ = g_J × J × μ_B = 1.9934 × 3.5 × μ_B ≈ 6.977 μ_B
#
# scattering_lengths is intentionally empty: individual a_S (S=0,2,...,12) are
# experimentally unknown for ¹⁵¹Eu (Matsui et al. 2026). Only the mean s-wave
# scattering length a_s ≈ 110 a₀ is known. Use interaction_params_from_constraint()
# to specify c₀+c₁ under the physical constraint c₀ + F²c₁ = 4π(a_s/a_ho)N.
const _EU151_G_J = 1.9934
const Eu151 = AtomSpecies(
    "151Eu",
    150.919857 * Units.AMU,
    6,
    110.0 * Units.BOHR_RADIUS,       # a_s (mean s-wave scattering length)
    0.0,                              # a2 unused — scattering channels unknown
    _EU151_G_J * 3.5 * Units.MU_BOHR,  # μ = g_J × J × μ_B ≈ 6.977 μ_B
    _EU151_G_J * 7.0 / 12.0,           # g_F ≈ 1.1628
)
