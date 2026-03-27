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

const Eu151 = AtomSpecies(
    "151Eu",
    150.919857 * Units.AMU,
    6,
    110.0 * Units.BOHR_RADIUS,  # a_s (s-wave scattering length)
    0.0,                         # a2 unused (use c0 + DDI only)
    7.0 * Units.MU_BOHR,        # μ = 7 μ_B
    7.0 / 6.0,                  # g_F = 7/6 (⁸S₇/₂, F=6)
)
