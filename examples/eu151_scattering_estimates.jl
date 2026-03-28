# Theoretical estimates for ¹⁵¹Eu scattering parameters
#
# Sources:
#   - Buchachenko et al. (2007) PRA 75, 042903: ab initio Eu-Eu potentials
#   - Frisch et al. (2014) PRL 115, 173003: Eu Feshbach spectroscopy
#   - Matsui et al. (2026) Science 391, 384: Eu BEC with EdH
#
# The individual channel scattering lengths a_S (S=0,2,...,12) are NOT
# experimentally resolved for ¹⁵¹Eu. Only the mean a_s ≈ 110 a₀ is known.
#
# Buchachenko's ab initio calculations provide:
#   - The dominant constraint: c₀ + F²c₁ = c_total (from a_s)
#   - Rough estimate: c₁/c₀ ≈ +1/36 (antiferromagnetic)
#   - Higher-rank couplings c₄, c₆, ..., c₁₂ are expected to be small
#     relative to c₀ but may affect ground state phase selection
#
# This script demonstrates how to use these estimates with the
# interaction_params_from_constraint API.

include(joinpath(@__DIR__, "eu151_params.jl"))
using Printf

println("=" ^ 60)
println("  ¹⁵¹Eu Scattering Parameter Estimates")
println("=" ^ 60)

# Known: a_s = 110 a₀, giving c_total = 4π(a_s/a_ho)N
@printf("  a_s = 110 a₀\n")
@printf("  c_total = %.1f (dimless)\n", EU_c_total)
@printf("  c_dd = %.1f (dimless)\n", EU_c_dd)
@printf("  ε_dd = %.3f\n\n", EU_ε_dd)

# Buchachenko best-fit estimate: r = c₁/c₀ ≈ +1/36
# This gives antiferromagnetic spin interaction
r_buchachenko = 1.0 / 36.0
ip_buch = eu_interaction_params(r_buchachenko)
@printf("Buchachenko estimate (r=+1/36, AFM):\n")
@printf("  c₀ = %.1f, c₁ = %+.1f\n", ip_buch.c0, ip_buch.c1)
@printf("  c₀ + F²c₁ = %.1f ✓\n\n", ip_buch.c0 + 36 * ip_buch.c1)

# Scan physical scenarios
scenarios = [
    ("DDI-only (c₁=0)",          0.0),
    ("Weak AFM (c₁/c₀=+1/72)",  +1.0 / 72.0),
    ("Buchachenko AFM (+1/36)",  +1.0 / 36.0),
    ("Weak FM (c₁/c₀=-1/72)",   -1.0 / 72.0),
    ("Strong FM (c₁/c₀=-1/36)", -1.0 / 36.0),
]

println("Scenario comparison:")
@printf("%-30s | %8s | %8s | %8s\n", "Scenario", "c₀", "c₁", "c₁/c_dd")
println("-" ^ 65)
for (label, r) in scenarios
    ip = eu_interaction_params(r)
    @printf("%-30s | %8.1f | %+8.1f | %+8.4f\n", label, ip.c0, ip.c1, ip.c1 / EU_c_dd)
end

# Higher-rank coupling estimates
# For F=6, even ranks k=2,4,6,8,10,12 contribute to g_S via 6j symbols.
# Without experimental data, we can only estimate their order of magnitude.
println("\n" * "=" ^ 60)
println("  Higher-Rank Coupling Estimates")
println("=" ^ 60)
println("""
For F=6, the contact interaction has channels S=0,2,...,12.
The coupling constants c_k (k=0,1,...,12) relate to g_S via:

  g_S = Σ_k (2k+1) {F F k; F F S} c_k

The c₀ and c₁ terms dominate. Higher-rank couplings c₄, c₆, ...
arise from anisotropy in the interaction potential.

Without experimental a_S values, we parameterize:
  c_extra = [c₂, c₃, c₄, c₅, c₆, ...]
  where c_extra[n-1] = cₙ

To explore the effect of c₄:
""")

ip_c4 = interaction_params_from_constraint(;
    c_total=EU_c_total, c1_ratio=r_buchachenko, F=6,
    c_extra=[0.0, 0.0, 50.0],  # c₄ = 50
)
@printf("  With c₄=50: c₀=%.1f, c₁=%+.1f, c_extra[3]=%.1f\n", ip_c4.c0, ip_c4.c1, ip_c4.c_extra[3])

# Channel coupling strengths from c₀+c₁ only (Buchachenko)
g_S = SpinorBEC._c0c1_to_gS(6, ip_buch.c0, ip_buch.c1)
println("\nChannel couplings g_S (Buchachenko, c₀+c₁ only):")
for S in 0:2:12
    @printf("  g_{%2d} = %+8.1f\n", S, g_S[S])
end
