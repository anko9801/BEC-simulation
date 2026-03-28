# Self-bound droplet with LHY correction
#
# Near mean-field collapse (c₀ < 0, |c₀| small), the LHY beyond-mean-field
# correction stabilizes a self-bound quantum droplet.
#
# The total energy density (1D effective, F=1 polar):
#   E/V = (c₀/2)n² + (2/5)c_lhy n^{5/2}
#
# Equilibrium density: n_eq = (|c₀| / c_lhy)^2  (in appropriate units)
# Self-bound when c₀ < 0 and c_lhy > 0.
#
# This script finds the droplet ground state by ITP without an external trap
# (box boundary conditions), then verifies the density plateau matches n_eq.
#
# Usage:
#   julia --project=. examples/droplet_lhy.jl
#   C0=-5.0 C_LHY=0.5 julia --project=. examples/droplet_lhy.jl

using SpinorBEC
using Printf

c0 = parse(Float64, get(ENV, "C0", "-5.0"))
c_lhy = parse(Float64, get(ENV, "C_LHY", "0.5"))
N_target = parse(Float64, get(ENV, "N_TARGET", "100.0"))

@assert c0 < 0 "Droplet requires c₀ < 0"
@assert c_lhy > 0 "Droplet requires c_lhy > 0"

println("=" ^ 60)
println("  Quantum Droplet with LHY Correction (1D)")
println("=" ^ 60)
@printf("  c₀ = %.2f (attractive)\n", c0)
@printf("  c_lhy = %.4f\n", c_lhy)
@printf("  N_target = %.0f\n\n", N_target)

n_eq = (abs(c0) / c_lhy)^2
@printf("  Predicted n_eq = (|c₀|/c_lhy)² = %.4f\n", n_eq)

gc = GridConfig((512,), (80.0,))
grid = make_grid(gc)
dV = cell_volume(grid)
atom = AtomSpecies("droplet", 1.0, 1, 0.0, 0.0)

ip = InteractionParams(c0, 0.0, c_lhy)

psi0 = zeros(ComplexF64, 512, 3)
x = grid.x[1]
sigma = 5.0
for i in 1:512
    psi0[i, 2] = sqrt(n_eq) * exp(-x[i]^2 / (2 * sigma^2))
end
norm0 = sqrt(sum(abs2, psi0) * dV)
psi0 .*= sqrt(N_target) / norm0

@printf("  Initial norm = %.4f\n", sum(abs2, psi0) * dV)

println("\nFinding droplet ground state (ITP, no trap)...")
result = find_ground_state(;
    grid, atom,
    interactions=ip,
    potential=NoPotential(),
    dt=0.001, n_steps=20000, tol=1e-10,
    psi_init=psi0,
)

@printf("  converged = %s\n", result.converged)
@printf("  energy = %.6f\n", result.energy)

psi = result.workspace.state.psi
n_final = SpinorBEC.total_density(psi, 1)
n_peak = maximum(n_final)
norm_final = sum(abs2, psi) * dV

@printf("  norm = %.4f\n", norm_final)
@printf("  n_peak = %.4f (predicted n_eq = %.4f)\n", n_peak, n_eq)

# Check plateau: density should be flat near center
center = length(x) ÷ 2
plateau_region = (center - 20):(center + 20)
n_plateau = [n_final[i] for i in plateau_region]
n_mean = sum(n_plateau) / length(n_plateau)
n_std = sqrt(sum((n_plateau .- n_mean).^2) / length(n_plateau))

@printf("  Plateau: n_mean = %.4f, n_std = %.2e\n", n_mean, n_std)
@printf("  |n_mean - n_eq| / n_eq = %.2e\n", abs(n_mean - n_eq) / n_eq)

# Droplet width: FWHM
half_max = n_peak / 2
left_idx = findfirst(>(half_max), n_final)
right_idx = findlast(>(half_max), n_final)
if left_idx !== nothing && right_idx !== nothing
    fwhm = x[right_idx] - x[left_idx]
    @printf("  FWHM = %.2f\n", fwhm)
end

println("\nDensity profile (sampled):")
@printf("%8s | %10s\n", "x", "n(x)")
println("-" ^ 22)
for i in 1:16:512
    @printf("%+7.2f  | %10.4f\n", x[i], n_final[i])
end
