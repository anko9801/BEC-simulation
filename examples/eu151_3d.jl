using SpinorBEC
using Printf

println("=== 3D Ground State: Contact vs Contact+DDI ===\n")

# --- Eu151 physical constants ---
atom = Eu151
a_dd = compute_a_dd(atom)
eps_dd = a_dd / atom.a0
println("Eu151: F=$(atom.F), μ=$(atom.mu_mag / Units.MU_BOHR) μ_B")
println("  a_s  = $(@sprintf("%.1f", atom.a0 / Units.BOHR_RADIUS)) a₀")
println("  a_dd = $(@sprintf("%.1f", a_dd / Units.BOHR_RADIUS)) a₀")
println("  ε_dd = $(@sprintf("%.3f", eps_dd))")
println()

# --- Dimensionless simulation with spin-1 (fast, 3 components) ---
# Demonstrates DDI physics: magnetostriction (elongation along B = z)
# Using Rb87 (spin-1) with artificial DDI strength matching ε_dd ≈ 0.55
atom_s1 = Rb87
N = 16
L = 12.0  # box in a_ho
grid = make_grid(GridConfig((N, N, N), (L, L, L)))
trap = HarmonicTrap(1.0, 1.0, 1.0)  # isotropic, ω = 1

c0 = 30.0
c1 = -0.5  # ferromagnetic
c_dd_val = eps_dd * c0  # match Eu151 DDI ratio
interactions = InteractionParams(c0, c1)

println("Spin-1 model (3 components, fast): c0=$c0, c1=$c1")
println("  DDI coupling: c_dd = $(@sprintf("%.1f", c_dd_val)) (ε_dd = $(@sprintf("%.3f", eps_dd)))")
println("  Grid: $(N)³, box = $(L) a_ho")
println()

# --- Ground state WITHOUT DDI ---
println("--- Ground state (no DDI) ---")
result_noddi = find_ground_state(;
    grid, atom=atom_s1, interactions, potential=trap,
    dt=0.002, n_steps=5000, tol=1e-10,
    initial_state=:ferromagnetic, enable_ddi=false,
)
println("  Converged: $(result_noddi.converged)")
println("  Energy: $(@sprintf("%.6f", result_noddi.energy))")

psi = result_noddi.workspace.state.psi
n = SpinorBEC.total_density(psi, 3)

# --- Ground state WITH DDI ---
println("--- Ground state (with DDI) ---")
result_ddi = find_ground_state(;
    grid, atom=atom_s1, interactions, potential=trap,
    dt=0.002, n_steps=5000, tol=1e-10,
    initial_state=:ferromagnetic, enable_ddi=true, c_dd=c_dd_val,
)
println("  Converged: $(result_ddi.converged)")
println("  Energy: $(@sprintf("%.6f", result_ddi.energy))")

psi_ddi = result_ddi.workspace.state.psi
n_ddi = SpinorBEC.total_density(psi_ddi, 3)

dE = result_ddi.energy - result_noddi.energy
println("  ΔE from DDI: $(@sprintf("%.6f", dE))")
println()

# --- Density widths (DDI should elongate along z for head-to-tail dipoles) ---
mid = N ÷ 2
x = grid.x[1]

println("--- Density widths (RMS) ---")
for (label, dens) in [("No DDI", n), ("W/ DDI", n_ddi)]
    nx = [dens[i, mid, mid] for i in 1:N]
    ny = [dens[mid, j, mid] for j in 1:N]
    nz = [dens[mid, mid, k] for k in 1:N]
    sx = sqrt(sum(x .^ 2 .* nx) / sum(nx))
    sy = sqrt(sum(x .^ 2 .* ny) / sum(ny))
    sz = sqrt(sum(x .^ 2 .* nz) / sum(nz))
    println("  $label: σ_x=$(@sprintf("%.4f", sx)), σ_y=$(@sprintf("%.4f", sy)), σ_z=$(@sprintf("%.4f", sz)) a_ho")
end
println()

# --- Population fractions ---
dV = cell_volume(grid)
println("--- Spin populations ---")
for (label, psi_gs) in [("No DDI", psi), ("W/ DDI", psi_ddi)]
    n1 = sum(abs2, view(psi_gs, :, :, :, 1)) * dV
    n2 = sum(abs2, view(psi_gs, :, :, :, 2)) * dV
    n3 = sum(abs2, view(psi_gs, :, :, :, 3)) * dV
    println("  $label: m=+1: $(@sprintf("%.4f", n1)), m=0: $(@sprintf("%.4f", n2)), m=-1: $(@sprintf("%.4f", n3))")
end
println()

# --- Real-time dynamics: time-dependent Zeeman ---
println("--- Real-time dynamics (quadratic Zeeman ramp) ---")
zee = TimeDependentZeeman(t -> ZeemanParams(0.0, 0.5 * min(t / 0.5, 1.0)))
sp = SimParams(dt=0.002, n_steps=200, imaginary_time=false, normalize_every=0, save_every=100)

ws = make_workspace(;
    grid, atom=atom_s1, interactions, potential=trap, zeeman=zee,
    sim_params=sp, psi_init=copy(psi_ddi), enable_ddi=true, c_dd=c_dd_val,
)

sys = SpinSystem(atom_s1.F)
N0 = total_norm(ws.state.psi, grid)
M0 = magnetization(ws.state.psi, grid, sys)

for _ in 1:sp.n_steps
    split_step!(ws)
end

N1 = total_norm(ws.state.psi, grid)
M1 = magnetization(ws.state.psi, grid, sys)
println("  t_final = $(@sprintf("%.2f", ws.state.t)) ω⁻¹")
println("  |ΔN/N| = $(@sprintf("%.2e", abs(N1 - N0) / N0))")
println("  |ΔM|   = $(@sprintf("%.2e", abs(M1 - M0)))")
println()

println("=== Done ===")
