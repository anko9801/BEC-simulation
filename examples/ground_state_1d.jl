using SpinorBEC

println("=== Spin-1 Ground State Finder (1D) ===\n")

grid = make_grid(GridConfig(256, 30.0))
trap = HarmonicTrap(1.0)

println("--- 87Rb (ferromagnetic, c1 < 0) ---")
int_rb = InteractionParams(10.0, -0.5)
result_rb = find_ground_state(;
    grid, atom=Rb87, interactions=int_rb, potential=trap,
    dt=0.005, n_steps=10000, initial_state=:ferromagnetic,
)

psi_rb = result_rb.workspace.state.psi
dV = cell_volume(grid)
n1 = sum(abs2, psi_rb[:, 1]) * dV
n2 = sum(abs2, psi_rb[:, 2]) * dV
n3 = sum(abs2, psi_rb[:, 3]) * dV
println("  Converged: $(result_rb.converged)")
println("  Energy: $(result_rb.energy)")
println("  Population: m=+1: $(round(n1, digits=4)), m=0: $(round(n2, digits=4)), m=-1: $(round(n3, digits=4))")
println("  Expected: ferromagnetic → most atoms in m=+1")

println("\n--- 23Na (antiferromagnetic, c1 > 0) ---")
int_na = InteractionParams(10.0, 0.5)
result_na = find_ground_state(;
    grid, atom=Na23, interactions=int_na, potential=trap,
    dt=0.005, n_steps=10000, initial_state=:polar,
)

psi_na = result_na.workspace.state.psi
n1 = sum(abs2, psi_na[:, 1]) * dV
n2 = sum(abs2, psi_na[:, 2]) * dV
n3 = sum(abs2, psi_na[:, 3]) * dV
println("  Converged: $(result_na.converged)")
println("  Energy: $(result_na.energy)")
println("  Population: m=+1: $(round(n1, digits=4)), m=0: $(round(n2, digits=4)), m=-1: $(round(n3, digits=4))")
println("  Expected: polar → most atoms in m=0")
