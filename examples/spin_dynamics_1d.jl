using SpinorBEC

println("=== Spin-1 Dynamics (1D) ===\n")

grid = make_grid(GridConfig(256, 30.0))
sys = SpinSystem(1)

psi0 = init_psi(grid, sys; state=:uniform)

interactions = InteractionParams(5.0, 0.5)
sp = SimParams(; dt=0.001, n_steps=2000, imaginary_time=false, save_every=100)

ws = make_workspace(;
    grid, atom=Rb87, interactions, sim_params=sp, psi_init=psi0,
)

println("Running dynamics...")
result = run_simulation!(ws)

println("\nConservation check:")
N_range = extrema(result.norms)
M_range = extrema(result.magnetizations)
E_range = extrema(result.energies)

println("  Norm:   min=$(N_range[1]), max=$(N_range[2]), drift=$(abs(N_range[2]-N_range[1]))")
println("  Mag:    min=$(M_range[1]), max=$(M_range[2]), drift=$(abs(M_range[2]-M_range[1]))")
println("  Energy: min=$(E_range[1]), max=$(E_range[2]), drift=$(abs(E_range[2]-E_range[1]))")
println("  Expected: N and M conserved to <1e-10, Energy O(dt²)")
