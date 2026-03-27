using SpinorBEC
using Printf

println("=" ^ 60)
println("  Rb87 Spin-1 BEC: Ground State + Quench Dynamics (1D)")
println("=" ^ 60)

grid = make_grid(GridConfig((128,), (20.0,)))
println("Grid: 128 pts, L=20")

# --- Ground state (ferromagnetic at high p, ITP) ---
println("\n--- Finding ground state (ITP) ---")
gs = find_ground_state(;
    grid, atom=Rb87,
    interactions=InteractionParams(500.0, -15.0),
    zeeman=ZeemanParams(5.0, 0.0),
    potential=HarmonicTrap((1.0,)),
    dt=0.01, n_steps=5000, tol=1e-8,
    initial_state=:ferromagnetic,
)
println("  converged = $(gs.converged), energy = $(round(gs.energy; digits=4)), steps = $(gs.workspace.state.step)")

psi_gs = copy(gs.workspace.state.psi)
dV = cell_volume(grid)
sm = spin_matrices(1)

for c in 1:3
    pop = sum(abs2, @view(psi_gs[:, c])) * dV
    m = 1 - (c - 1)
    @printf("  |m=%+d|² = %.6f\n", m, pop)
end

# --- Quench: remove Zeeman field, observe spin dynamics ---
println("\n--- Quench dynamics: p=5 → p=0 (ferromagnetic, c1<0) ---")
sp = SimParams(; dt=0.002, n_steps=1)
ws = make_workspace(;
    grid, atom=Rb87,
    interactions=InteractionParams(500.0, -15.0),
    zeeman=ZeemanParams(0.0, 0.0),
    potential=HarmonicTrap((1.0,)),
    sim_params=sp,
    psi_init=psi_gs,
)

n_total = 2500
save_every = 500
println("  dt=0.002, total_steps=$n_total (t_final=$(n_total*0.002))")

for step in 1:n_total
    split_step!(ws)
    if step % save_every == 0
        t = ws.state.t
        norm_now = sum(abs2, ws.state.psi) * dV
        mag_now = magnetization(ws.state.psi, grid, sm.system)
        pops = [sum(abs2, @view(ws.state.psi[:, c])) * dV for c in 1:3]
        @printf("  t=%.2f | norm=%.8f | ⟨Fz⟩=%+.6f | pop=[%.4f, %.4f, %.4f]\n",
            t, norm_now, mag_now, pops[1], pops[2], pops[3])
    end
end

norm_f = sum(abs2, ws.state.psi) * dV
mag_f = magnetization(ws.state.psi, grid, sm.system)
@printf("\nFinal: norm_drift=%.2e, mag_drift=%.2e\n",
    abs(norm_f - 1.0), abs(mag_f - 1.0))

# ============================================================
println("\n" * "=" ^ 60)
println("  Eu151 F=6 BEC: DDI-driven spin relaxation (3D, 16³)")
println("=" ^ 60)

include(joinpath(@__DIR__, "eu151_params.jl"))

N_GRID = 16
grid3 = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (16.0, 16.0, 16.0)))
atom_eu = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)

println("Grid: $(N_GRID)³, L=16")
println("c0=$(round(EU_c0; digits=1)), c_dd=$(round(EU_c_dd; digits=1)), p=$(round(EU_p_weak; digits=3))")

# Ground state (no DDI, high p)
println("\n--- Finding ground state (ITP, no DDI) ---")
gs_eu = find_ground_state(;
    grid=grid3, atom=atom_eu,
    interactions=InteractionParams(EU_c0, 0.0),
    zeeman=ZeemanParams(100.0, 0.0),
    potential=HarmonicTrap((1.0, 1.0, EU_λ_z)),
    dt=0.005, n_steps=10000, tol=1e-8,
    initial_state=:ferromagnetic,
    enable_ddi=false,
)
println("  converged = $(gs_eu.converged), energy = $(round(gs_eu.energy; digits=2)), steps = $(gs_eu.workspace.state.step)")

psi_eu = copy(gs_eu.workspace.state.psi)
dV3 = cell_volume(grid3)
sm6 = spin_matrices(6)
D = 13

println("  norm = $(round(sum(abs2, psi_eu) * dV3; digits=8))")
pop_gs = [sum(abs2, @view(psi_eu[:,:,:,c])) * dV3 for c in 1:D]
@printf("  m=+6 pop = %.6f\n", pop_gs[1])

# Add noise and run dynamics
println("\n--- DDI dynamics: quench to p=$(round(EU_p_weak; digits=3)), 5 ms ---")
psi_noisy = copy(psi_eu)
SpinorBEC._add_noise!(psi_noisy, 0.001, D, 3, grid3)

dt_eu = 0.005
n_steps_eu = 100  # 0.5 ω⁻¹ ≈ 0.72 ms
sp_eu = SimParams(; dt=dt_eu, n_steps=1)
ws_eu = make_workspace(;
    grid=grid3, atom=atom_eu,
    interactions=InteractionParams(EU_c0, 0.0),
    zeeman=ZeemanParams(EU_p_weak, 0.0),
    potential=HarmonicTrap((1.0, 1.0, EU_λ_z)),
    sim_params=sp_eu,
    psi_init=psi_noisy,
    enable_ddi=true, c_dd=EU_c_dd,
)

save_every_eu = 25
println("  dt=$dt_eu, total_steps=$n_steps_eu")

t0 = time()
for step in 1:n_steps_eu
    split_step!(ws_eu)
    if step % save_every_eu == 0
        t = ws_eu.state.t
        t_ms = t * EU_t_unit * 1000
        norm_now = sum(abs2, ws_eu.state.psi) * dV3
        mag_now = magnetization(ws_eu.state.psi, grid3, sm6.system)
        pop6 = sum(abs2, @view(ws_eu.state.psi[:,:,:,1])) * dV3
        @printf("  step=%4d | t=%.3f (%.2f ms) | norm=%.6f | ⟨Fz⟩=%+.4f | m=+6: %.4f\n",
            step, t, t_ms, norm_now, mag_now, pop6)
    end
end
wall = time() - t0
@printf("\n  %d steps in %.1fs (%.1f ms/step)\n", n_steps_eu, wall, wall/n_steps_eu*1000)

norm_eu_f = sum(abs2, ws_eu.state.psi) * dV3
@printf("  norm drift: %.2e\n", abs(norm_eu_f - 1.0))
