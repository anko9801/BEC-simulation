include(joinpath(@__DIR__, "eu151_params.jl"))
using Printf, Random, FFTW

println("=" ^ 60)
println("  Eu151 F=6 DDI Spin Relaxation (3D, 64³)")
println("  Matsui et al., Science 391, 384-388 (2026)")
println("=" ^ 60)

N_GRID = 64
BOX = 20.0
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (BOX, BOX, BOX)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
D = 13

dV = cell_volume(grid)
sm = spin_matrices(6)
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))

@printf("Grid: %d³, L=%.1f, dV=%.4e\n", N_GRID, BOX, dV)
@printf("c0=%.1f, c_dd=%.1f, p=%.3f, ε_dd=%.3f\n", EU_c0, EU_c_dd, EU_p_weak, EU_ε_dd)
@printf("t_unit=%.4f ms, a_ho=%.4f μm\n", EU_t_unit * 1e3, EU_a_ho * 1e6)

# --- Ground state ---
println("\n--- Finding ground state (ITP, no DDI, p=100) ---")
t_gs = time()
gs = find_ground_state(;
    grid, atom,
    interactions=InteractionParams(EU_c0, 0.0),
    zeeman=ZeemanParams(100.0, 0.0),
    potential=trap,
    dt=0.005, n_steps=20000, tol=1e-9,
    initial_state=:ferromagnetic,
    enable_ddi=false,
    fft_flags=FFTW.MEASURE,
)
wall_gs = time() - t_gs
@printf("  converged=%s, energy=%.2f, steps=%d (%.1fs)\n",
    gs.converged, gs.energy, gs.workspace.state.step, wall_gs)

psi_gs = copy(gs.workspace.state.psi)
@printf("  norm=%.8f, m=+6 pop=%.6f\n",
    sum(abs2, psi_gs) * dV,
    sum(abs2, @view(psi_gs[:,:,:,1])) * dV)

# --- Seed noise ---
psi = copy(psi_gs)
Random.seed!(42)
SpinorBEC._add_noise!(psi, 0.001, D, 3, grid)
println("  noise seeded (amp=0.001)")

# --- Workspace ---
dt = 0.002
n_steps = 500  # 1.0 ω⁻¹ ≈ 1.45 ms
sp = SimParams(; dt, n_steps=1)
ws = make_workspace(;
    grid, atom,
    interactions=InteractionParams(EU_c0, 0.0),
    zeeman=ZeemanParams(EU_p_weak, 0.0),
    potential=trap,
    sim_params=sp,
    psi_init=psi,
    enable_ddi=true, c_dd=EU_c_dd,
    fft_flags=FFTW.MEASURE,
)

t_final = dt * n_steps
t_final_ms = t_final * EU_t_unit * 1e3
@printf("\n--- DDI dynamics: %d steps, dt=%.3f, t_final=%.2f (%.2f ms) ---\n",
    n_steps, dt, t_final, t_final_ms)

# --- Warmup ---
print("Warmup (3 steps)... ")
for _ in 1:3; split_step!(ws); end
println("done")

# --- Run ---
save_every = 50
println()
@printf("  %6s | %8s | %8s | %8s | %8s | %8s | %8s\n",
    "step", "t", "t(ms)", "norm", "⟨Fz⟩", "m=+6", "m=-6")
println("  " * "-"^70)

t0 = time()
for step in 4:(n_steps+3)
    split_step!(ws)
    if (step - 3) % save_every == 0
        t = ws.state.t
        t_ms = t * EU_t_unit * 1e3
        norm_now = sum(abs2, ws.state.psi) * dV
        mag_now = magnetization(ws.state.psi, grid, sm.system)
        pop_p6 = sum(abs2, @view(ws.state.psi[:,:,:,1])) * dV
        pop_m6 = sum(abs2, @view(ws.state.psi[:,:,:,D])) * dV
        @printf("  %6d | %8.3f | %8.2f | %8.6f | %+8.4f | %8.4f | %8.4f\n",
            step-3, t, t_ms, norm_now, mag_now, pop_p6, pop_m6)
    end
end
wall = time() - t0

# --- Summary ---
norm_f = sum(abs2, ws.state.psi) * dV
mag_f = magnetization(ws.state.psi, grid, sm.system)

println()
println("=" ^ 60)
@printf("  %d steps in %.1fs (%.1f ms/step)\n", n_steps, wall, wall/n_steps*1000)
@printf("  norm drift: %.2e\n", abs(norm_f - 1.0))
@printf("  ⟨Fz⟩: %.4f → %.4f\n", 6.0, mag_f)

# Population distribution
println("\n  Population per m:")
for c in 1:D
    m = 6 - (c - 1)
    pop = sum(abs2, @view(ws.state.psi[:,:,:,c])) * dV
    bar = repeat("█", round(Int, pop * 50))
    @printf("    m=%+3d: %.4f %s\n", m, pop, bar)
end

# Projection to full 40ms
steps_40ms = round(Int, 40e-3 / (EU_t_unit * dt))
@printf("\n  Full 40ms (~%d steps): ~%.0fs (%.1f min)\n",
    steps_40ms, wall/n_steps * steps_40ms, wall/n_steps * steps_40ms / 60)
