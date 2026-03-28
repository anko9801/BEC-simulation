# Benchmark Eu151 3D split-step using TimerOutputs tracing.
include(joinpath(@__DIR__, "eu151_setup.jl"))

println("Eu151 3D benchmark (Matsui et al.)")
println("Threads: $(Threads.nthreads())")

const c1 = 0.0  # Eu151: a_F unknown, DDI dominates
println("c0=$(round(EU_c0;digits=1)), c1=$(round(c1;digits=1)), c_dd=$(round(EU_c_dd;digits=1)), p=$(round(EU_p_weak;digits=3))")

# --- Setup ---
N_GRID = 32
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (20.0, 20.0, 20.0)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)

psi_gs = load_or_compute_gs(grid)

# Workspace with full physics
dt = 0.005
sp = SimParams(; dt, n_steps=1)
ws = make_workspace(;
    grid, atom, interactions=InteractionParams(EU_c0, c1),
    zeeman=ZeemanParams(EU_p_weak, 0.0),
    potential=HarmonicTrap((1.0, 1.0, EU_λ_z)),
    sim_params=sp, psi_init=copy(psi_gs),
    enable_ddi=true, c_dd=EU_c_dd,
)

# --- Warmup ---
println("\nWarmup (3 steps)...")
for _ in 1:3; split_step!(ws); end

# --- Enable tracing ---
enable_tracing!()
reset_tracing!()

# --- Timed run ---
N_STEPS = 100
println("Running $N_STEPS steps with tracing...\n")
t0 = time()
for _ in 1:N_STEPS
    split_step!(ws)
end
wall = time() - t0

# --- Results ---
println(TIMER)

println("\n=== Summary ===")
println("  $N_STEPS steps in $(round(wall; digits=2))s  ($(round(wall/N_STEPS*1000; digits=2)) ms/step)")

# Projection
t_total = 40e-3 / EU_t_unit
n_est = round(Int, t_total / dt)
println("  40 ms simulation (~$n_est steps): ~$(round(wall/N_STEPS * n_est; digits=0))s ($(round(wall/N_STEPS * n_est / 60; digits=1)) min)")

disable_tracing!()
