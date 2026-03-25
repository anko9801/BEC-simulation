# Benchmark Eu151 3D split-step using TimerOutputs tracing.
using SpinorBEC
using JLD2

println("Eu151 3D benchmark (Matsui et al.)")
println("Threads: $(Threads.nthreads())")

# --- Parameters ---
const ω_ref  = 2π * 110.0
const N_atoms = 50_000
const m_Eu   = Eu151.mass
const a_ho   = sqrt(Units.HBAR / (m_Eu * ω_ref))
const a_s_dl = Eu151.a0 / a_ho
const c0     = 4π * a_s_dl * N_atoms
const c1     = 0.0                # Eu151: a_F unknown, DDI dominates
const c_dd   = N_atoms * compute_c_dd(Eu151) / (Units.HBAR * ω_ref * a_ho^3)
const λ_z    = 130.0 / 110.0
const p_weak = (7.0/6.0) * Units.MU_BOHR * 2.6e-9 / (Units.HBAR * ω_ref)
const t_unit = 1.0 / ω_ref

println("c0=$(round(c0;digits=1)), c1=$(round(c1;digits=1)), c_dd=$(round(c_dd;digits=1)), p=$(round(p_weak;digits=3))")

# --- Setup ---
N_GRID = 32
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (20.0, 20.0, 20.0)))
atom = AtomSpecies("Eu151", 1.0, 6, a_s_dl, 0.0)

# Ground state (c1=0 for ITP speed)
gs_cache = joinpath(@__DIR__, "cache_eu151_gs_3d.jld2")
psi_gs = if isfile(gs_cache)
    load(gs_cache, "psi")
else
    println("Finding ground state...")
    gs = find_ground_state(;
        grid, atom, interactions=InteractionParams(c0, 0.0),
        zeeman=ZeemanParams(100.0, 0.0),
        potential=HarmonicTrap((1.0, 1.0, λ_z)),
        dt=0.005, n_steps=20000, tol=1e-9,
        initial_state=:ferromagnetic, enable_ddi=false,
    )
    psi_out = copy(gs.workspace.state.psi)
    jldsave(gs_cache; psi=psi_out)
    psi_out
end

# Workspace with full physics
dt = 0.005
sp = SimParams(; dt, n_steps=1)
ws = make_workspace(;
    grid, atom, interactions=InteractionParams(c0, c1),
    zeeman=ZeemanParams(p_weak, 0.0),
    potential=HarmonicTrap((1.0, 1.0, λ_z)),
    sim_params=sp, psi_init=copy(psi_gs),
    enable_ddi=true, c_dd,
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
t_total = 40e-3 / t_unit
n_est = round(Int, t_total / dt)
println("  40 ms simulation (~$n_est steps): ~$(round(wall/N_STEPS * n_est; digits=0))s ($(round(wall/N_STEPS * n_est / 60; digits=1)) min)")

disable_tracing!()
