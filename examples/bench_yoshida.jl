# Benchmark: Yoshida 4th vs Strang 2nd for Eu151 3D
include(joinpath(@__DIR__, "eu151_setup.jl"))

println("=== Yoshida vs Strang Comparison (Eu151 3D) ===\n")

const c1 = 0.0
const t_end_dim = 10e-3  # 10 ms
const t_end = t_end_dim / EU_t_unit

println("c0=$(round(EU_c0; digits=1)), c_dd=$(round(EU_c_dd; digits=1)), p=$(round(EU_p_weak; digits=3))")
println("t_end=$(round(t_end; digits=2)) ω⁻¹ ($(t_end_dim*1e3) ms)\n")

# --- Setup ---
N_GRID = 32
grid = make_grid(GridConfig((N_GRID, N_GRID, N_GRID), (20.0, 20.0, 20.0)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)

psi_gs = load(joinpath(@__DIR__, "cache_eu151_gs_3d.jld2"), "psi")

trap = HarmonicTrap((1.0, 1.0, EU_λ_z))
interactions = InteractionParams(EU_c0, c1)
sys = SpinSystem(atom.F)
n_comp = sys.n_components

function make_seeded_psi()
    seed_noise(psi_gs, n_comp, 3, grid)
end

# --- Enable tracing ---
enable_tracing!()
reset_tracing!()

# --- Compare fixed dt ---
println("--- Fixed dt comparison (10 ms) ---")
for dt in [0.01, 0.005, 0.002]
    n_steps = round(Int, t_end / dt)

    # Strang
    sp = SimParams(; dt, n_steps, save_every=n_steps)
    ws_s = make_workspace(; grid, atom, interactions, zeeman=ZeemanParams(EU_p_weak, 0.0),
                          potential=trap, sim_params=sp, psi_init=make_seeded_psi(),
                          enable_ddi=true, c_dd=EU_c_dd)
    t0 = time()
    for _ in 1:n_steps; split_step!(ws_s); end
    wall_s = time() - t0
    E_s = total_energy(ws_s)
    pops_s = [sum(abs2, view(ws_s.state.psi, :, :, :, c)) * cell_volume(grid) for c in 1:n_comp]

    # Yoshida
    ws_y = make_workspace(; grid, atom, interactions, zeeman=ZeemanParams(EU_p_weak, 0.0),
                          potential=trap, sim_params=sp, psi_init=make_seeded_psi(),
                          enable_ddi=true, c_dd=EU_c_dd)
    t0 = time()
    for _ in 1:n_steps
        SpinorBEC._yoshida_core!(ws_y, dt, n_comp)
        ws_y.state.t += dt
    end
    wall_y = time() - t0
    E_y = total_energy(ws_y)
    pops_y = [sum(abs2, view(ws_y.state.psi, :, :, :, c)) * cell_volume(grid) for c in 1:n_comp]

    dp = maximum(abs.(pops_y .- pops_s))
    println("  dt=$dt: Strang $(round(wall_s; digits=1))s  Yoshida $(round(wall_y; digits=1))s " *
            "($(round(wall_y/wall_s; digits=2))×)  Δp_max=$(round(dp; digits=6))")
end

# --- Compare adaptive ---
println("\n--- Adaptive comparison (10 ms) ---")
for tol in [0.05, 0.01, 0.005]
    adaptive = AdaptiveDtParams(; dt_init=0.005, dt_min=1e-4, dt_max=0.05, tol)

    # Strang adaptive
    sp = SimParams(; dt=0.005, n_steps=1)
    ws_s = make_workspace(; grid, atom, interactions, zeeman=ZeemanParams(EU_p_weak, 0.0),
                          potential=trap, sim_params=sp, psi_init=make_seeded_psi(),
                          enable_ddi=true, c_dd=EU_c_dd)
    t0 = time()
    res_s = run_simulation_adaptive!(ws_s; adaptive, t_end, save_interval=t_end)
    wall_s = time() - t0

    # Yoshida adaptive
    ws_y = make_workspace(; grid, atom, interactions, zeeman=ZeemanParams(EU_p_weak, 0.0),
                          potential=trap, sim_params=sp, psi_init=make_seeded_psi(),
                          enable_ddi=true, c_dd=EU_c_dd)
    t0 = time()
    res_y = run_simulation_yoshida!(ws_y; adaptive, t_end, save_interval=t_end)
    wall_y = time() - t0

    pops_s = [sum(abs2, view(ws_s.state.psi, :, :, :, c)) * cell_volume(grid) for c in 1:n_comp]
    pops_y = [sum(abs2, view(ws_y.state.psi, :, :, :, c)) * cell_volume(grid) for c in 1:n_comp]
    dp = maximum(abs.(pops_y .- pops_s))

    println("  tol=$tol: Strang $(round(wall_s; digits=1))s ($(res_s.n_accepted) steps) " *
            " Yoshida $(round(wall_y; digits=1))s ($(res_y.n_accepted) steps)  Δp=$(round(dp; digits=6))")
end

println("\n--- Timer breakdown ---")
println(TIMER)
disable_tracing!()
