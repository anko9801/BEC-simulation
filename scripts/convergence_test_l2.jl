include(joinpath(@__DIR__, "eu151_setup.jl"))
using Printf

println("=== Convergence test: L2 wavefunction estimator ===")

const c1 = EU_c0 / 36

atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
grid = make_grid(GridConfig((32, 32, 32), (20.0, 20.0, 20.0)))
interactions = InteractionParams(EU_c0, c1)
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))
sys = SpinSystem(atom.F)
n_comp = sys.n_components
n_pts = grid.config.n_points
ndim = 3

psi_gs = load(joinpath(@__DIR__, "cache_eu151_gs_3d.jld2"), "psi")

# Seed noise ONCE and reuse for all tests
psi_seeded = seed_noise(psi_gs, n_comp, ndim, grid)

t_test = 2.0e-3 / EU_t_unit
println("t_test = $(round(t_test, digits=3)) ω⁻¹")
println("Using _wavefunction_l2_change (phase-sensitive)")

# --- Enable tracing ---
enable_tracing!()
reset_tracing!()

test_configs = [
    ("fixed dt=0.0005", nothing, 0.0005),
    ("fixed dt=0.001",  nothing, 0.001),
    ("tol=0.1",         0.1,    nothing),
    ("tol=0.05",        0.05,   nothing),
    ("tol=0.02",        0.02,   nothing),
    ("tol=0.005",       0.005,  nothing),
    ("tol=0.001",       0.001,  nothing),
]

results = []

for (label, tol, fixed_dt) in test_configs
    sp = SimParams(; dt=0.001, n_steps=1)
    ws = make_workspace(;
        grid, atom, interactions,
        zeeman=ZeemanParams(EU_p_weak, 0.0),
        potential=trap,
        sim_params=sp,
        psi_init=copy(psi_seeded),
        enable_ddi=true, c_dd=EU_c_dd,
    )

    t_start = time()

    if tol !== nothing
        adaptive = AdaptiveDtParams(dt_init=0.005, dt_min=0.0001, dt_max=0.01, tol=tol)
        out = run_simulation_adaptive!(ws; adaptive, t_end=t_test, save_interval=t_test)
        n_steps = out.n_accepted
        n_rej = out.n_rejected
        final_dt = out.final_dt
    else
        n_fixed = round(Int, t_test / fixed_dt)
        sp2 = SimParams(; dt=fixed_dt, n_steps=n_fixed, save_every=n_fixed)
        ws2 = make_workspace(;
            grid, atom, interactions,
            zeeman=ZeemanParams(EU_p_weak, 0.0),
            potential=trap,
            sim_params=sp2,
            psi_init=copy(psi_seeded),
            enable_ddi=true, c_dd=EU_c_dd,
        )
        run_simulation!(ws2)
        ws = ws2
        n_steps = n_fixed
        n_rej = 0
        final_dt = fixed_dt
    end

    elapsed = time() - t_start

    pops = Float64[]
    dV = cell_volume(grid)
    for c in 1:n_comp
        slice = SpinorBEC._component_slice(ndim, n_pts, c)
        push!(pops, sum(abs2, view(ws.state.psi, slice...)) * dV)
    end
    total = sum(pops)
    if total > 0; pops ./= total; end

    E = total_energy(ws)
    nrm = total_norm(ws.state.psi, grid)

    push!(results, (label=label, pops=pops, E=E, norm=nrm, n_steps=n_steps,
                    n_rej=n_rej, final_dt=final_dt, elapsed=elapsed))

    @printf("  %-20s %6.1fs %5d+%3drej  m6=%.5f  E=%.4f  dt_f=%.2e\n",
            label, elapsed, n_steps, n_rej, pops[1], E, final_dt)
    flush(stdout)
end

println("\n--- Summary ---")
@printf("  %-20s | %8s | %8s | %11s | %5s+%3s | %6s\n",
        "Label", "m₆", "m₅", "E", "Steps", "Rej", "Time")
println("  " * "-"^80)
for r in results
    @printf("  %-20s | %8.5f | %8.5f | %11.4f | %5d+%3d | %5.1fs\n",
            r.label, r.pops[1], r.pops[2], r.E, r.n_steps, r.n_rej, r.elapsed)
end

ref = results[1]
println("\nReference: $(ref.label)")
println("Max population difference from reference:")
for r in results[2:end]
    max_diff = maximum(abs.(r.pops .- ref.pops))
    @printf("  %-20s Δp_max = %.4f\n", r.label, max_diff)
end

# Energy drift from initial
E0 = total_energy(make_workspace(;
    grid, atom, interactions,
    zeeman=ZeemanParams(EU_p_weak, 0.0),
    potential=trap,
    sim_params=SimParams(; dt=0.001, n_steps=1),
    psi_init=copy(psi_seeded),
    enable_ddi=true, c_dd=EU_c_dd,
))
println("\nInitial energy E₀ = $(round(E0, digits=4))")
println("Energy drift |E-E₀|/|E₀|:")
for r in results
    drift = abs(r.E - E0) / abs(E0)
    @printf("  %-20s ΔE/E = %.2e\n", r.label, drift)
end

println("\n--- Timer breakdown ---")
println(TIMER)
disable_tracing!()

println("\nDone!")
