include(joinpath(@__DIR__, "eu151_setup.jl"))
using Printf

println("=== Adaptive dt convergence test for Eu151 3D ===")

const c1 = EU_c0 / 36

atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
grid = make_grid(GridConfig((32, 32, 32), (20.0, 20.0, 20.0)))
interactions = InteractionParams(EU_c0, c1)
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))
sys = SpinSystem(atom.F)
n_comp = sys.n_components
n_pts = grid.config.n_points
ndim = 3

# --- CFL analysis ---
println("\n--- CFL Analysis ---")

psi_gs = load(joinpath(@__DIR__, "cache_eu151_gs_3d.jld2"), "psi")

# Energy scales
dV = cell_volume(grid)
n_peak = maximum(sum(abs2.(psi_gs[:,:,:,c]) for c in 1:n_comp))
println("Peak density n_peak = $(round(n_peak, sigdigits=4))")

μ_contact = EU_c0 * n_peak
μ_ddi     = EU_c_dd * n_peak
μ_spin    = c1 * n_peak
E_zeeman  = EU_p_weak * 6.0

ξ = 1.0 / sqrt(2 * μ_contact)
k_heal = 1.0 / ξ
E_kin_heal = k_heal^2 / 2

Δx = grid.dx[1]
k_max = π / Δx
E_kin_max = k_max^2 / 2

println("μ_contact = $(round(μ_contact, sigdigits=4))")
println("μ_ddi     = $(round(μ_ddi, sigdigits=4))")
println("μ_spin    = $(round(μ_spin, sigdigits=4))")
println("E_zeeman  = $(round(E_zeeman, sigdigits=4))")
println("ξ (healing length) = $(round(ξ, sigdigits=4)) [a_ho]")
println("E_kin at ξ = $(round(E_kin_heal, sigdigits=4))")
println("E_kin_max (grid) = $(round(E_kin_max, sigdigits=4))")
println("Δx = $(round(Δx, sigdigits=4)), k_max = $(round(k_max, sigdigits=4))")

commutator_scale = E_kin_heal * μ_contact
println("\n||[T,V]|| estimate = $(round(commutator_scale, sigdigits=4))")

T_total = 27.6  # ω⁻¹
for ε_target in [0.01, 0.001, 0.0001]
    dt_cfl = sqrt(ε_target / (T_total * commutator_scale))
    n_steps_est = round(Int, T_total / dt_cfl)
    println("  ε_total=$ε_target → dt_CFL=$(round(dt_cfl, sigdigits=3)), ~$(n_steps_est) steps")
end

# --- Convergence test: run 2ms at multiple tolerances ---
println("\n--- Convergence test (t=0 to 2ms) ---")

t_test = 2.0e-3 / EU_t_unit
println("t_test = $(round(t_test, digits=3)) ω⁻¹")

test_configs = [
    ("fixed dt=0.001",  nothing, 0.001),
    ("fixed dt=0.0005", nothing, 0.0005),
    ("tol=0.1",         0.1,    nothing),
    ("tol=0.05",        0.05,   nothing),
    ("tol=0.02",        0.02,   nothing),
    ("tol=0.005",       0.005,  nothing),
]

results = []

for (label, tol, fixed_dt) in test_configs
    psi_seeded = seed_noise(psi_gs, n_comp, ndim, grid)

    sp = SimParams(; dt=0.001, n_steps=1)
    ws = make_workspace(;
        grid, atom, interactions,
        zeeman=ZeemanParams(EU_p_weak, 0.0),
        potential=trap,
        sim_params=sp,
        psi_init=psi_seeded,
        enable_ddi=true, c_dd=EU_c_dd,
    )

    t_start = time()

    if tol !== nothing
        adaptive = AdaptiveDtParams(dt_init=0.005, dt_min=0.0001, dt_max=0.01, tol=tol)
        out = run_simulation_adaptive!(ws; adaptive, t_end=t_test, save_interval=t_test)
        n_steps = out.n_accepted
        final_dt = out.final_dt
    else
        n_fixed = round(Int, t_test / fixed_dt)
        sp2 = SimParams(; dt=fixed_dt, n_steps=n_fixed, save_every=n_fixed)
        ws2 = make_workspace(;
            grid, atom, interactions,
            zeeman=ZeemanParams(EU_p_weak, 0.0),
            potential=trap,
            sim_params=sp2,
            psi_init=psi_seeded,
            enable_ddi=true, c_dd=EU_c_dd,
        )
        run_simulation!(ws2)
        ws = ws2
        n_steps = n_fixed
        final_dt = fixed_dt
    end

    elapsed = time() - t_start

    pops = Float64[]
    for c in 1:n_comp
        slice = SpinorBEC._component_slice(ndim, n_pts, c)
        push!(pops, sum(abs2, view(ws.state.psi, slice...)) * dV)
    end
    total = sum(pops)
    if total > 0; pops ./= total; end

    E = total_energy(ws)
    nrm = total_norm(ws.state.psi, grid)

    push!(results, (label=label, pops=pops, E=E, norm=nrm, n_steps=n_steps,
                    final_dt=final_dt, elapsed=elapsed))

    println("  $label: $(round(elapsed, digits=1))s, $(n_steps) steps, " *
            "m6=$(round(pops[1], digits=4)), m5=$(round(pops[2], digits=4)), " *
            "E=$(round(E, sigdigits=6)), norm=$(round(nrm, sigdigits=8))")
    flush(stdout)
end

# Convergence analysis
println("\n--- Convergence summary ---")
println("  Label              |  m₆      |  m₅      |  E          | Steps | Time")
println("  " * "-"^80)
for r in results
    @printf("  %-20s| %8.5f | %8.5f | %11.4f | %5d | %.1fs\n",
            r.label, r.pops[1], r.pops[2], r.E, r.n_steps, r.elapsed)
end

ref = results[2]  # dt=0.0005
println("\nReference: $(ref.label)")
println("Max population difference from reference:")
for r in results
    if r === ref; continue; end
    max_diff = maximum(abs.(r.pops .- ref.pops))
    println("  $(r.label): Δp_max = $(round(max_diff, sigdigits=3))")
end

println("\nDone!")
