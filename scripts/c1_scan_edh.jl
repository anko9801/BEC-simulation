include(joinpath(@__DIR__, "eu151_params.jl"))
using Printf, Random, FFTW, JLD2

# ================================================================
# c₁ scan for EdH dynamics (constraint-preserving)
#
# Physical constraint: c₀ + F²c₁ = c_total = 4π(a_s/a_ho)N ≈ 4689
# Parameterized by ratio r = c₁/c₀:
#   r = 0      → c₁ = 0, DDI-only spin dynamics
#   r = +1/36  → AFM (Buchachenko best-fit estimate)
#   r = -1/36  → FM (ferromagnetic)
#
# Physics:
#   r = 0    → DDI instability saturates to uniform ~1/13
#   r < 0 FM → polarized state stabilized, slower depletion
#   r > 0 AFM→ polarized state further destabilized, faster depletion
# ================================================================

N_GRID = parse(Int, get(ENV, "NGRID", "32"))
cache_file = joinpath(@__DIR__, "cache_eu151_gs_3d_$(N_GRID).jld2")
BOX = 20.0

c1_configs = [
    ("r=0",      0.0),
    ("r=-1/72", -1.0 / 72.0),   # FM (r=-1/F² is singular; use half-magnitude)
    ("r=+1/36", +1.0 / 36.0),   # AFM (Buchachenko best-fit)
]

C1_SELECT = get(ENV, "C1_SELECT", "all")

println("=" ^ 70)
@printf("  Eu151 c₁ scan (%d³, box=%.0f)\n", N_GRID, BOX)
@printf("  c_total = %.1f, c_dd = %.1f, p = %.4f\n", EU_c_total, EU_c_dd, EU_p_weak)
for (label, ratio) in c1_configs
    ip = eu_interaction_params(ratio)
    @printf("  %8s: c0=%.1f, c1=%+.1f (c0+36c1=%.1f)\n", label, ip.c0, ip.c1, ip.c0 + 36 * ip.c1)
end
println("=" ^ 70)

grid = make_grid(GridConfig(ntuple(_ -> N_GRID, 3), ntuple(_ -> BOX, 3)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
D = 13
dV = cell_volume(grid)
sm = spin_matrices(6)
sys = SpinSystem(6)
plans = make_fft_plans(grid.config.n_points; flags=FFTW.MEASURE)
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))

# --- Ground state (shared across all c₁ runs) ---
psi_gs = if isfile(cache_file)
    @printf("Loading cached ground state: %s\n", cache_file)
    load(cache_file, "psi")
else
    println("Computing ground state (ITP)...")
    gs = find_ground_state(;
        grid, atom,
        interactions=InteractionParams(EU_c_total, 0.0),
        zeeman=ZeemanParams(100.0, 0.0),
        potential=trap,
        dt=0.005, n_steps=15000, tol=1e-9,
        initial_state=:ferromagnetic,
        enable_ddi=false,
        fft_flags=FFTW.MEASURE,
    )
    @printf("  converged=%s, energy=%.2f\n", gs.converged, gs.energy)
    save(cache_file, "psi", gs.workspace.state.psi)
    gs.workspace.state.psi
end

n_dens = SpinorBEC.total_density(psi_gs, 3)
n_peak = maximum(n_dens)
@printf("  n_peak = %.4e, norm = %.8f\n\n", n_peak, sum(abs2, psi_gs) * dV)

# --- Dynamics parameters ---
dt = 2e-4
t_final = parse(Float64, get(ENV, "T_FINAL", "0.20"))  # ω⁻¹
n_steps = round(Int, t_final / dt)
save_every = max(1, round(Int, 0.002 / dt))  # every ~2.9 μs

# --- Summary storage ---
results = Dict{String, NamedTuple}()

for (label, c1_ratio) in c1_configs
    if C1_SELECT != "all" && label != C1_SELECT
        continue
    end

    ip = eu_interaction_params(c1_ratio)

    println("\n" * "=" ^ 70)
    @printf("  %s  (c0=%.1f, c1=%+.1f, c1/c0=%+.5f)\n", label, ip.c0, ip.c1, ip.c1 / ip.c0)
    @printf("  constraint check: c0 + F²c1 = %.1f (should be %.1f)\n", ip.c0 + 36 * ip.c1, EU_c_total)
    println("=" ^ 70)

    psi = copy(psi_gs)
    Random.seed!(42)
    SpinorBEC._add_noise!(psi, 0.001, D, 3, grid)

    sp = SimParams(; dt, n_steps=1)
    ws = make_workspace(;
        grid, atom,
        interactions=ip,
        zeeman=ZeemanParams(EU_p_weak, 0.0),
        potential=trap,
        sim_params=sp,
        psi_init=psi,
        enable_ddi=true, c_dd=EU_c_dd,
        fft_flags=FFTW.MEASURE,
    )

    Sz0 = magnetization(psi, grid, sys)
    @printf("  Initial Sz = %.4f\n\n", Sz0)

    @printf("%8s | %6s | ", "t(μs)", "P(+6)")
    for m in 5:-1:-6
        @printf("P(%+d) ", m)
    end
    @printf("| %8s\n", "Sz")
    println("-" ^ 110)

    data_t = Float64[]
    data_pops = Vector{Float64}[]
    data_sz = Float64[]

    function record_snap!(ws)
        t = ws.state.t
        t_us = t * EU_t_unit * 1e6
        pops = Float64[sum(abs2, @view(ws.state.psi[:,:,:,c])) * dV for c in 1:D]
        Sz = magnetization(ws.state.psi, grid, sys)

        push!(data_t, t)
        push!(data_pops, pops)
        push!(data_sz, Sz)

        @printf("%8.1f | %6.4f | ", t_us, pops[1])
        for c in 2:D
            @printf("%.3f ", pops[c])
        end
        @printf("| %+8.4f\n", Sz)
    end

    record_snap!(ws)

    for _ in 1:3; split_step!(ws); end

    t0 = time()
    for step in 4:(n_steps + 3)
        split_step!(ws)
        if (step - 3) % save_every == 0
            record_snap!(ws)
        end
    end
    wall = time() - t0
    @printf("\n  %d steps in %.1fs (%.2f ms/step)\n", n_steps, wall, wall / n_steps * 1000)

    # Analysis
    t_p6_09 = let idx = findfirst(p -> p[1] < 0.9, data_pops)
        idx !== nothing ? data_t[idx] * EU_t_unit * 1e6 : NaN
    end
    t_p6_05 = let idx = findfirst(p -> p[1] < 0.5, data_pops)
        idx !== nothing ? data_t[idx] * EU_t_unit * 1e6 : NaN
    end
    t_p6_01 = let idx = findfirst(p -> p[1] < 0.1, data_pops)
        idx !== nothing ? data_t[idx] * EU_t_unit * 1e6 : NaN
    end

    seq_ratio = NaN
    idx50 = findfirst(p -> p[1] < 0.5, data_pops)
    if idx50 !== nothing
        pops_50 = data_pops[idx50]
        seq_ratio = pops_50[2] / pops_50[3]  # P(+5)/P(+4)
        println("\n  Growth pattern at P(+6)=0.5:")
        sorted = sort(collect(enumerate(pops_50)), by=x -> -x[2])
        for (rank, (c, pop)) in enumerate(sorted[1:min(5, length(sorted))])
            m = 6 - (c - 1)
            @printf("    rank %d: m=%+3d, P=%.4f\n", rank, m, pop)
        end
        sequential = pops_50[2] > pops_50[3] > pops_50[4]
        @printf("  Sequential (P₅>P₄>P₃)? %s\n", sequential ? "YES" : "NO")
        @printf("  P(+5)/P(+4) = %.2f\n", seq_ratio)
    end

    final_pops = data_pops[end]
    final_Sz = data_sz[end]
    @printf("\n  Final (t=%.0f μs): Sz=%.3f\n", data_t[end] * EU_t_unit * 1e6, final_Sz)
    @printf("  Final pops: ")
    for c in 1:D
        m = 6 - (c - 1)
        @printf("m=%+d:%.3f ", m, final_pops[c])
    end
    println()

    results[label] = (;
        c1_ratio,
        c0=ip.c0,
        c1=ip.c1,
        t_p6_09, t_p6_05, t_p6_01,
        seq_ratio,
        final_Sz,
        final_pops,
    )
end

# --- Comparison table ---
if length(results) > 1
    println("\n" * "=" ^ 70)
    println("  COMPARISON TABLE")
    println("=" ^ 70)
    @printf("%-10s | %8s | %8s | %10s | %10s | %10s | %10s | %8s\n",
        "Config", "c0", "c1", "P6<0.9(μs)", "P6<0.5(μs)", "P6<0.1(μs)", "P5/P4@0.5", "Sz_final")
    println("-" ^ 90)
    for (label, _) in c1_configs
        haskey(results, label) || continue
        r = results[label]
        @printf("%-10s | %8.1f | %+8.1f | %10.1f | %10.1f | %10.1f | %10.2f | %8.3f\n",
            label, r.c0, r.c1, r.t_p6_09, r.t_p6_05, r.t_p6_01, r.seq_ratio, r.final_Sz)
    end
end
