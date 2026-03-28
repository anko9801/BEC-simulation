include(joinpath(@__DIR__, "eu151_params.jl"))
using Printf, Random, FFTW, JLD2

# ================================================================
# High-resolution early dynamics diagnostic
# Captures DDI instability growth on μs timescale
#
# Theory prediction (c₁=0):
#   γ_max ~ c_dd × n_peak × F ≈ 170 ω
#   e-folding time ≈ 1/170 ≈ 0.006 ω⁻¹ ≈ 8 μs
#   saturation time ≈ ln(1/ε₀)/γ ≈ 0.04 ω⁻¹ ≈ 60 μs
# ================================================================

N_GRID = parse(Int, get(ENV, "NGRID", "32"))
cache_file = joinpath(@__DIR__, "cache_eu151_gs_3d_$(N_GRID).jld2")
BOX = 20.0  # always use box=20 (consistent with all cached ground states)

println("=" ^ 70)
@printf("  Eu151 Early Dynamics Diagnostic (%d³, box=%.0f)\n", N_GRID, BOX)
println("=" ^ 70)

grid = make_grid(GridConfig(ntuple(_ -> N_GRID, 3), ntuple(_ -> BOX, 3)))
atom = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
D = 13
dV = cell_volume(grid)
sm = spin_matrices(6)
sys = SpinSystem(6)
plans = make_fft_plans(grid.config.n_points; flags=FFTW.MEASURE)
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))

# --- Ground state ---
psi_gs = if isfile(cache_file)
    @printf("Loading cached ground state: %s\n", cache_file)
    load(cache_file, "psi")
else
    println("Computing ground state (ITP)...")
    gs = find_ground_state(;
        grid, atom,
        interactions=InteractionParams(EU_c0, 0.0),
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
@printf("  n_peak = %.4e, norm = %.8f\n", n_peak, sum(abs2, psi_gs) * dV)

# Theory predictions
γ_max = EU_c_dd * n_peak * 6.0
t_efold = 1.0 / γ_max
t_sat = log(1.0 / 0.001) / γ_max
@printf("\nTheory predictions:\n")
@printf("  γ_max = %.1f ω\n", γ_max)
@printf("  e-folding = %.4f ω⁻¹ = %.1f μs\n", t_efold, t_efold * EU_t_unit * 1e6)
@printf("  saturation = %.4f ω⁻¹ = %.1f μs\n", t_sat, t_sat * EU_t_unit * 1e6)
@printf("  p_weak = %.4f (Zeeman gap)\n", EU_p_weak)
@printf("  DDI/Zeeman ratio = %.0f\n", γ_max / EU_p_weak)

# --- Seed noise ---
psi = copy(psi_gs)
Random.seed!(42)
SpinorBEC._add_noise!(psi, 0.001, D, 3, grid)

# --- Dynamics setup ---
dt = 2e-4
t_final = 0.20    # ω⁻¹ ≈ 290 μs — well past predicted saturation
n_steps = round(Int, t_final / dt)
save_every = max(1, round(Int, 0.001 / dt))  # every ~1.4 μs

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

Sz0 = magnetization(psi, grid, sys)
Lz0 = orbital_angular_momentum(psi, grid, plans)
Jz0 = Sz0 + Lz0
@printf("\nInitial: Sz=%.4f, Lz=%.6f, Jz=%.4f\n", Sz0, Lz0, Jz0)
@printf("dt=%.0e, %d steps, save_every=%d, t_final=%.3f ω⁻¹ (%.0f μs)\n\n",
    dt, n_steps, save_every, t_final, t_final * EU_t_unit * 1e6)

# --- Header ---
@printf("%8s | %6s | ", "t(μs)", "P(+6)")
for m in 5:-1:-6
    @printf("P(%+d) ", m)
end
@printf("| %8s | %8s | %9s\n", "Sz", "Lz", "Jz_drift")
println("-" ^ 130)

# --- Collect data ---
data_t = Float64[]
data_pops = Vector{Float64}[]
data_sz = Float64[]
data_lz = Float64[]

function record!(ws, plans, grid, sys, dV, Jz0, data_t, data_pops, data_sz, data_lz)
    t = ws.state.t
    t_us = t * EU_t_unit * 1e6
    pops = Float64[]
    for c in 1:13
        p = sum(abs2, @view(ws.state.psi[:,:,:,c])) * dV
        push!(pops, p)
    end
    Sz = magnetization(ws.state.psi, grid, sys)
    Lz = orbital_angular_momentum(ws.state.psi, grid, plans)
    Jz = Sz + Lz

    push!(data_t, t)
    push!(data_pops, pops)
    push!(data_sz, Sz)
    push!(data_lz, Lz)

    @printf("%8.1f | %6.4f | ", t_us, pops[1])
    for c in 2:13
        @printf("%.3f ", pops[c])
    end
    @printf("| %+8.4f | %+8.4f | %+9.5f\n", Sz, Lz, Jz - Jz0)
end

# Initial snapshot
record!(ws, plans, grid, sys, dV, Jz0, data_t, data_pops, data_sz, data_lz)

# Warmup
for _ in 1:3; split_step!(ws); end

# --- Main loop ---
t0 = time()
for step in 4:(n_steps + 3)
    split_step!(ws)
    if (step - 3) % save_every == 0
        record!(ws, plans, grid, sys, dV, Jz0, data_t, data_pops, data_sz, data_lz)
    end
end
wall = time() - t0

# --- Summary ---
println("\n" * "=" ^ 70)
@printf("  %d steps in %.1fs (%.2f ms/step)\n", n_steps, wall, wall / n_steps * 1000)

# Find when P(+6) first drops below 0.9 and 0.5
for threshold in [0.9, 0.5, 0.1]
    idx = findfirst(p -> p[1] < threshold, data_pops)
    if idx !== nothing
        t_us = data_t[idx] * EU_t_unit * 1e6
        @printf("  P(+6) < %.1f at t = %.1f μs (%.4f ω⁻¹)\n", threshold, t_us, data_t[idx])
    else
        @printf("  P(+6) never drops below %.1f\n", threshold)
    end
end

# Check if growth is sequential or simultaneous
println("\n  Growth pattern analysis (at P(+6) = 0.5):")
idx50 = findfirst(p -> p[1] < 0.5, data_pops)
if idx50 !== nothing
    pops_at_50 = data_pops[idx50]
    sorted = sort(collect(enumerate(pops_at_50)), by=x -> -x[2])
    for (rank, (c, pop)) in enumerate(sorted[1:min(5, length(sorted))])
        m = 6 - (c - 1)
        @printf("    rank %d: m=%+3d, P=%.4f\n", rank, m, pop)
    end
    # Sequential indicator: is P(+5) > P(+4) > P(+3)?
    sequential = pops_at_50[2] > pops_at_50[3] > pops_at_50[4]
    @printf("  Sequential (P₅>P₄>P₃)? %s\n", sequential ? "YES" : "NO")
end

# Jz conservation
jz_drift_max = maximum(abs, [data_sz[i] + data_lz[i] - Jz0 for i in eachindex(data_t)])
@printf("\n  Max |ΔJz| = %.5f (relative: %.2e)\n", jz_drift_max, jz_drift_max / abs(Jz0))
