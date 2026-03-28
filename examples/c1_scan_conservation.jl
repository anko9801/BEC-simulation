# c₁ scan with conservation monitoring
# Sweeps c₁/c₀ ratio and tracks E, N, Sz drift during EdH dynamics.
# Outputs JSON summary for plotting.
#
# Usage:
#   julia --project=. examples/c1_scan_conservation.jl
#   NGRID=64 T_FINAL=0.5 julia --project=. examples/c1_scan_conservation.jl

include(joinpath(@__DIR__, "eu151_params.jl"))
include(joinpath(@__DIR__, "json_utils.jl"))
using Printf, Random, FFTW, JLD2

N_GRID = parse(Int, get(ENV, "NGRID", "32"))
BOX = 20.0
t_final = parse(Float64, get(ENV, "T_FINAL", "0.20"))

c1_ratios = [
    ("r=0",       0.0),
    ("r=-1/72",  -1.0 / 72.0),
    ("r=+1/72",  +1.0 / 72.0),
    ("r=-1/36",  -1.0 / 36.0),
    ("r=+1/36",  +1.0 / 36.0),
]

@printf("Eu151 c₁ scan with conservation monitoring (%d³)\n", N_GRID)
@printf("c_total=%.1f, c_dd=%.1f, p=%.4f, t_final=%.2f ω⁻¹\n\n", EU_c_total, EU_c_dd, EU_p_weak, t_final)

grid = make_grid(GridConfig(ntuple(_ -> N_GRID, 3), ntuple(_ -> BOX, 3)))
trap = HarmonicTrap((1.0, 1.0, EU_λ_z))
dV = cell_volume(grid)
sys = SpinSystem(6)

cache_file = joinpath(@__DIR__, "cache_eu151_gs_3d_$(N_GRID).jld2")
psi_gs = if isfile(cache_file)
    @printf("Loading cached ground state: %s\n", cache_file)
    load(cache_file, "psi")
else
    println("Computing ground state (ITP)...")
    atom_gs = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
    gs = find_ground_state(;
        grid, atom=atom_gs,
        interactions=InteractionParams(EU_c_total, 0.0),
        zeeman=ZeemanParams(100.0, 0.0),
        potential=trap,
        dt=0.005, n_steps=15000, tol=1e-9,
        initial_state=:ferromagnetic,
        fft_flags=FFTW.MEASURE,
    )
    @printf("  converged=%s, energy=%.2f\n", gs.converged, gs.energy)
    save(cache_file, "psi", gs.workspace.state.psi)
    gs.workspace.state.psi
end

dt = 2e-4
n_steps = round(Int, t_final / dt)
save_every = max(1, round(Int, 0.005 / dt))

results = Dict{String,Any}[]

for (label, c1_ratio) in c1_ratios
    ip = eu_interaction_params(c1_ratio)
    @printf("\n--- %s (c0=%.1f, c1=%+.2f) ---\n", label, ip.c0, ip.c1)

    psi = copy(psi_gs)
    Random.seed!(42)
    SpinorBEC._add_noise!(psi, 0.001, 13, 3, grid)

    atom_dyn = AtomSpecies("Eu151", 1.0, 6, EU_a_s_dl, 0.0)
    sp = SimParams(; dt, n_steps, imaginary_time=false, save_every)
    ws = make_workspace(;
        grid, atom=atom_dyn,
        interactions=ip,
        zeeman=ZeemanParams(EU_p_weak, 0.0),
        potential=trap,
        sim_params=sp,
        psi_init=psi,
        enable_ddi=true, c_dd=EU_c_dd,
        fft_flags=FFTW.MEASURE,
    )

    cb, mon = make_conservation_monitor(ws)
    t0 = time()
    run_simulation!(ws; callback=cb)
    wall = time() - t0

    E_drift = isempty(mon.E) ? NaN : abs(mon.E[end] - mon.E[1]) / abs(mon.E[1])
    N_drift = isempty(mon.N) ? NaN : abs(mon.N[end] - mon.N[1])
    Sz_drift = isempty(mon.Sz) ? NaN : abs(mon.Sz[end] - mon.Sz[1])

    pops_final = component_populations(ws.state.psi, grid, sys).populations

    @printf("  %d steps in %.1fs, ΔE/E=%.2e, ΔN=%.2e, ΔSz=%.2e\n",
            n_steps, wall, E_drift, N_drift, Sz_drift)
    @printf("  P(+6)=%.4f, Sz=%.4f\n", pops_final[1], mon.Sz[end])

    push!(results, Dict(
        "label" => label,
        "c1_ratio" => c1_ratio,
        "c0" => ip.c0,
        "c1" => ip.c1,
        "t" => collect(mon.t),
        "E" => collect(mon.E),
        "N" => collect(mon.N),
        "Sz" => collect(mon.Sz),
        "pops_final" => collect(pops_final),
        "E_drift" => E_drift,
        "N_drift" => N_drift,
        "Sz_drift" => Sz_drift,
        "wall_s" => wall,
    ))
end

outfile = joinpath(@__DIR__, "c1_scan_conservation_$(N_GRID).json")
open(outfile, "w") do io
    write(io, _to_json(results))
end
@printf("\nResults written to %s\n", outfile)
