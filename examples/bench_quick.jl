#!/usr/bin/env julia
# Quick benchmark (~10s total) for CI regression detection.
#
# Usage:
#   julia --project=. examples/bench_quick.jl

using SpinorBEC
using FFTW

println("SpinorBEC quick benchmark")
println("Julia $(VERSION), Threads: $(Threads.nthreads())")
println()

function bench_split_step(label; grid, atom, interactions, potential, sim_params,
                          n_warmup=3, n_bench=50, kwargs...)
    ws = make_workspace(; grid, atom, interactions, potential, sim_params,
                        fft_flags=FFTW.ESTIMATE, kwargs...)
    for _ in 1:n_warmup
        split_step!(ws)
    end
    t0 = time()
    for _ in 1:n_bench
        split_step!(ws)
    end
    wall = time() - t0
    ms_per_step = wall / n_bench * 1000
    println("  $label: $(round(ms_per_step; digits=2)) ms/step ($n_bench steps)")
    ms_per_step
end

results = Dict{String,Float64}()

# --- 1D F=1, no DDI ---
let
    grid = make_grid(GridConfig(256, 20.0))
    sp = SimParams(; dt=0.005, n_steps=1)
    ms = bench_split_step("1D F=1 (256pt)";
        grid, atom=Rb87, interactions=InteractionParams(10.0, -0.5),
        potential=HarmonicTrap(1.0), sim_params=sp, n_bench=200)
    results["1d_f1"] = ms
end

# --- 2D F=1 with DDI ---
let
    grid = make_grid(GridConfig((64, 64), (16.0, 16.0)))
    sp = SimParams(; dt=0.005, n_steps=1)
    ms = bench_split_step("2D F=1+DDI (64²)";
        grid, atom=Rb87, interactions=InteractionParams(10.0, -0.5),
        potential=HarmonicTrap((1.0, 1.0)), sim_params=sp,
        enable_ddi=true, c_dd=1.0, n_bench=50)
    results["2d_f1_ddi"] = ms
end

# --- 3D F=6, small grid ---
let
    grid = make_grid(GridConfig((16, 16, 16), (12.0, 12.0, 12.0)))
    sp = SimParams(; dt=0.005, n_steps=1)
    atom = AtomSpecies("Eu151", 1.0, 6, 1.0, 0.0)
    ms = bench_split_step("3D F=6 (16³)";
        grid, atom, interactions=InteractionParams(100.0, 0.0),
        potential=HarmonicTrap((1.0, 1.0, 1.0)), sim_params=sp, n_bench=20)
    results["3d_f6"] = ms
end

# --- 3D F=6 with DDI ---
let
    grid = make_grid(GridConfig((16, 16, 16), (12.0, 12.0, 12.0)))
    sp = SimParams(; dt=0.005, n_steps=1)
    atom = AtomSpecies("Eu151", 1.0, 6, 1.0, 0.0)
    ms = bench_split_step("3D F=6+DDI (16³)";
        grid, atom, interactions=InteractionParams(100.0, 0.0),
        potential=HarmonicTrap((1.0, 1.0, 1.0)), sim_params=sp,
        enable_ddi=true, c_dd=100.0, n_bench=10)
    results["3d_f6_ddi"] = ms
end

println("\n=== Summary ===")
for (k, v) in sort(collect(results))
    println("  $k: $(round(v; digits=2)) ms/step")
end
