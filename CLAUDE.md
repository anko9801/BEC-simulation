# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (8982 cases)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. -e 'using SpinorBEC; include("test/test_simulation.jl")'

# REPL with package loaded
julia --project=. -e 'using SpinorBEC'

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Benchmark Eu151 3D (with tracing)
julia --project=. examples/bench_eu151.jl
```

## Architecture

SpinorBEC.jl simulates spin-F Bose-Einstein condensates using the split-step Fourier method in 1D/2D/3D.

### Dimensionless Units

All propagators use ℏ=m=1. Physical quantities are scaled via `harmonic_scales(mass, omega)` in `units.jl`. Kinetic step: `exp(-ik²dt/2)`.

### Include Order (in `SpinorBEC.jl`)

`types.jl` must be first — all struct definitions live there. The rest follows dependency order:

```
types → units → grid → spin_matrices → spinor_utils → atoms → interactions →
potentials → zeeman → propagators → spin_mixing → nematic → losses → split_step →
raman → ddi → ddi_padded → optical_trap → optics → laser_potential → thomas_fermi →
fft_utils → observables → energy → currents → vorticity → diagnostics → majorana →
simulation_utils → initialization → ground_state → simulation → adaptive → yoshida →
io → experiment → experiment_runner → unitful_support
```

### Core Types (`types.jl`)

- `GridConfig{N}` — grid configuration (n_points, box_size) with validation
- `Grid{N}` — N-dimensional spatial grid with FFT wavenumbers
- `SpinSystem(F)` — spin quantum number, `n_components = 2F+1`
- `SpinMatrices{D}` — static spin-F matrices (Fx, Fy, Fz, F·F) as `SMatrix`
- `SimState{N,A}` — mutable: wavefunction `psi`, time, step counter
- `Workspace{N,A,P,IP,SM,ZEE,DDI,DDIB,RAM,LOSS,DDIP,BK}` — fully parameterized (12 type params) immutable container holding all simulation state
- `AdaptiveDtParams` — adaptive time-stepping parameters (dt_init, dt_min, dt_max, tol)
- `LossParams` — dipolar relaxation and 3-body loss parameters

### Wavefunction Layout

`psi` is an `Array{ComplexF64, N+1}` where spatial dimensions come first, spinor component last: `psi[x, y, ..., c]` for component `c ∈ 1:2F+1`.

Access helpers in `spinor_utils.jl`:
- `_component_slice(ndim, n_pts, c)` — indexing tuple for `view(psi, ...)`
- `_get_spinor(psi, I, Val(D))` → `SVector{D}` at CartesianIndex `I`
- `_set_spinor!(psi, I, spinor, Val(D))` — write spinor back
- `_matvec(V, x)` — allocation-free `Matrix × SVector` (avoids SMatrix heap alloc at large D)
- `_apply_euler_spin_rotation(...)` — Euler angle spin rotation with MVector scratch buffers

### N-Dimensional Genericity

All loops use `CartesianIndices(n_pts)` — no specialized 1D/2D/3D code paths. Grid dimension `N` is a type parameter propagated through `Grid{N}`, `Workspace{N,...}`, etc.

### Split-Step Pipeline (`split_step.jl`)

Strang splitting (2nd order symmetric):
1. Half potential step — symmetric inner splitting:
   `diag(dt/4) → SM(dt/4) → nematic(dt/4) → Raman(dt/4) → DDI(dt/2) → Raman(dt/4) → nematic(dt/4) → SM(dt/4) → diag(dt/4)`
2. Full kinetic step (batched FFT → multiply phase → batched IFFT)
3. Half potential step (mirror of step 1)
4. Loss step (if enabled, real-time only)

Instrumented with `@timeit_debug TIMER` on all sub-steps for profiling.

**Spin mixing** (`spin_mixing.jl`): skips when `c1 ≈ 0`. D=3: Rodrigues' formula (machine-precision unitarity). D>3: Euler angle decomposition with O(D) spin expectation via raising/lowering operators and O(D²) rotation via Fy eigendecomposition.

**DDI** (`ddi.jl` + `ddi_padded.jl`): k-space convolution with tensor `Q_αβ(k) = k̂_αk̂_β - δ_αβ/3`. Shared `_build_q_tensor!` helper used by both padded and unpadded paths. Uses Euler angle rotation for applying DDI potential (shared code with spin mixing).

**Losses** (`losses.jl`): density-dependent dipolar relaxation with m-dependent rate `γ_m = Γ_dr × (F+m)(F-m+1) / (2F(2F+1))`. m=-F is stable.

### Simulation Entry Points

- `find_ground_state(; grid, atom, ...)` — imaginary-time evolution
- `make_workspace(; ...) → Workspace` then `run_simulation!(ws)` — real-time dynamics
- `run_simulation_adaptive!(ws; adaptive, t_end, ...)` — adaptive dt with PI controller
- `load_experiment("path.yaml") → ExperimentConfig` then `run_experiment(config)` — YAML-driven
- `save_experiment_result(path, result)` / `load_experiment_result(path)` — JLD2 round-trip
- `examples/run_experiment.jl` — batch runner: pass a directory to run all YAMLs

### Tracing / Profiling

Uses TimerOutputs.jl with `@timeit_debug` (zero-cost when disabled):
```julia
enable_tracing!()   # compile-time enable
reset_tracing!()    # clear counters
# ... run simulation ...
println(TIMER)      # print results
disable_tracing!()
```

### Extensions

`ext/SpinorBECMakieExt/` and `ext/SpinorBECPlotlyExt/` provide visualization as weak-dep extensions.

## Performance Notes

- **SMatrix{D,D} at large D** (D≥~10): StaticArrays heap-allocates temporaries. Use `Matrix{ComplexF64}` for V_Fy eigendecomposition and `MVector{D}` for scratch buffers. Never use SMatrix{D,D} in hot loops for D=13.
- **`Threads.@threads` closures**: Can box captured untyped arguments, causing massive allocations (65 MiB/call). Prefer plain `@inbounds for` loops for element-wise operations, reserve threading for compute-heavy per-point work.
- **`Val(ndim)` vs `Val(N)`**: `Val(ndim::Int)` causes dynamic dispatch. Always use `Val(N)` from a type parameter (e.g., `Workspace{N}`).
- **`ntuple(f, ndim::Int)`**: Returns type-unstable tuple. Use `ntuple(f, Val(N))` for type-stable n_pts.

## Key Constraints

- New struct types must be defined in `types.jl` (included first, referenced everywhere)
- Julia 1.12: use inner constructors for normalization/validation (method overwriting forbidden during precompilation)
- Atoms defined in `atoms.jl` as constants (`Rb87`, `Na23`, `Eu151`)
- YAML configs in `examples/` follow the schema in `experiment.jl`
- Workspace is fully parameterized — auto-inferred constructor, no explicit type params needed
