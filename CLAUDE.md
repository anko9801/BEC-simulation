# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run all tests (4619 cases)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a single test file
julia --project=. -e 'using SpinorBEC; include("test/test_simulation.jl")'

# REPL with package loaded
julia --project=. -e 'using SpinorBEC'

# Install dependencies
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Architecture

SpinorBEC.jl simulates spin-F Bose-Einstein condensates using the split-step Fourier method in 1D/2D/3D.

### Dimensionless Units

All propagators use ℏ=m=1. Physical quantities are scaled via `harmonic_scales(mass, omega)` in `units.jl`. Kinetic step: `exp(-ik²dt/2)`.

### Include Order (in `SpinorBEC.jl`)

`types.jl` must be first — all struct definitions live there and are referenced by everything else. The rest follows dependency order: grid → spin matrices → spinor utils → atoms → interactions → potentials → zeeman → propagators → spin mixing → split step → raman → ddi → optics → observables → simulation → experiment → unitful.

### Core Types (`types.jl`)

- `Grid{N}` — N-dimensional spatial grid with FFT wavenumbers
- `SpinSystem(F)` — spin quantum number, `n_components = 2F+1`
- `SpinMatrices{D}` — static spin-F matrices (Fx, Fy, Fz, F·F) as `SMatrix`
- `SimState{N,A}` — mutable: wavefunction `psi`, time, step counter
- `Workspace{N,A,P,IP}` — immutable container holding all simulation state (grid, atom, interactions, zeeman, potential, sim_params, ddi, ddi_bufs, raman, fft_plans, spin_matrices)

### Wavefunction Layout

`psi` is an `Array{ComplexF64, N+1}` where spatial dimensions come first, spinor component last: `psi[x, y, ..., c]` for component `c ∈ 1:2F+1`.

Access helpers in `spinor_utils.jl`:
- `_component_slice(ndim, n_pts, c)` — indexing tuple for `view(psi, ...)`
- `_get_spinor(psi, I, n_comp)` → `SVector{n_comp}` at CartesianIndex `I`
- `_set_spinor!(psi, I, spinor, n_comp)` — write spinor back

### N-Dimensional Genericity

All loops use `CartesianIndices(n_pts)` — no specialized 1D/2D/3D code paths. Grid dimension `N` is a type parameter propagated through `Grid{N}`, `Workspace{N,...}`, etc.

### Split-Step Pipeline (`split_step.jl`)

Strang splitting (2nd order symmetric):
1. Half potential step (diagonal potential + spin mixing + DDI + Raman)
2. Full kinetic step (FFT → multiply phase → IFFT, per component)
3. Half potential step (mirror)

Spin mixing (`spin_mixing.jl`): skips entirely when `c1 ≈ 0` (e.g. Eu151). Uses Rodrigues' formula for spin-1, eigendecomposition (`_exp_i_hermitian`) for higher spins.

DDI (`ddi.jl`): k-space convolution with tensor `Q_αβ(k) = k̂_αk̂_β - δ_αβ/3`. Compute spin density → FFT → convolve → IFFT → apply as potential.

### Simulation Entry Points

- `find_ground_state(; grid, atom, ...)` — imaginary-time evolution
- `make_workspace(; ...) → Workspace` then `run_simulation!(ws)` — real-time dynamics
- `load_experiment("path.yaml") → ExperimentConfig` then `run_experiment(config)` — YAML-driven

### Extensions

`ext/SpinorBECMakieExt/` and `ext/SpinorBECPlotlyExt/` provide visualization as weak-dep extensions.

## Key Constraints

- New struct types must be defined in `types.jl` (included first, referenced everywhere)
- Julia 1.12: use inner constructors for normalization/validation (method overwriting forbidden during precompilation)
- Atoms defined in `atoms.jl` as constants (`Rb87`, `Na23`, `Eu151`)
- YAML configs in `examples/` follow the schema in `experiment.jl`
