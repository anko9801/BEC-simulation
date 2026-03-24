# SpinorBEC.jl Architecture

## Overview

SpinorBEC.jl is a Julia package for simulating spinor Bose-Einstein condensates (BECs) using the split-step Fourier method. It solves the spin-F Gross-Pitaevskii equation (GPE) in 1D, 2D, or 3D, supporting contact interactions, Zeeman effects, dipole-dipole interactions (DDI), and various external potentials.

The spatial dimensionality `N` is handled generically via Julia's parametric types and `CartesianIndices`, with no dimension-specific code paths.

## Module Structure

```
src/
  SpinorBEC.jl       # Module definition, includes, exports
  types.jl           # All struct definitions (must be included first)
  units.jl           # Physical constants and unit conversion
  grid.jl            # Spatial/momentum grid construction
  atoms.jl           # Predefined atom species (Rb87, Na23, Eu151)
  interactions.jl    # Contact interaction parameters (c0, c1, C_dd)
  potentials.jl      # External potential evaluation
  optical_trap.jl    # Gaussian beam and crossed dipole trap
  zeeman.jl          # Zeeman energy shifts
  spin_matrices.jl   # Spin-F matrix construction via StaticArrays
  propagators.jl     # Kinetic and diagonal potential propagators
  spin_mixing.jl     # Spin-dependent interaction propagator
  ddi.jl             # Dipole-dipole interaction via k-space convolution
  split_step.jl      # Strang splitting orchestration
  observables.jl     # Energy, density, magnetization, spin density
  simulation.jl      # Workspace, ground state finder, time evolution
  io.jl              # JLD2 save/load
  experiment.jl      # YAML-driven experiment configuration and runner
```

## Core Data Flow

```
GridConfig -> Grid (x, k, k^2 arrays)
AtomSpecies -> SpinSystem -> SpinMatrices (Fx, Fy, Fz as SMatrix)
                          -> InteractionParams (c0, c1)
AbstractPotential -> evaluate_potential -> potential_values array
SimParams + all above -> Workspace (holds everything for simulation)

Workspace -> split_step! (one time step)
          -> find_ground_state (imaginary time propagation)
          -> run_simulation! (real time evolution, returns SimulationResult)
```

## Key Types

### Grid (`types.jl`, `grid.jl`)

`GridConfig{N}` specifies the number of grid points (must be positive even integers) and box size per dimension. `Grid{N}` holds the computed real-space coordinates `x`, momentum-space coordinates `k`, grid spacings `dx`/`dk`, and the precomputed `k_squared` array.

The grid is centered at the origin: `x` ranges from `[-L/2 + dx/2, L/2 - dx/2]`. Momentum-space vectors use `FFTW.fftfreq`.

### Spin System (`types.jl`, `spin_matrices.jl`)

`SpinSystem` stores the total spin `F`, the number of components `2F+1`, and the magnetic quantum numbers `m = F, F-1, ..., -F`.

`SpinMatrices{D,M}` holds the spin-F operators `Fx`, `Fy`, `Fz`, `F+`, `F-`, and `F.F` as `SMatrix` (StaticArrays) for compile-time-sized, stack-allocated matrix operations. The raising/lowering operators are constructed from the standard angular momentum algebra:

```
F+|F,m> = sqrt(F(F+1) - m(m+1)) |F,m+1>
```

### Atom Species (`types.jl`, `atoms.jl`)

`AtomSpecies` stores mass, spin `F`, scattering lengths `a0`/`a2`, and magnetic dipole moment `mu_mag`. Three atoms are predefined:

| Atom | F | Scattering lengths | Dipolar |
|------|---|--------------------|---------|
| Rb87 | 1 | a0=101.8 a_B, a2=100.4 a_B | No |
| Na23 | 1 | a0=50.0 a_B, a2=55.0 a_B | No |
| Eu151 | 6 | a_s=110.0 a_B | Yes (7 mu_B) |

### Interactions (`interactions.jl`)

For spin-1 BECs, the contact interaction is parameterized by:

```
c0 = 4pi hbar^2 (a0 + 2*a2) / (3m)    (density-density)
c1 = 4pi hbar^2 (a2 - a0) / (3m)       (spin-spin)
```

- `c1 < 0`: ferromagnetic (Rb87)
- `c1 > 0`: antiferromagnetic (Na23)

Quasi-low-dimensional reductions (1D, 2D) divide by the transverse confinement area/length. For general F, only `c0` (s-wave) is computed.

The DDI coupling constant is `C_dd = mu_0 * mu^2 / (4pi)`.

### Potentials (`types.jl`, `potentials.jl`, `optical_trap.jl`)

All potentials inherit from `AbstractPotential` and implement `evaluate_potential(pot, grid) -> Array{Float64,N}`.

| Type | Formula | Parameters |
|------|---------|------------|
| `NoPotential` | V = 0 | None |
| `HarmonicTrap{N}` | V = 0.5 * sum(omega_d^2 * x_d^2) | omega per axis |
| `GravityPotential{N}` | V = g * x[axis] | g, axis (validated: 1 <= axis <= N) |
| `CrossedDipoleTrap{N}` | V = -alpha * sum(I_beam) | GaussianBeam list, polarizability |
| `CompositePotential{N}` | V = sum(V_component) | Vector of AbstractPotential |

`GaussianBeam` models a focused laser beam with wavelength, power, waist (1/e^2 radius), position, and propagation direction. The intensity profile neglects axial (Rayleigh range) variation, using only the transverse Gaussian: `I(r) = I0 * exp(-2 * r_perp^2 / w0^2)`.

### Zeeman Effect (`types.jl`, `zeeman.jl`)

`ZeemanParams` holds the linear (`p`) and quadratic (`q`) Zeeman coefficients. The energy shift for magnetic sublevel `m` is:

```
E_m = -p * m + q * m^2
```

`TimeDependentZeeman` wraps a function `t -> ZeemanParams`, enabling time-varying magnetic fields. The `zeeman_at(z, t)` dispatch handles both static and dynamic cases.

### Units (`units.jl`)

The `Units` submodule defines SI constants (hbar, AMU, Bohr radius, Bohr magneton, mu_0, k_B) and provides `DimensionlessScales` for converting between SI and simulation units based on a harmonic oscillator reference (`mass`, `omega`).

## Simulation Engine

### Wavefunction Layout

The wavefunction `psi` is stored as a single `Array{ComplexF64, N+1}` with shape `(n_x, [n_y, [n_z,]] n_components)`. The spatial dimensions come first, and the last axis indexes the spin components `m = F, F-1, ..., -F`.

Component `c` is accessed via `_component_slice(ndim, n_pts, c)`, which returns a tuple of index ranges suitable for `view(psi, idx...)`.

### Split-Step Method (`split_step.jl`)

Each time step uses Strang (symmetric) splitting with a nested sub-splitting for the potential part:

```
1. Half potential step:
   a. Quarter diagonal potential (trap + Zeeman + c0*density)
   b. Half spin-mixing (c1 * F.F interaction)
   c. [DDI sub-step if enabled]
   d. Quarter diagonal potential (recomputed density)
2. Full kinetic step (FFT -> multiply by exp(-i k^2 dt/2) -> IFFT)
3. Half potential step (symmetric repeat of step 1)
```

For imaginary time evolution, all `exp(-i H dt)` become `exp(-H dt)`, and the wavefunction is renormalized after each step.

### Kinetic Propagator (`propagators.jl`)

The kinetic phase `exp(-i k^2 dt/2)` (or `exp(-k^2 dt/2)` for imaginary time) is precomputed once and stored in the workspace. Each component of psi is independently FFT'd, multiplied by this phase, and inverse-FFT'd. The FFT plans are created via FFTW's in-place planning.

### Diagonal Potential Propagator (`propagators.jl`)

Applies `exp(-i (V_trap + E_Zeeman(m) + c0*n_total) * dt)` to each spin component. The total density `n_total = sum_m |psi_m|^2` is computed on the fly.

### Spin-Mixing Propagator (`spin_mixing.jl`)

At each spatial point, the local spinor is extracted as an `SVector{n_comp, ComplexF64}`. The local spin expectation values `<Fx>`, `<Fy>`, `<Fz>` are computed, and the spin Hamiltonian `H_spin = c1 * (<F> . F)` is constructed as an `SMatrix`. The propagator `exp(-i H_spin dt)` is computed via eigendecomposition of this small Hermitian matrix (`_exp_i_hermitian`), and the rotated spinor is written back.

### DDI Propagator (`ddi.jl`)

The dipole-dipole interaction is handled via k-space convolution:

1. Compute the local spin density `(Fx(r), Fy(r), Fz(r))` at each grid point.
2. FFT each component to k-space.
3. Convolve with the DDI kernel `Q_ab(k) = k_a * k_b / k^2 - delta_ab / 3` (precomputed at workspace creation).
4. IFFT to get the dipolar mean-field `Phi_alpha(r)`.
5. At each spatial point, construct `H_ddi = Phi_x Fx + Phi_y Fy + Phi_z Fz` and apply the matrix exponential.

The `DDIParams{N}` stores `C_dd` and the six independent components of the symmetric `Q` tensor. `DDIBuffers{N}` provides pre-allocated arrays for the spin density and convolution intermediates.

## Observables (`observables.jl`)

| Observable | Function | Description |
|------------|----------|-------------|
| Total density | `total_density(psi, ndim)` | sum_m \|psi_m\|^2 |
| Component density | `component_density(psi, ndim, c)` | \|psi_c\|^2 |
| Total norm | `total_norm(psi, grid)` | integral of total density |
| Magnetization | `magnetization(psi, grid, sys)` | integral of sum_m m\|psi_m\|^2 |
| Spin density | `spin_density_vector(psi, sm, ndim)` | (Fx(r), Fy(r), Fz(r)) |
| Total energy | `total_energy(ws)` | E_kin + E_trap + E_Zee + E_c0 + E_c1 + E_ddi |

The kinetic energy is computed in momentum space: `E_kin = 0.5 * integral(k^2 * |psi_k|^2)`. All other energy contributions are computed in real space.

## High-Level API (`simulation.jl`)

### `make_workspace`

Constructs the `Workspace` struct, which bundles all simulation state and parameters. Accepts keyword arguments for grid, atom, interactions, Zeeman, potential, simulation parameters, and optional DDI configuration. If no initial wavefunction is provided, `init_psi` generates a Gaussian in one of three configurations: `:polar` (m=0 only), `:ferromagnetic` (m=+F only), or `:uniform` (equal population).

### `find_ground_state`

Uses imaginary time propagation with per-step renormalization. Convergence is checked every `save_every` steps by comparing successive total energies against a tolerance `tol`. Returns a named tuple `(workspace, converged, energy)`.

### `run_simulation!`

Runs real-time evolution for `n_steps` steps, recording observables (time, energy, norm, magnetization) and wavefunction snapshots at intervals specified by `save_every`. Supports an optional callback function. Returns a `SimulationResult`.

## Experiment System (`experiment.jl`)

The experiment system provides a YAML-driven interface for defining multi-phase simulation protocols.

### Configuration Hierarchy

```
ExperimentConfig
  +-- SystemConfig (atom, grid, interactions, DDI)
  +-- GroundStateConfig (imaginary time parameters, potential, Zeeman)
  +-- PhaseConfig[] (sequence of real-time evolution phases)
```

Each `PhaseConfig` specifies duration, time step, Zeeman parameters (constant or linear ramp), and an optional potential override. If a phase omits the potential, it inherits from the previous phase or ground state.

### YAML Format

```yaml
experiment:
  name: "example"
  system:
    atom: Rb87              # Rb87 | Na23 | Eu151
    grid:
      n_points: [64, 64]
      box_size: [20.0, 20.0]
    interactions:
      c0: 10.0
      c1: -0.5
    ddi:                     # optional
      enabled: true
      c_dd: 1.5e-5
  ground_state:              # optional
    dt: 0.005
    n_steps: 1000
    tol: 1.0e-8
    initial_state: polar     # polar | ferromagnetic | uniform
    zeeman: { p: 0.0, q: 0.1 }
    potential:
      type: harmonic
      omega: [1.0, 1.0]
  sequence:
    - name: quench
      duration: 1.0
      dt: 0.01
      save_every: 10
      zeeman:
        p: { from: 0.0, to: 0.5 }   # linear ramp
        q: 0.0                        # constant
      potential:                       # optional override
        type: harmonic
        omega: [1.0, 1.0]
```

### Potential Specification in YAML

Single potential (Dict form):

```yaml
potential:
  type: harmonic
  omega: [1.0, 1.0]
```

```yaml
potential:
  type: gravity
  g: 9.81          # default: 9.81
  axis: 2          # default: last axis
```

```yaml
potential:
  type: crossed_dipole
  polarizability: 1.5e-37
  beams:
    - wavelength: 1064.0e-9
      power: 10.0
      waist: 50.0e-6
      position: [0, 0, 0]
      direction: [1, 0, 0]
```

Composite potential (List form, automatically summed):

```yaml
potential:
  - type: harmonic
    omega: [1.0, 1.0, 1.0]
  - type: gravity
    g: 9.81
    axis: 3
```

### Execution Flow

`run_experiment(config)`:

1. Resolve atom species from registry.
2. Build grid from config.
3. If `ground_state` is specified, run `find_ground_state` with imaginary time propagation.
4. For each phase in `sequence`:
   - Build potential (or inherit from previous phase).
   - Build Zeeman (static or time-dependent ramp).
   - Create workspace with the ground state wavefunction (or previous phase output) as initial condition.
   - Run real-time simulation.
   - Chain the output wavefunction to the next phase.
5. Return `ExperimentResult` with all phase results.

## I/O (`io.jl`)

State serialization uses JLD2 format, saving the wavefunction array, time, step count, and grid parameters. This enables checkpoint/restart workflows.

## Dependencies

| Package | Purpose |
|---------|---------|
| FFTW | Fast Fourier transforms (in-place, planned) |
| StaticArrays | Stack-allocated spin matrices and spinors |
| LinearAlgebra | Eigendecomposition for matrix exponentials |
| JLD2 | Binary state serialization |
| YAML | Experiment configuration parsing |
| PlotlyJS | Visualization (weak extension) |
| Makie | Visualization (weak extension) |
