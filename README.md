# SpinorBEC.jl

[![CI](https://github.com/anko9801/BEC-simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/anko9801/BEC-simulation/actions/workflows/ci.yml)

A Julia package for simulating spin- $F$ Bose-Einstein condensates by solving the spinor Gross-Pitaevskii equation in 1D/2D/3D via the split-step Fourier method.

**[Live Demo](https://anko9801.github.io/BEC-simulation/)** — Interactive Plotly visualizations of ${}^{151}\mathrm{Eu}$ BEC dynamics (density distributions, Einstein-de Haas effect).

## Features

### Core Simulation

- **Arbitrary spin $F$** — spin matrices from angular momentum algebra (`StaticArrays` for stack allocation)
- **N-dimensional** — unified code path for 1D / 2D / 3D via `CartesianIndices`
- **Split-step Fourier** — 2nd-order Strang splitting with nested symmetric substeps
- **Yoshida 4th-order integrator** — adaptive time stepping with embedded Strang error estimator
- **Adaptive $\Delta t$** — PI controller for both Strang and Yoshida integrators
- **Ground state search** — imaginary-time propagation with energy and wavefunction convergence
- **Multistart search** — tries multiple initial states, returns lowest energy
- **Constrained ITP** — target magnetization via Lagrange multiplier
- **Parameter continuation** — sweep with previous ground state as initial guess
- **Phase boundary bisection** — automatic phase transition detection
- **Checkpoint/restart** — save and resume long-running simulations

### Interactions

- **Contact interaction** — spin-independent ( $c_0 n$ ) and spin-dependent ( $c_1 \langle\mathbf{F}\rangle \cdot \mathbf{F}$ )
- **General- $F$ tensor interaction** — Clebsch-Gordan based mean-field for all scattering channels ( $S = 0, 2, \ldots, 2F$ ), 6j symbol transform from rank- $k$ tensor couplings
- **DDI** — dipolar interaction via $k$-space tensor convolution ( $Q_{\alpha\beta}$ ), 6 FFTs per step, optional zero-padded convolution
- **Nematic interaction** — singlet pair amplitude $A_{00}$ and $c_2$ nematic energy
- **LHY correction** — beyond-mean-field Lee-Huang-Yang term $\propto n^{5/2}$
- **Raman coupling** — two-photon transitions with spatially dependent matrix exponential
- **Dipolar relaxation losses** — $m$-dependent loss rates, 3-body loss

### Potentials

- **Harmonic trap**, gravity, crossed dipole trap (Gaussian beams), laser beam potential, composites

### Observables & Analysis

- **Phase classification** — spin order, nematic order, channel weights, automatic phase identification
- **Pair amplitudes** — general $A_{SM}(\mathbf{r})$ for all scattering channels
- **Nematic tensor eigenvalues** — biaxiality parameter for uniaxial/biaxial classification
- **Spherical harmonic representation** — spinor angular density $\rho(\theta, \phi)$
- **Bogoliubov-de Gennes spectrum** — uniform BdG dispersion $\omega(k)$ with instability detection
- **Time-of-flight / Stern-Gerlach imaging** — synthetic far-field momentum distributions
- **Power spectrum analysis** — FFT-based spectral analysis with windowing
- **Topological observables** — Berry curvature, superfluid vorticity, skyrmion charge, Majorana stars
- **Hydrodynamics** — probability current, superfluid velocity, orbital angular momentum

### Infrastructure

- **YAML experiment config** — declarative, reproducible multi-phase simulations
- **Unitful support** — direct input of physical quantities with units
- **Built-in atoms** — ${}^{87}\mathrm{Rb}$ ( $F{=}1$, ferromagnetic), ${}^{23}\mathrm{Na}$ ( $F{=}1$, antiferromagnetic), ${}^{151}\mathrm{Eu}$ ( $F{=}6$, dipolar)
- **Gaussian beam optics** — complex beam parameter $q$, ABCD matrix propagation, fiber coupling
- **PlotlyJS / Makie extensions** — weak-dependency visualization

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/BEC-simulation")
```

Requires Julia 1.10+. Dependencies are installed automatically via `Project.toml`.

## Quick Start

### Julia API

```julia
using SpinorBEC

# 1D spin-1 ground state
grid = make_grid(GridConfig((64,), (20.0,)))
interactions = InteractionParams(10.0, -0.5)

result = find_ground_state(;
    grid, atom=Rb87, interactions,
    potential=HarmonicTrap((1.0,)),
    dt=0.005, n_steps=5000, tol=1e-10,
    initial_state=:polar,
)
println("Converged: $(result.converged), E = $(result.energy)")

# Real-time dynamics
ws = make_workspace(;
    grid, atom=Rb87, interactions,
    potential=HarmonicTrap((1.0,)),
    zeeman=ZeemanParams(0.0, 0.1),
    sim_params=SimParams(dt=0.001, n_steps=5000),
    psi_init=result.workspace.state.psi,
)
run_simulation!(ws)
```

### Multistart Ground State Search

```julia
result = find_ground_state_multistart(;
    grid, atom=Na23,
    interactions=InteractionParams(10.0, 0.5),
    potential=HarmonicTrap((1.0,)),
    dt=0.005, n_steps=10000, tol=1e-10,
    initial_states=[:polar, :ferromagnetic, :uniform, :antiferromagnetic],
    n_random=3, seed=42,
)
println("Best: $(result.initial_state), E = $(result.energy)")
```

### Phase Diagram Scan

```julia
results = scan_continuation(;
    param_values=range(-1.0, 1.0, length=50),
    make_interactions=c1 -> InteractionParams(10.0, c1),
    grid, atom=Rb87,
    potential=HarmonicTrap((1.0,)),
    dt=0.005, n_steps_fresh=5000,
)
for r in results
    println("c1=$(r.param), E=$(r.energy), phase=$(r.phase)")
end
```

### Bogoliubov-de Gennes Spectrum

```julia
bdg = bogoliubov_spectrum(;
    spinor=[0.0+0im, 1.0+0im, 0.0+0im],  # polar state
    n0=1.0, F=1,
    interactions=InteractionParams(10.0, 0.5),
    k_max=10.0, n_k=200,
)
println("Unstable: $(bdg.unstable), max growth rate: $(bdg.max_growth_rate)")
```

### Time-of-Flight Imaging

```julia
images = simulate_tof(psi, grid, SpinSystem(1),
    TOFParams(t_tof=20.0, gradient=0.5, imaging_axis=3))
# images[m] = 2D momentum density for component m
```

### YAML Experiment Config

```yaml
experiment:
  name: "Rb87 quench dynamics"
  system:
    atom: Rb87
    grid:
      n_points: [256]
      box_size: [30.0]
    interactions:
      c0: 10.0
      c1: -0.5

  ground_state:
    dt: 0.005
    n_steps: 10000
    tol: 1.0e-10
    initial_state: ferromagnetic
    zeeman: { p: 0.0, q: 0.1 }
    potential:
      type: harmonic
      omega: [1.0]

  sequence:
    - name: ramp_field
      duration: 5.0
      dt: 0.001
      save_every: 100
      zeeman:
        p: 0.0
        q: { from: 0.1, to: -0.5 }

    - name: free_expansion
      duration: 2.0
      dt: 0.0005
      save_every: 50
      zeeman: { p: 0.0, q: 0.0 }
      potential:
        type: none
```

```julia
config = load_experiment("experiment.yaml")
result = run_experiment(config)
```

## Physical Model

### Hamiltonian

The full spinor BEC Hamiltonian:

$$H = \sum_m \int \psi_m^{*} \left[ -\frac{\hbar^2 \nabla^2}{2M} + V(\mathbf{r}) - pm + qm^2 + c_0 n + c_1 \langle\mathbf{F}\rangle \cdot \mathbf{F} + H_{\mathrm{ddi}} + H_{\mathrm{Raman}} \right] \psi_m \, d^3r$$

| Term | Description | Implementation |
|------|-------------|----------------|
| Kinetic $-\frac{\hbar^2 \nabla^2}{2M}$ | Free-particle kinetic energy | FFT to $k$-space, multiply by $k^2$ |
| $V(\mathbf{r})$ | External trapping potential | Diagonal in real space |
| $-pm$ | Linear Zeeman shift | Constant shift per component |
| $qm^2$ | Quadratic Zeeman shift | Constant shift per component |
| $c_0 n$ | Spin-independent contact | Multiply by total density in real space |
| $c_1 \langle\mathbf{F}\rangle \cdot \mathbf{F}$ | Spin-dependent contact | Matrix exponential at each grid point |
| $H_{\mathrm{ddi}}$ | Dipolar interaction | $k$-space convolution with $Q_{\alpha\beta}(\mathbf{k})$ |
| $c_2 \lvert A_{00} \rvert^2$ | Nematic (singlet pair) | Bogoliubov $(m, -m)$ pair coupling |
| $\sum_S g_S \lvert A_{SM} \rvert^2$ | General- $F$ tensor (all channels) | CG-based mean-field $h_{mm'}$, eigendecomposition |
| $c_{\mathrm{LHY}} n^{5/2}$ | Lee-Huang-Yang correction | Beyond-mean-field in diagonal step |
| $H_{\mathrm{Raman}}$ | Two-photon Raman | Spatially dependent matrix exponential |

### Interaction Parameters

**Spin-1** ( $F{=}1$ ): from $s$-wave scattering lengths $a_0$ and $a_2$ :

$$c_0 = \frac{4\pi\hbar^2 (a_0 + 2a_2)}{3M}, \qquad c_1 = \frac{4\pi\hbar^2 (a_2 - a_0)}{3M}$$

- $c_1 < 0$ : ferromagnetic ( ${}^{87}\mathrm{Rb}$ )
- $c_1 > 0$ : antiferromagnetic ( ${}^{23}\mathrm{Na}$ )

**General $F$** : channel couplings $g_S$ for total pair-spin $S = 0, 2, \ldots, 2F$ :

$$g_S = c_0 + c_1 \frac{S(S+1) - 2F(F+1)}{2}$$

Higher-rank tensor couplings $c_k$ ( $k = 4, 6, \ldots, 2F$ ) are converted to $g_S$ via 6j symbols. When a `TensorInteractionCache` is active, it handles all contact interactions and replaces the separate $c_0$ / $c_1$ / nematic steps.

**Constraint-based parameterization** (for atoms like ${}^{151}\mathrm{Eu}$ where individual $a_S$ are unknown): `interaction_params_from_constraint(; c_total, c1_ratio, F, c_extra)` constructs $c_0, c_1$ from the constraint $c_0 + F^2 c_1 = c_{\mathrm{total}}$ , with optional higher-rank $c_k$ via `c_extra`.

### Supported Atom Species

| Atom | Spin $F$ | $g_F$ | Scattering lengths | Magnetic moment | Character |
|------|----------|-------|-------------------|-----------------|-----------|
| ${}^{87}\mathrm{Rb}$ | 1 | $-1/2$ | $a_0 = 101.8\,a_B$ , $a_2 = 100.4\,a_B$ | — | Ferromagnetic ( $c_1 < 0$ ) |
| ${}^{23}\mathrm{Na}$ | 1 | $-1/2$ | $a_0 = 50.0\,a_B$ , $a_2 = 55.0\,a_B$ | — | Antiferromagnetic ( $c_1 > 0$ ) |
| ${}^{151}\mathrm{Eu}$ | 6 | $7/6$ | $a_s = 110.0\,a_B$ (channels unknown) | $7\,\mu_B$ | Dipolar ( $\varepsilon_{\mathrm{dd}} \approx 0.55$ ) |

## Numerical Methods

### Strang Splitting (2nd-order Symmetric)

Each time step $S_2(\Delta t)$ :

```
1. Half potential step V(dt/2) — symmetric inner splitting:
     diag(dt/4) → SM(dt/4) → nematic(dt/4) → Raman(dt/4)
       → DDI(dt/2) →
     Raman(dt/4) → nematic(dt/4) → SM(dt/4) → diag(dt/4)

2. Full kinetic step K(dt):
     Batched FFT → multiply by exp(-ik²dt/2) → batched IFFT

3. Half potential step V(dt/2):
     mirror of step 1

4. Loss step (real-time only, if enabled)
```

All non-commuting operators within the potential step are symmetrized for 2nd-order accuracy. DDI is innermost (most expensive: 6 FFTs). Each substep is skipped when its coupling constant is negligible.

When a `TensorInteractionCache` is active, the **tensor interaction step replaces both spin mixing and nematic** in the inner splitting to avoid double-counting. The tensor step builds the Hermitian mean-field matrix $h_{mm'}$ from all active scattering channels and applies $\exp(-ih\,dt)$ via eigendecomposition at each grid point, using thread-local pre-allocated buffers.

**Spin mixing** (`spin_mixing.jl`): $D{=}3$ uses Rodrigues' formula (machine-precision unitarity). $D{>}3$ uses Euler angle decomposition — $O(D)$ spin expectation via raising/lowering operators and $O(D^2)$ rotation via cached $F_y$ eigendecomposition.

**Nematic** (`nematic.jl`): Bogoliubov-type coupling of $(m, -m)$ pairs via singlet pair amplitude $A_{00}$ , conserving total norm.

**Tensor** (`tensor_interaction.jl`): General- $F$ contact interaction for all channels $S = 0, 2, \ldots, 2F$ . Per grid point: $h_{mm'} = \sum_S g_S \sum_\mu \langle m,\mu|S,M\rangle\langle m',\nu|S,M\rangle\, \psi_\mu^* \psi_\nu$ , then $\psi \to e^{-ih\,dt}\psi$ .

**DDI** (`ddi.jl` + `ddi_padded.jl`): $k$-space convolution with $Q_{\alpha\beta}(\mathbf{k}) = \hat{k}_\alpha \hat{k}_\beta - \delta_{\alpha\beta}/3$ , applied via Euler angle spin rotation. Optional zero-padded convolution (2x grid in each dim) for reduced aliasing. Secular approximation available for $\omega_L \gg c_{\mathrm{dd}} n$ .

**Kinetic step** (`propagators.jl`): Batched FFT — single forward/inverse FFT for all $D$ components simultaneously (vs $D$ individual FFTs). Uses `BatchedKineticCache` with pre-allocated work array.

**Losses** (`losses.jl`): $m$-dependent dipolar relaxation rates from rank-2 DDI tensor selection rules ( $\Delta m = -1, -2$ ), with $m = -F$ stable.

For real-time dynamics, a leapfrog-fused loop merges adjacent half potential steps between time steps, splitting only at snapshot save points.

### Imaginary-Time Propagation (Ground State Search)

- Replace $e^{-iH\Delta t} \to e^{-H\Delta t}$ (Wick rotation)
- Renormalize $\psi$ after each step
- Convergence when both $|\Delta E| < \mathrm{tol}$ and $\|\Delta\psi\| < \mathrm{tol}$
- Initial states: `:polar` ( $m{=}0$ ), `:ferromagnetic` ( $m{=}{+}F$ ), `:antiferromagnetic`, `:uniform`, `:random` (seeded), `:spin_helix`
- Thomas-Fermi initialization: `init_psi_thomas_fermi` constructs the density profile from the chemical potential
- **Multistart**: `find_ground_state_multistart` tries multiple initial states and returns the lowest energy
- **Constrained**: `target_magnetization` keyword constrains $\langle F_z \rangle$ via Lagrange multiplier weights

### Yoshida 4th-Order Integrator

$S_4(\Delta t) = S_2(w_1 \Delta t) \circ S_2(w_0 \Delta t) \circ S_2(w_1 \Delta t)$ where $w_1 = 1/(2 - 2^{1/3})$ , $w_0 = 1 - 2w_1$ .

- Embedded error estimator: $\|S_4(\Delta t)\psi - S_2(\Delta t)\psi\|/\|\psi\|$
- PI controller: $(tol/err)^{1/(p+1)}$ with $p{=}4$
- Fixed- $\Delta t$ cost: 1.94x Strang (3K + 4V vs 1K + 2V)
- Adaptive benefit: 2-5x faster than adaptive Strang at same accuracy

### Adaptive Time Stepping

Both `run_simulation_adaptive!` (Strang) and `run_simulation_yoshida!` support adaptive $\Delta t$ :

- Wavefunction L2 relative change as error estimator
- Step rejection when error exceeds tolerance
- FSAL (first same as last) optimization for Strang
- Configurable via `AdaptiveDtParams(dt_init, dt_min, dt_max, tol)`

Benchmark on ${}^{151}\mathrm{Eu}$ 3D ( $32^3$ , 5 ms, $c_1 = 0$ ):

| Tolerance | Yoshida steps | Strang steps | Speedup |
|-----------|--------------|-------------|---------|
| 0.05 | 71 | 532 | 2.3x |
| 0.01 | 77 | 1199 | 4.6x |
| 0.005 | 86 | 1699 | 4.9x |

### Checkpoint/Restart

```julia
run_simulation_checkpointed!(ws;
    checkpoint_dir="checkpoints",
    checkpoint_every=1000,
    callback=my_callback,
    resume=false,  # set true to resume from latest checkpoint
)
```

Saves state to JLD2 files at regular intervals. On `resume=true`, finds the latest `checkpoint_dir/step_NNNNNN.jld2` and restores state.

## Observables

### Basic Quantities

| Observable | Function | Definition |
|------------|----------|------------|
| Total density | `total_density(psi, ndim)` | $n(\mathbf{r}) = \sum_m \lvert\psi_m\rvert^2$ |
| Component density | `component_density(psi, ndim, c)` | $\lvert\psi_c\rvert^2$ |
| Total norm | `total_norm(psi, grid)` | $\int n\, dV$ |
| Magnetization | `magnetization(psi, grid, sys)` | $\int \sum_m m\,\lvert\psi_m\rvert^2\, dV$ |
| Spin density vector | `spin_density_vector(psi, sm, ndim)` | $(\langle F_x \rangle, \langle F_y \rangle, \langle F_z \rangle)$ at each point |
| Total energy | `total_energy(ws)` | Sum of all energy components |
| Component populations | `component_populations(psi, grid, sys)` | Normalized occupation of each spin component |

### Pair Amplitudes & Phase Classification

| Observable | Function | Description |
|------------|----------|-------------|
| Singlet pair amplitude | `singlet_pair_amplitude(psi, F, ndim)` | $A_{00} = \sum_m (-1)^{F-m} \psi_m \psi_{-m} / \sqrt{2F{+}1}$ |
| General pair amplitude | `pair_amplitude(psi, F, S, M, ndim, cg)` | $A_{SM}(\mathbf{r})$ for any even- $S$ channel |
| Pair amplitude spectrum | `pair_amplitude_spectrum(psi, F, grid)` | Integrated $\int\lvert A_{SM}\rvert^2 dV$ per channel |
| Phase classification | `classify_phase(psi, F, grid, sm)` | Returns spin order, nematic order, phase label |

Phase labels: `:ferromagnetic`, `:polar`, `:nematic`, `:cyclic`, `:mixed`.

### Nematic Tensor

| Observable | Function | Description |
|------------|----------|-------------|
| Nematic eigenvalues | `nematic_tensor_eigenvalues(psi, sm, ndim)` | Eigenvalues $(\lambda_1, \lambda_2, \lambda_3)$ of rank-2 nematic tensor |
| Biaxiality parameter | `biaxiality_parameter(l1, l2, l3)` | $\beta = (\lambda_2 - \lambda_3)/(\lambda_1 - \lambda_3)$ , 0 = uniaxial, 1 = biaxial |

### Spherical Harmonic Representation

```julia
# Compute Y_lm
ylm = spherical_harmonic(l, m, theta, phi)

# Spinor angular density: rho(theta, phi) = |sum_m Y_{Fm} zeta_m|^2
theta, phi, rho = spinor_angular_density(spinor, F; n_theta=64, n_phi=128)
```

### Hydrodynamic Quantities

| Observable | Function | Definition |
|------------|----------|------------|
| Probability current | `probability_current(psi, grid, plans)` | $\mathbf{j} = \sum_c \mathrm{Im}(\psi_c^{*} \nabla\psi_c)$ |
| Superfluid velocity | `superfluid_velocity(psi, grid, plans)` | $\mathbf{v}_s = \mathbf{j}/n$ |
| Orbital angular momentum | `orbital_angular_momentum(psi, grid, plans)` | $\langle L_z \rangle$ |
| Total angular momentum | `total_angular_momentum(psi, grid, plans, sys)` | $J_z = L_z + S_z$ |
| Superfluid vorticity | `superfluid_vorticity(psi, grid, plans)` | $\boldsymbol{\omega} = \nabla \times \mathbf{v}_s$ |

### Topological Quantities

| Observable | Function | Description |
|------------|----------|-------------|
| Berry curvature | `berry_curvature(psi, grid, plans, sm)` | Mermin-Ho relation |
| Skyrmion charge | `spin_texture_charge(psi, grid, plans, sm)` | $Q = \frac{1}{4\pi F} \int \Omega\, d^2r$ (2D) |
| Majorana stars | `majorana_stars(spinor, F)` | Roots of Majorana polynomial ( $2F$ stars on $S^2$ ) |
| Icosahedral order | `icosahedral_order_parameter(psi, sm, ndim)` | Steinhardt $Q_6$ ( $F \geq 6$ ) |

## Bogoliubov-de Gennes Spectrum

Uniform approximation: at each $k$ , build $2D \times 2D$ BdG matrix and diagonalize.

```julia
bdg = bogoliubov_spectrum(;
    spinor=[0, 1, 0] .+ 0im,  # normalized spinor
    n0=1.0,                     # condensate density
    F=1,
    interactions=InteractionParams(10.0, 0.5),
    zeeman=ZeemanParams(0.0, 0.1),
    c_dd=0.0,                   # DDI strength
    k_max=10.0, n_k=200,
    k_direction=(0.0, 0.0, 1.0),
)
# bdg.k_values, bdg.omega, bdg.max_growth_rate, bdg.unstable
```

**L matrix** (normal mean-field): kinetic + Zeeman + $2 n_0 h_{mm'}$ with CG contraction from tensor interaction pattern.

**M matrix** (anomalous pairing): $n_0 \sum_S g_S \sum_M \langle m,m'|S,M\rangle A_{SM}$ where $A_{SM}$ is the pair amplitude.

**DDI terms**: $n_0 c_{\mathrm{dd}} \sum_{ab} (F_a)_{mm'} Q_{ab}(\hat{k})$ added to both L and M.

Key physics checks: scalar BEC phonon dispersion, Goldstone mode at $k{=}0$ , particle-hole symmetry, instability detection via $\mathrm{Im}(\omega) > 0$ .

## Time-of-Flight / Stern-Gerlach Imaging

Far-field approximation: momentum distribution with magnetic gradient separation.

```julia
params = TOFParams(
    t_tof=20.0,        # expansion time
    gradient=0.5,       # Stern-Gerlach gradient
    imaging_axis=3,     # column integration axis
)
images = simulate_tof(psi, grid, SpinSystem(1), params)
# images[m] = 2D density for each m component
```

Per component: FFT $\to$ $\lvert\psi_k\rvert^2$ $\to$ shift by $m \cdot g \cdot t^2/2$ $\to$ column integrate.

## Potentials

| Type | YAML `type` | Formula | Parameters |
|------|-------------|---------|------------|
| None | `none` | $V = 0$ | — |
| Harmonic | `harmonic` | $V = \frac{1}{2}\sum_d \omega_d^2 x_d^2$ | `omega: [w_x, ...]` |
| Gravity | `gravity` | $V = g \cdot x_{\mathrm{axis}}$ | `g`, `axis` |
| Crossed dipole | `crossed_dipole` | $V = -\alpha \sum_i I_i$ | `polarizability`, `beams` |
| Laser beam | — | Gaussian beam intensity | `LaserBeamPotential` |
| Composite | list syntax | $V = \sum_i V_i$ | combination of the above |

Composite potentials via YAML list syntax:

```yaml
potential:
  - type: harmonic
    omega: [1.0, 1.0, 1.0]
  - type: gravity
    g: 9.81
    axis: 3
```

## Gaussian Beam Optics

```julia
beam = OpticalBeam(wavelength=1064e-9, power=1.0, waist=50e-6)
propagated = propagate(beam, abcd_free_space(0.1))

w  = waist_radius(beam)
zR = rayleigh_length(beam)
I0 = peak_intensity(beam)
```

ABCD matrices: `abcd_free_space`, `abcd_thin_lens`, `abcd_curved_mirror`, `abcd_flat_mirror`

Fiber coupling: `mode_overlap`, `fiber_coupling`

Unitful support: `OpticalBeam(wavelength=1064u"nm", power=1u"W", waist=50u"μm")`

## Diagnostics

| Function | Description |
|----------|-------------|
| `spin_mixing_period(c1, q)` | Spin mixing oscillation period |
| `quadratic_zeeman_from_field(g_F, B, dE_hf)` | Quadratic Zeeman shift from field $B$ |
| `healing_length_contact(m, c0, n)` | Contact healing length $\xi = 1/\sqrt{2mc_0 n}$ |
| `healing_length_spin(m, c1, n)` | Spin healing length $\xi_s = 1/\sqrt{2m\lvert c_1\rvert n}$ |
| `healing_length_ddi(m, C_dd, n)` | DDI healing length $\xi_d$ |
| `thomas_fermi_radius(density, x)` | Thomas-Fermi radius from density profile |
| `phase_diagram_point(...)` | Phase diagram coordinates |
| `power_spectrum(times, signal)` | FFT-based power spectrum with windowing |
| `estimate_splitting_error(ws)` | Richardson extrapolation of time-step error |
| `validate_conservation(ws)` | Check norm/energy/magnetization drift |

## Performance

### Large- $D$ Optimization ( $D{=}13$ for ${}^{151}\mathrm{Eu}$ )

`SMatrix{13,13,ComplexF64}` (2704 bytes) exceeds the StaticArrays stack threshold, causing heap allocation in tight loops. Key optimizations:

- **Spin expectation**: $O(D)$ raising/lowering operators for $\langle\mathbf{F}\rangle$ instead of $O(D^2)$ matrix-vector
- **Euler rotation**: `MVector` scratch buffers and `Matrix` eigendecomposition (not `SMatrix`) — only 1 `SVector` at output
- **$F_y$ eigencache**: `SpinMatrices` stores eigenvectors, adjoint, eigenvalues — avoids repeated eigendecomposition
- **cis recurrence**: $F_y$ eigenvalues are integers $(-F \ldots F)$ , so $\operatorname{cis}(m\theta) = \operatorname{cis}(\theta)^m$ reduces 65 `cis` calls to 6 + recurrence for $D{=}13$

Result: 167 GiB $\to$ 43 MiB allocation, 698 $\to$ 122 ms/step (5.7x speedup) on ${}^{151}\mathrm{Eu}$ $32^3$ .

### General Optimizations

- **Batched FFT**: single forward + inverse FFT for all $D$ components (vs $2D$ individual FFTs)
- **DDI FFT reduction**: 6 FFTs per DDI step (FFT $F_x, F_y, F_z$ once, reuse in $k$-space)
- **Fused diagonal step**: single-pass loop combining trap + Zeeman + $c_0 n$ + LHY
- **FFTW planning**: `FFTW.MEASURE` flag for optimized FFT plans; `save_fftw_wisdom` / `load_fftw_wisdom` for persistence
- **`cis(-x)`** replaces `exp(-im*x)` in all propagators

### Performance Pitfalls

- **`Threads.@threads` closures** can box captured untyped arguments (65 MiB/call for $D{=}13$ ). Prefer plain `@inbounds for` loops.
- **`Val(ndim::Int)`** causes dynamic dispatch. Always use `Val(N)` from a type parameter.
- **`ntuple(f, ndim::Int)`** returns type-unstable tuple. Use `ntuple(f, Val(N))`.

## Tracing / Profiling

All substeps in `split_step!` are instrumented with `@timeit_debug TIMER` from TimerOutputs.jl (zero-cost when disabled):

```julia
using SpinorBEC
enable_tracing!()
reset_tracing!()
# ... run simulation ...
println(TIMER)
disable_tracing!()
```

## I/O

```julia
# State save/load (JLD2 format)
save_state("checkpoint.jld2", ws)
state = load_state("checkpoint.jld2")

# Checkpointed simulation with auto-resume
run_simulation_checkpointed!(ws;
    checkpoint_dir="checkpoints",
    checkpoint_every=1000,
    resume=true,
)
```

## Units

The `Units` submodule provides SI constants:

| Constant | Symbol |
|----------|--------|
| Reduced Planck constant | `Units.HBAR` |
| Atomic mass unit | `Units.AMU` |
| Bohr radius | `Units.BOHR_RADIUS` |
| Bohr magneton | `Units.MU_BOHR` |
| Vacuum permeability | `Units.MU_0` |
| Boltzmann constant | `Units.K_B` |

`DimensionlessScales` converts between SI and dimensionless harmonic oscillator units ( $\hbar = m = 1$ ).

Unitful.jl quantities are accepted directly as input.

## Source Organization

| File | Lines | Responsibility |
|------|------:|---------------|
| `types.jl` | 403 | All struct definitions (`GridConfig`, `Workspace`, `SimState`, etc.) |
| `units.jl` | 48 | SI constants, `DimensionlessScales`, harmonic oscillator unit conversion |
| `grid.jl` | 69 | `make_grid`, `make_fft_plans`, FFT wavenumber arrays |
| `spin_matrices.jl` | 50 | `SpinMatrices` construction, $F_x, F_y, F_z$ , eigencache |
| `spinor_utils.jl` | 166 | `_get_spinor` / `_set_spinor!`, Euler rotation, `_exp_i_hermitian` |
| `clebsch_gordan.jl` | 180 | `wigner_3j`, `clebsch_gordan`, `wigner_6j`, `precompute_cg_table` |
| `atoms.jl` | 40 | `AtomSpecies` definitions: `Rb87`, `Na23`, `Eu151` |
| `interactions.jl` | 232 | `InteractionParams`, `compute_c0`, `compute_c_dd`, constraint parameterization |
| `potentials.jl` | 32 | `HarmonicTrap`, `GravityPotential`, `CompositePotential` |
| `zeeman.jl` | 21 | `ZeemanParams`, `TimeDependentZeeman`, `zeeman_diagonal` |
| `propagators.jl` | 166 | Kinetic step (batched + individual), diagonal potential step |
| `spin_mixing.jl` | 98 | $c_1$ interaction: Rodrigues' ( $D{=}3$ ), Euler rotation ( $D{>}3$ ) |
| `nematic.jl` | 119 | $c_2$ singlet pair interaction |
| `tensor_interaction.jl` | 312 | General- $F$ CG-based mean-field, eigendecomposition propagator |
| `losses.jl` | 82 | Dipolar relaxation, 3-body loss |
| `split_step.jl` | 195 | `split_step!`, Strang core, Yoshida core |
| `raman.jl` | 31 | Two-photon Raman transition |
| `ddi.jl` | 201 | Core DDI: $Q_{\alpha\beta}$ tensor, $k$-space convolution |
| `ddi_padded.jl` | 144 | Zero-padded DDI convolution |
| `optical_trap.jl` | 44 | `GaussianBeam`, `CrossedDipoleTrap` |
| `optics.jl` | 104 | `OpticalBeam`, ABCD propagation, fiber coupling |
| `laser_potential.jl` | 54 | `LaserBeamPotential`, crossed laser trap |
| `thomas_fermi.jl` | 77 | Thomas-Fermi density profile, chemical potential bisection |
| `fft_utils.jl` | 44 | `_fft_partial_derivative`, `_fft_gradient` (N-dim) |
| `observables.jl` | 246 | Density, norm, magnetization, pair amplitudes, phase classification, nematic tensor |
| `energy.jl` | 135 | `total_energy` and all energy component helpers |
| `currents.jl` | 107 | Probability current, superfluid velocity, angular momentum |
| `vorticity.jl` | 95 | Superfluid vorticity, Berry curvature, skyrmion charge |
| `diagnostics.jl` | 363 | Healing lengths, spin mixing period, power spectrum, conservation validation |
| `majorana.jl` | 132 | Majorana stars, Steinhardt $Q_6$ order parameter |
| `spherical_harmonics.jl` | 86 | $Y_{lm}$ , spinor angular density $\rho(\theta,\phi)$ |
| `simulation_utils.jl` | 26 | Snapshot recording, shared simulation helpers |
| `initialization.jl` | 191 | `init_psi`, `make_workspace`, initial state construction |
| `ground_state.jl` | 347 | `find_ground_state`, multistart, continuation scan |
| `simulation.jl` | 148 | `run_simulation!`, `run_simulation_checkpointed!` |
| `adaptive.jl` | 183 | Adaptive Strang integration with FSAL, PI controller |
| `yoshida.jl` | 98 | Adaptive Yoshida 4th-order integration |
| `tof.jl` | 72 | Time-of-flight / Stern-Gerlach imaging |
| `bogoliubov.jl` | 194 | Bogoliubov-de Gennes spectrum (uniform approximation) |
| `phase_boundary.jl` | 58 | Phase boundary bisection search |
| `io.jl` | 101 | `save_state` / `load_state` (JLD2 format) |
| `experiment.jl` | 248 | YAML schema: `ExperimentConfig`, `PhaseConfig` |
| `experiment_runner.jl` | 176 | `run_experiment`, noise seeding, multi-phase execution |
| `unitful_support.jl` | 67 | Unitful.jl quantity conversion |
| **Total** | **6180** | |

Tests: 7244 cases in 6598 lines (`test/`).

## Examples

| Script | Description |
|--------|-------------|
| `examples/eu151/eu151_64.jl` | 3D ${}^{151}\mathrm{Eu}$ BEC simulation ( $64^3$ ) |
| `examples/eu151/edh_full_64.jl` | Einstein-de Haas dynamics (40 ms, per-component) |
| `examples/eu151/stern_gerlach_3d.jl` | 3D Stern-Gerlach Plotly visualization |
| `examples/eu151/vortex_3d.jl` | 3D vortex core WebGL visualization |
| `examples/eu151/phase_diagram_scan.jl` | Phase diagram $c_1$ sweep with classification |
| `examples/eu151/bench_eu151.jl` | Performance benchmark with tracing |
| `examples/configs/run_experiment.jl` | YAML batch runner |

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## License

MIT
