# SpinorBEC.jl

A Julia package for simulating spin-$F$ Bose-Einstein condensates by solving the spinor Gross-Pitaevskii equation in 1D/2D/3D via the split-step Fourier method.

## Features

- **Arbitrary spin $F$**: spin matrices built from angular momentum algebra (`StaticArrays` for stack allocation)
- **N-dimensional**: unified code path for 1D / 2D / 3D via `CartesianIndices`
- **Split-step Fourier**: 2nd-order Strang splitting with nested substeps (potential, spin mixing, DDI, Raman)
- **Ground state search**: imaginary-time propagation with convergence criterion
- **Real-time dynamics**: multi-phase sequences, time-dependent Zeeman ramps, leapfrog fusion, adaptive $\Delta t$
- **Potentials**: harmonic trap, gravity, crossed dipole trap (Gaussian beams), laser beam potential, composites
- **DDI**: dipolar interaction via $k$-space tensor convolution ($Q_{\alpha\beta}$), optimized to 6 FFTs per step, optional zero-padded convolution for reduced aliasing
- **Raman coupling**: two-photon transitions with spatially dependent matrix exponential
- **Gaussian beam optics**: complex beam parameter $q$, ABCD matrix propagation, mode coupling
- **Thomas-Fermi initialization**: chemical potential bisection for density profiles
- **LHY correction**: beyond-mean-field Lee-Huang-Yang term $\propto n^{5/2}$
- **Nematic interaction**: singlet pair amplitude $A_{00}$ and $c_2$ nematic energy
- **Dipolar relaxation losses**: $m$-dependent loss rates, 3-body loss
- **Yoshida 4th-order integrator**: adaptive time stepping with embedded Strang error estimator
- **Topological observables**: Berry curvature, superfluid vorticity, skyrmion charge, Majorana star representation
- **YAML experiment config**: declarative, reproducible multi-phase simulations
- **Unitful support**: direct input of physical quantities with units
- **Built-in atom species**: Rb87 ($F=1$, ferromagnetic), Na23 ($F=1$, antiferromagnetic), Eu151 ($F=6$, dipolar)

## Installation

```julia
using Pkg
Pkg.develop(path="path/to/BEC-simulation")
```

## Quick Start

### Julia API

```julia
using SpinorBEC

grid = make_grid(GridConfig((64,), (20.0,)))
atom = Rb87
interactions = InteractionParams(10.0, -0.5)
potential = HarmonicTrap((1.0,))

result = find_ground_state(;
    grid, atom, interactions, potential,
    dt=0.005, n_steps=5000, tol=1e-10,
    initial_state=:polar,
)

println("Converged: $(result.converged), E = $(result.energy)")
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
using SpinorBEC
config = load_experiment("experiment.yaml")
result = run_experiment(config)
```

## Physical Model

### Hamiltonian

The full spinor BEC Hamiltonian is:

$$H = \sum_m \int \psi_m^{*} \left[ -\frac{\hbar^2 \nabla^2}{2M} + V(\mathbf{r}) - pm + qm^2 + c_0 n(\mathbf{r}) + c_1 \langle\mathbf{F}\rangle \cdot \mathbf{F} + H_{\mathrm{ddi}} + H_{\mathrm{Raman}} \right] \psi_m \, d^3r$$

| Term | Description | Implementation |
|------|-------------|----------------|
| $-\frac{\hbar^2 \nabla^2}{2M}$ | Free-particle kinetic energy | FFT to $k$-space, multiply by $k^2$ |
| $V(\mathbf{r})$ | External trapping potential | Diagonal in real space |
| $-pm$ | Linear Zeeman shift | Constant shift per component |
| $qm^2$ | Quadratic Zeeman shift | Constant shift per component |
| $c_0 n$ | Spin-independent contact interaction | Multiply by $c_0 |\psi|^2$ in real space |
| $c_1 \langle\mathbf{F}\rangle \cdot \mathbf{F}$ | Spin-dependent contact interaction | Matrix exponential at each grid point |
| $H_{\mathrm{ddi}}$ | Long-range anisotropic dipolar interaction | $k$-space convolution: $Q_{\alpha\beta}(\mathbf{k}) = \hat{k}_\alpha \hat{k}_\beta - \delta_{\alpha\beta}/3$ |
| $c_2 \|A_{00}\|^2$ | Nematic (singlet pair) interaction | $A_{00} = \sum_m (-1)^{F-m} \psi_m \psi_{-m} / \sqrt{2F+1}$ |
| $c_{\mathrm{LHY}} n^{5/2}$ | Lee-Huang-Yang beyond-mean-field correction | $V_{\mathrm{LHY}} = c_{\mathrm{LHY}} n \sqrt{n}$ in diagonal step |
| $H_{\mathrm{Raman}}$ | Two-photon Raman transition | Spatially dependent matrix exponential: $(\Omega_R/2)(e^{i\mathbf{k}\cdot\mathbf{r}} F_+ + \text{h.c.}) + \delta F_z$ |

### Interaction Parameters

For spin-1 BECs, the interaction strengths are determined by $s$-wave scattering lengths $a_0$ and $a_2$ (total spin $F_{\mathrm{tot}} = 0, 2$ channels):

$$c_0 = \frac{4\pi\hbar^2 (a_0 + 2a_2)}{3M}, \qquad c_1 = \frac{4\pi\hbar^2 (a_2 - a_0)}{3M}$$

- $c_1 < 0$: ferromagnetic (${}^{87}$Rb)
- $c_1 > 0$: antiferromagnetic (${}^{23}$Na)

For quasi-low-dimensional systems (1D, 2D), transverse confinement provides dimensional reduction. The DDI coupling constant is $C_{\mathrm{dd}} = \mu_0 \mu^2 / (4\pi)$.

### Supported Atom Species

| Atom | Spin $F$ | Scattering lengths | Magnetic moment | Character |
|------|----------|-------------------|-----------------|-----------|
| ${}^{87}$Rb | 1 | $a_0 = 101.8\,a_B$, $a_2 = 100.4\,a_B$ | — | Ferromagnetic ($c_1 < 0$) |
| ${}^{23}$Na | 1 | $a_0 = 50.0\,a_B$, $a_2 = 55.0\,a_B$ | — | Antiferromagnetic ($c_1 > 0$) |
| ${}^{151}$Eu | 6 | $a_s = 110.0\,a_B$ | $7\,\mu_B$ | Dipolar ($\varepsilon_{\mathrm{dd}} \approx 0.55$) |

Spin matrices for arbitrary $F$ are constructed from angular momentum algebra.

## Numerical Methods

### Strang Splitting (2nd-order Symmetric)

Each time step $S_2(\Delta t)$ consists of:

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

All non-commuting operators within the potential step are symmetrized for 2nd-order accuracy. DDI is innermost (most expensive: 6 FFTs). Each substep is skipped when its coupling constant is negligible (e.g., spin mixing when $c_1 \approx 0$, nematic when $c_2 = 0$).

**Spin mixing** (`spin_mixing.jl`): $D=3$: Rodrigues' formula (machine-precision unitarity). $D>3$: Euler angle decomposition — $O(D)$ spin expectation via raising/lowering operators and $O(D^2)$ rotation via cached $F_y$ eigendecomposition.

**Nematic** (`nematic.jl`): Bogoliubov-type coupling of $(m, -m)$ pairs via singlet pair amplitude $A_{00}$, conserving $|\psi_m|^2 + |\psi_{-m}|^2$ per pair.

**DDI** (`ddi.jl` + `ddi_padded.jl`): $k$-space convolution with $Q_{\alpha\beta}(\mathbf{k}) = \hat{k}_\alpha \hat{k}_\beta - \delta_{\alpha\beta}/3$, applied via Euler angle spin rotation. Optional zero-padded convolution (2$\times$ grid in each dim) for reduced aliasing.

**Kinetic step** (`propagators.jl`): Batched FFT — single forward/inverse FFT for all $D$ spinor components simultaneously (vs $D$ individual FFTs). Uses `BatchedKineticCache` with pre-allocated work array.

For real-time dynamics, a leapfrog-fused loop merges adjacent half potential steps $V(\Delta t/2) + V(\Delta t/2) = V(\Delta t)$ between time steps, splitting only at snapshot save points.

### Imaginary-Time Propagation (Ground State Search)

- Replace $e^{-iH\Delta t} \to e^{-H\Delta t}$ (Wick rotation)
- Renormalize $\psi$ after each step
- Convergence when $|\Delta E| < \mathrm{tol}$
- Initial states: `:polar` ($m=0$), `:ferromagnetic` ($m = +F$), `:uniform` (equal weight)
- Thomas-Fermi initialization: `init_psi_thomas_fermi` constructs the density profile from the chemical potential

### Yoshida 4th-Order Integrator

$S_4(\Delta t) = S_2(w_1 \Delta t) \circ S_2(w_0 \Delta t) \circ S_2(w_1 \Delta t)$ where $w_1 = 1/(2 - 2^{1/3})$, $w_0 = 1 - 2w_1$.

- Embedded error estimator: $\|S_4(\Delta t)\psi - S_2(\Delta t)\psi\|/\|\psi\|$
- PI controller: $(tol/err)^{1/(p+1)}$ with $p=4$
- Fixed-$\Delta t$ cost: 1.94$\times$ Strang (3K + 4V vs 1K + 2V)
- Adaptive benefit: 2--5$\times$ faster than adaptive Strang at same accuracy

### Adaptive Time Stepping

Both `run_simulation_adaptive!` (Strang) and `run_simulation_yoshida!` support adaptive $\Delta t$:

- Wavefunction L2 relative change as error estimator
- Step rejection when error exceeds tolerance
- FSAL (first same as last) optimization for Strang
- Configurable via `AdaptiveDtParams(dt_init, dt_min, dt_max, tol)`

Benchmark on ${}^{151}$Eu 3D (32$^3$, 5 ms, $c_1 = 0$):

| Tolerance | Yoshida steps | Strang steps | Speedup |
|-----------|--------------|-------------|---------|
| 0.05 | 71 | 532 | 2.3$\times$ |
| 0.01 | 77 | 1199 | 4.6$\times$ |
| 0.005 | 86 | 1699 | 4.9$\times$ |

### Real-Time Dynamics

- Multi-phase sequences (output of phase $n$ feeds into phase $n+1$)
- `TimeDependentZeeman` for linear ramps of $p(t)$, $q(t)$
- Noise seeding: `noise_amplitude` in YAML phase config breaks symmetry (required for e.g. ${}^{151}$Eu EdH instability)
- Callback functions for intermediate state access
- Adaptive time step with Strang or Yoshida integrators

## Potentials

| Type | YAML `type` | Formula | Parameters |
|------|-------------|---------|------------|
| None | `none` | $V = 0$ | — |
| Harmonic | `harmonic` | $V = \frac{1}{2}\sum_d \omega_d^2 x_d^2$ | `omega: [ω_x, ...]` |
| Gravity | `gravity` | $V = g \cdot x_{\mathrm{axis}}$ | `g` (default 9.81), `axis` |
| Crossed dipole | `crossed_dipole` | $V = -\alpha \sum_i I_{\mathrm{beam},i}$ | `polarizability`, `beams` |
| Laser beam | — | Gaussian beam intensity profile | `LaserBeamPotential` |
| Composite | list syntax | $V = \sum_i V_i$ | combination of the above |

Composite potentials are defined with YAML list syntax:

```yaml
potential:
  - type: harmonic
    omega: [1.0, 1.0, 1.0]
  - type: gravity
    g: 9.81
    axis: 3
```

## Gaussian Beam Optics

`OpticalBeam` implements exact Gaussian beam propagation via the complex beam parameter $q$.

```julia
beam = OpticalBeam(wavelength=1064e-9, power=1.0, waist=50e-6)
propagated = propagate(beam, abcd_free_space(0.1))

w  = waist_radius(beam)      # beam waist radius
zR = rayleigh_length(beam)    # Rayleigh length
I0 = peak_intensity(beam)     # peak intensity
```

ABCD matrices: `abcd_free_space`, `abcd_thin_lens`, `abcd_curved_mirror`, `abcd_flat_mirror`

Fiber coupling: `mode_overlap`, `fiber_coupling`

Unitful support: `OpticalBeam(wavelength=1064u"nm", power=1u"W", waist=50u"μm")`

## Observables

### Basic Quantities

| Observable | Function | Definition |
|------------|----------|------------|
| Total density | `total_density(psi, ndim)` | $n(\mathbf{r}) = \sum_m |\psi_m|^2$ |
| Component density | `component_density(psi, ndim, c)` | $|\psi_c|^2$ |
| Total norm | `total_norm(psi, grid)` | $\int n\, dV$ |
| Magnetization | `magnetization(psi, grid, sys)` | $\int \sum_m m\,|\psi_m|^2\, dV$ |
| Spin density vector | `spin_density_vector(psi, sm, ndim)` | $(\langle F_x \rangle, \langle F_y \rangle, \langle F_z \rangle)$ at each point |
| Total energy | `total_energy(ws)` | $E_{\mathrm{kin}} + E_{\mathrm{trap}} + E_{\mathrm{Zee}} + E_{c_0} + E_{c_1} + E_{\mathrm{ddi}} + E_{\mathrm{LHY}} + E_{c_2}$ |
| Singlet pair amplitude | `singlet_pair_amplitude(psi, F, ndim)` | $A_{00} = \sum_m (-1)^{F-m} \psi_m \psi_{-m} / \sqrt{2F+1}$ |
| Component populations | `component_populations(psi, grid, sys)` | Normalized occupation of each spin component |

### Hydrodynamic Quantities

| Observable | Function | Definition |
|------------|----------|------------|
| Probability current | `probability_current(psi, grid, plans)` | $\mathbf{j}(\mathbf{r}) = \sum_c \mathrm{Im}(\psi_c^{*} \nabla\psi_c)$ |
| Superfluid velocity | `superfluid_velocity(psi, grid, plans)` | $\mathbf{v}_s = \mathbf{j}/n$ |
| Orbital angular momentum | `orbital_angular_momentum(psi, grid, plans)` | $\langle L_z \rangle = \int \psi^{*}(-i)(x\partial_y - y\partial_x)\psi\, dV$ |
| Total angular momentum | `total_angular_momentum(psi, grid, plans, sys)` | $J_z = L_z + S_z$ |
| Superfluid vorticity | `superfluid_vorticity(psi, grid, plans)` | $\boldsymbol{\omega} = \nabla \times \mathbf{v}_s$ (2D: scalar, 3D: vector) |

### Topological Quantities

| Observable | Function | Definition |
|------------|----------|------------|
| Berry curvature | `berry_curvature(psi, grid, plans, sm)` | Mermin-Ho relation: $\Omega = \hat{\mathbf{s}} \cdot (\partial_i \hat{\mathbf{s}} \times \partial_j \hat{\mathbf{s}})$ |
| Skyrmion charge | `spin_texture_charge(psi, grid, plans, sm)` | $Q = \frac{1}{4\pi F} \int \Omega\, d^2r$ (2D only) |
| Majorana stars | `majorana_stars(spinor, F)` | Roots of the Majorana polynomial ($2F$ stars on $S^2$) |
| Icosahedral order | `icosahedral_order_parameter(psi, sm, ndim)` | Steinhardt $Q_6$ bond-order parameter ($F \geq 6$) |

## Diagnostics

| Function | Description |
|----------|-------------|
| `spin_mixing_period(c1, q)` | Spin mixing oscillation period (dimensionless) |
| `spin_mixing_period_si(c1, q)` | Spin mixing oscillation period (SI units) |
| `quadratic_zeeman_from_field(g_F, B, ΔE_hf)` | Quadratic Zeeman shift from magnetic field $B$ |
| `healing_length_contact(m, c0, n)` | Healing length for contact interaction: $\xi = 1/\sqrt{2mc_0 n}$ |
| `healing_length_spin(m, c1, n)` | Healing length for spin interaction: $\xi_s = 1/\sqrt{2m|c_1|n}$ |
| `healing_length_ddi(m, C_dd, n)` | Healing length for DDI: $\xi_d = 1/\sqrt{2mC_{\mathrm{dd}}n}$ |
| `thomas_fermi_radius(density, x)` | Extract Thomas-Fermi radius from density profile |
| `phase_diagram_point(...)` | Phase diagram coordinates $(R_{\mathrm{TF}}/\xi_s,\; R_{\mathrm{TF}}/\xi_d)$ |

## I/O

```julia
# Ground state search
result = find_ground_state(; grid, atom, interactions, potential,
    dt=0.005, n_steps=5000, tol=1e-10, initial_state=:polar)

# Real-time evolution
ws = make_workspace(; grid, atom, interactions, ...)
result = run_simulation!(ws; callback=nothing)

# State save/load (JLD2 format)
save_state("checkpoint.jld2", ws)
state = load_state("checkpoint.jld2")
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

`DimensionlessScales` provides conversion between SI and dimensionless harmonic oscillator units ($\hbar = m = 1$).

Unitful.jl quantities are accepted directly as input.

## Visualization (Weak-dependency Extensions)

| Extension | Capabilities |
|-----------|-------------|
| PlotlyJS | `plot_density`, `plot_spinor`, `plot_spin_texture`, `animate_dynamics` |
| Makie | 3D surfaces, volume rendering, real-time animation |

## Performance

### Large-$D$ Optimization ($D = 13$ for ${}^{151}$Eu)

`SMatrix{13,13,ComplexF64}` (2704 bytes, 169 elements) exceeds the StaticArrays stack threshold, causing heap allocation in tight loops. Key optimizations:

- **Spin expectation**: $O(D)$ raising/lowering operators for $\langle\mathbf{F}\rangle$ instead of $O(D^2)$ matrix-vector
- **Euler rotation**: `MVector{D}` scratch buffers and `Matrix` eigendecomposition (not `SMatrix`) — only 1 `SVector` at output
- **$F_y$ eigencache**: `SpinMatrices` stores `Fy_eigvecs`, `Fy_eigvecs_adj`, `Fy_eigvals` — avoids repeated eigendecomposition
- **cis recurrence**: $F_y$ eigenvalues are integers $(-F \ldots F)$ → `cis(m \cdot \theta) = cis(\theta)^m$, reducing 65 `cis` calls to 6 + recurrence for $D=13$

Result: 167 GiB $\to$ 43 MiB allocation, 698 $\to$ 122 ms/step (5.7$\times$ speedup) on ${}^{151}$Eu 32$^3$.

### General Optimizations

- **Batched FFT**: single forward + inverse FFT for all $D$ components (vs $2D$ individual FFTs)
- **DDI FFT reduction**: 6 FFTs per DDI step (FFT $F_x, F_y, F_z$ once, reuse in $k$-space)
- **Fused diagonal step**: single-pass loop combining trap + Zeeman + $c_0 n$ + LHY
- **FFTW planning**: `FFTW.MEASURE` flag for optimized FFT plans; `save_fftw_wisdom`/`load_fftw_wisdom` for persistence across sessions
- **`cis(-x)`** replaces `exp(-im*x)` in all propagators (avoids complex multiply)

### Performance Pitfalls

- **`Threads.@threads` closures** can box captured untyped arguments → massive allocations (65 MiB/call for $D=13$). Prefer plain `@inbounds for` loops for element-wise operations.
- **`Val(ndim::Int)`** causes dynamic dispatch. Always use `Val(N)` from a type parameter.
- **`ntuple(f, ndim::Int)`** returns type-unstable tuple. Use `ntuple(f, Val(N))`.

## Tracing / Profiling

All substeps in `split_step!` are instrumented with `@timeit_debug TIMER` from TimerOutputs.jl (zero-cost when disabled):

```julia
using SpinorBEC
enable_tracing!()    # compile-time enable
reset_tracing!()     # clear counters
# ... run simulation ...
println(TIMER)       # hierarchical timing report
disable_tracing!()
```

Benchmark scripts in `scripts/` (e.g., `bench_eu151.jl`) include tracing setup.

## Source Organization

| File | Lines | Responsibility |
|------|-------|---------------|
| `types.jl` | | All struct definitions (`GridConfig`, `Workspace`, `SimState`, etc.) |
| `units.jl` | | SI constants, `DimensionlessScales`, harmonic oscillator unit conversion |
| `grid.jl` | | `make_grid`, `make_fft_plans`, FFT wavenumber arrays |
| `spin_matrices.jl` | | `SpinMatrices{D}` construction, $F_x, F_y, F_z$ static matrices, $F_y$ eigencache |
| `spinor_utils.jl` | | `_get_spinor`/`_set_spinor!`, `_apply_euler_spin_rotation`, `_exp_i_hermitian` |
| `atoms.jl` | | `AtomSpecies` definitions: `Rb87`, `Na23`, `Eu151` |
| `interactions.jl` | | `InteractionParams`, `compute_c0`, `compute_c_dd`, `get_cn` |
| `potentials.jl` | | `HarmonicTrap`, `GravityPotential`, `CompositePotential`, `evaluate_potential` |
| `zeeman.jl` | | `ZeemanParams`, `TimeDependentZeeman`, `zeeman_diagonal` |
| `propagators.jl` | | `apply_kinetic_step!`, `apply_kinetic_step_batched!`, `apply_diagonal_potential_step!` |
| `spin_mixing.jl` | | $c_1$ spin-dependent interaction: Rodrigues' ($D=3$), Euler rotation ($D>3$) |
| `nematic.jl` | | $c_2$ singlet pair interaction: Bogoliubov $(m, -m)$ pair coupling |
| `losses.jl` | | Dipolar relaxation ($m$-dependent rates), 3-body loss |
| `split_step.jl` | | `split_step!`, `_half_potential_step!`, `_strang_core!`, `_yoshida_core!` |
| `raman.jl` | | `RamanCoupling`, two-photon Raman transition step |
| `ddi.jl` | | Core DDI: `_build_q_tensor!`, $k$-space convolution, unpadded step |
| `ddi_padded.jl` | | Zero-padded DDI convolution (2$\times$ grid, reduced aliasing) |
| `optical_trap.jl` | | `GaussianBeam`, `CrossedDipoleTrap` |
| `optics.jl` | | `OpticalBeam`, ABCD propagation, fiber coupling |
| `laser_potential.jl` | | `LaserBeamPotential`, crossed laser trap |
| `thomas_fermi.jl` | | Thomas-Fermi density profile, chemical potential bisection |
| `fft_utils.jl` | | `_fft_partial_derivative`, `_fft_gradient` (N-dim) |
| `observables.jl` | | Density, norm, magnetization, spin density, singlet pair amplitude |
| `energy.jl` | | `total_energy` and all energy component helpers |
| `currents.jl` | | Probability current, superfluid velocity, angular momentum |
| `vorticity.jl` | | Superfluid vorticity, Berry curvature, skyrmion charge |
| `diagnostics.jl` | | Healing lengths, spin mixing period, phase diagram coordinates |
| `majorana.jl` | | Majorana polynomial → stars on $S^2$, Steinhardt $Q_6$ order parameter |
| `simulation_utils.jl` | | `_record_snapshot!`, `_check_energy_drift`, shared simulation helpers |
| `initialization.jl` | | `init_psi`, initial state construction (polar, ferromagnetic, uniform) |
| `ground_state.jl` | | `find_ground_state` — imaginary-time propagation |
| `simulation.jl` | | `run_simulation!`, `make_workspace` |
| `adaptive.jl` | | Adaptive Strang integration with FSAL, PI controller |
| `yoshida.jl` | | Adaptive Yoshida 4th-order integration with embedded error estimator |
| `io.jl` | | `save_state`/`load_state` (JLD2 format) |
| `experiment.jl` | | YAML schema: `ExperimentConfig`, `PhaseConfig`, `GroundStateConfig` |
| `experiment_runner.jl` | | `run_experiment`, noise seeding, multi-phase execution |

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
