# SpinorBEC.jl

A Julia package for simulating spin-$F$ Bose-Einstein condensates by solving the spinor Gross-Pitaevskii equation in 1D/2D/3D via the split-step Fourier method.

## Features

- **Arbitrary spin $F$**: spin matrices built from angular momentum algebra (`StaticArrays` for stack allocation)
- **N-dimensional**: unified code path for 1D / 2D / 3D via `CartesianIndices`
- **Split-step Fourier**: 2nd-order Strang splitting with nested substeps (potential, spin mixing, DDI, Raman)
- **Ground state search**: imaginary-time propagation with convergence criterion
- **Real-time dynamics**: multi-phase sequences, time-dependent Zeeman ramps, leapfrog fusion, adaptive $\Delta t$
- **Potentials**: harmonic trap, gravity, crossed dipole trap (Gaussian beams), laser beam potential, composites
- **DDI**: dipolar interaction via $k$-space tensor convolution ($Q_{\alpha\beta}$), optimized to 6 FFTs per step
- **Raman coupling**: two-photon transitions with spatially dependent matrix exponential
- **Gaussian beam optics**: complex beam parameter $q$, ABCD matrix propagation, mode coupling
- **Thomas-Fermi initialization**: chemical potential bisection for density profiles
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

```
1. Half potential step (dt/2):
   a. 1/4 diagonal potential (trap + Zeeman + c0 * density)
   b. 1/2 spin mixing (c1 interaction, matrix exponential)
   c. [DDI substep (if enabled)]
   d. [Raman substep (if enabled)]
   e. 1/4 diagonal potential (density recomputed)
2. Full kinetic step (dt):
   FFT → multiply by exp(-ik²dt/2) → IFFT
3. Half potential step (dt/2):
   mirror of step 1
```

Spin mixing is automatically skipped when $c_1 \approx 0$ (e.g., ${}^{151}$Eu). Uses Rodrigues' formula for spin-1; eigendecomposition via `_exp_i_hermitian` for higher spins.

For real-time dynamics, a leapfrog-fused loop merges adjacent half potential steps $V(\Delta t/2) + V(\Delta t/2) = V(\Delta t)$ between time steps, splitting only at snapshot save points. Kinetic steps use batched FFT over all spinor components.

### Imaginary-Time Propagation (Ground State Search)

- Replace $e^{-iH\Delta t} \to e^{-H\Delta t}$ (Wick rotation)
- Renormalize $\psi$ after each step
- Convergence when $|\Delta E| < \mathrm{tol}$
- Initial states: `:polar` ($m=0$), `:ferromagnetic` ($m = +F$), `:uniform` (equal weight)
- Thomas-Fermi initialization: `init_psi_thomas_fermi` constructs the density profile from the chemical potential

### Real-Time Dynamics

- Multi-phase sequences (output of phase $n$ feeds into phase $n+1$)
- `TimeDependentZeeman` for linear ramps of $p(t)$, $q(t)$
- Callback functions for intermediate state access
- Adaptive time step support

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
| Total energy | `total_energy(ws)` | $E_{\mathrm{kin}} + E_{\mathrm{trap}} + E_{\mathrm{Zee}} + E_{c_0} + E_{c_1} + E_{\mathrm{ddi}}$ |
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

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
