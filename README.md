# SpinorBEC.jl

A Julia package for simulating spinor Bose-Einstein condensates using the split-step Fourier method. Supports spin-F GPE in 1D/2D/3D with contact interactions, Zeeman effects, dipole-dipole interactions, and configurable external potentials.

## Features

- **Arbitrary spin F**: spin matrices constructed via angular momentum algebra using StaticArrays
- **N-dimensional**: 1D, 2D, 3D with a single generic code path (`CartesianIndices`)
- **Split-step Fourier**: Strang splitting with nested sub-steps for potential, spin-mixing, and DDI
- **Ground state search**: imaginary time propagation with convergence detection
- **Real-time dynamics**: multi-phase sequences with time-dependent Zeeman ramps
- **Potentials**: harmonic trap, gravity, crossed dipole trap (Gaussian beams), and composites
- **DDI**: dipole-dipole interaction via k-space convolution of the Q tensor
- **YAML experiments**: declarative configuration for reproducible multi-phase simulations
- **Predefined atoms**: Rb87 (F=1, ferromagnetic), Na23 (F=1, antiferromagnetic), Eu151 (F=6, dipolar)

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

### YAML Experiment

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

### Composite Potentials

Multiple potentials can be combined using YAML list syntax:

```yaml
potential:
  - type: harmonic
    omega: [1.0, 1.0, 1.0]
  - type: gravity
    g: 9.81
    axis: 3
```

## Available Potentials

| Type | YAML `type` | Parameters |
|------|-------------|------------|
| Harmonic trap | `harmonic` | `omega: [w_x, w_y, ...]` |
| No potential | `none` | (none) |
| Gravity | `gravity` | `g` (default 9.81), `axis` (default: last) |
| Crossed dipole | `crossed_dipole` | `polarizability`, `beams: [...]` |
| Composite | list syntax | automatically sums components |

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed documentation of the internal design, data flow, and numerical methods.

## Testing

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```
