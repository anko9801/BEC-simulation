module SpinorBEC

using LinearAlgebra
using StaticArrays
using FFTW
using JLD2
using YAML

include("types.jl")
include("units.jl")
include("grid.jl")
include("spin_matrices.jl")
include("spinor_utils.jl")
include("atoms.jl")
include("interactions.jl")
include("potentials.jl")
include("zeeman.jl")
include("propagators.jl")
include("spin_mixing.jl")
include("split_step.jl")
include("ddi.jl")
include("optical_trap.jl")
include("observables.jl")
include("simulation.jl")
include("io.jl")
include("experiment.jl")
include("experiment_runner.jl")

# Types
export GridConfig, Grid, SpinSystem, SpinMatrices
export AtomSpecies, InteractionParams, ZeemanParams
export SimParams, SimState, FFTPlans, Workspace
export HarmonicTrap, NoPotential, GravityPotential, CompositePotential

# Grid
export make_grid, make_fft_plans, cell_volume, n_spatial_points

# Spin
export spin_matrices

# Atoms
export Rb87, Na23, Eu151

# Interactions
export compute_interaction_params, compute_c0, compute_c_dd, compute_a_dd

# DDI
export DDIParams, DDIBuffers, make_ddi_params, make_ddi_buffers
export compute_ddi_potential!, apply_ddi_step!

# Potentials
export evaluate_potential

# Zeeman
export zeeman_diagonal, zeeman_energies, TimeDependentZeeman, zeeman_at

# Optical trap
export GaussianBeam, CrossedDipoleTrap

# Propagators
export apply_kinetic_step!, apply_diagonal_potential_step!

# Spin mixing
export apply_spin_mixing_step!

# Split-step
export split_step!, prepare_kinetic_phase

# Observables
export total_density, component_density, magnetization
export spin_density_vector, total_norm, total_energy

# Simulation
export find_ground_state, run_simulation!, make_workspace, init_psi

# I/O
export save_state, load_state

# Experiment
export ConstantValue, LinearRamp, RampOrConstant, interpolate_value
export PotentialConfig, PhaseConfig, GroundStateConfig, DDIConfig
export SystemConfig, ExperimentConfig, ExperimentResult
export load_experiment, load_experiment_from_string, run_experiment

# Units
export Units

# Visualization (defined in extension, exported here for discoverability)
function plot_density end
function plot_spinor end
function plot_spin_texture end
function animate_dynamics end
export plot_density, plot_spinor, plot_spin_texture, animate_dynamics

end # module
