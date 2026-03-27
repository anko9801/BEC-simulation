module SpinorBEC

using LinearAlgebra
using StaticArrays
using FFTW
using JLD2
using YAML
using Unitful
using TimerOutputs

const TIMER = TimerOutput()

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
include("nematic.jl")
include("losses.jl")
include("split_step.jl")
include("raman.jl")
include("ddi.jl")
include("ddi_padded.jl")
include("optical_trap.jl")
include("optics.jl")
include("laser_potential.jl")
include("thomas_fermi.jl")
include("fft_utils.jl")
include("observables.jl")
include("energy.jl")
include("currents.jl")
include("vorticity.jl")
include("diagnostics.jl")
include("majorana.jl")
include("simulation_utils.jl")
include("initialization.jl")
include("ground_state.jl")
include("simulation.jl")
include("adaptive.jl")
include("yoshida.jl")
include("io.jl")
include("experiment.jl")
include("experiment_runner.jl")
include("unitful_support.jl")

# Types
export GridConfig, Grid, SpinSystem, SpinMatrices
export AtomSpecies, InteractionParams, ZeemanParams, LossParams
export SimParams, SimState, FFTPlans, RFFTPlans, Workspace, AdaptiveDtParams
export HarmonicTrap, NoPotential, GravityPotential, CompositePotential

# Grid
export make_grid, make_fft_plans, make_rfft_plans, rfft_output_shape, cell_volume, n_spatial_points
export load_fftw_wisdom, save_fftw_wisdom

# Spin
export spin_matrices

# Atoms
export Rb87, Na23, Eu151

# Interactions
export compute_interaction_params, compute_c0, compute_c_dd, compute_a_dd

# DDI
export DDIParams, DDIBuffers, DDIPaddedContext, make_ddi_params, make_ddi_buffers, make_ddi_padded
export compute_ddi_potential!, apply_ddi_step!

# Potentials
export evaluate_potential

# Zeeman
export zeeman_diagonal, zeeman_energies, TimeDependentZeeman, zeeman_at

# Optical trap
export GaussianBeam, CrossedDipoleTrap

# Optics (Gaussian beam with complex q, ABCD)
export OpticalBeam, propagate, waist_radius, rayleigh_length
export radius_of_curvature, divergence_angle, peak_intensity, beam_intensity
export abcd_free_space, abcd_thin_lens, abcd_curved_mirror, abcd_flat_mirror
export mode_overlap, fiber_coupling

# Laser beam potential
export LaserBeamPotential, crossed_laser_trap

# Thomas-Fermi
export thomas_fermi_density, init_psi_thomas_fermi

# Propagators
export apply_kinetic_step!, apply_kinetic_step_batched!, apply_diagonal_potential_step!
export BatchedKineticCache

# Spin mixing
export apply_spin_mixing_step!

# Nematic
export apply_nematic_step!

# Losses
export apply_loss_step!

# Raman coupling
export RamanCoupling, apply_raman_step!

# Split-step
export split_step!, prepare_kinetic_phase

# Observables
export total_density, component_density, magnetization
export spin_density_vector, total_norm, total_energy
export probability_current, orbital_angular_momentum
export superfluid_velocity, total_angular_momentum, spin_texture_charge
export superfluid_vorticity, berry_curvature, singlet_pair_amplitude
export majorana_stars, icosahedral_order_parameter
export get_cn

# Diagnostics
export spin_mixing_period, spin_mixing_period_si, quadratic_zeeman_from_field
export healing_length_contact, healing_length_spin, healing_length_ddi
export thomas_fermi_radius, thomas_fermi_radius_harmonic
export phase_diagram_point, component_populations

# Simulation
export find_ground_state, run_simulation!, run_simulation_adaptive!, run_simulation_yoshida!, make_workspace, init_psi

# I/O
export save_state, load_state

# Experiment
export ConstantValue, LinearRamp, RampOrConstant, interpolate_value
export PotentialConfig, PhaseConfig, GroundStateConfig, DDIConfig
export SystemConfig, ExperimentConfig, ExperimentResult
export load_experiment, load_experiment_from_string, run_experiment

# Units
export Units

# Tracing
export TIMER, enable_tracing!, disable_tracing!, reset_tracing!

function enable_tracing!()
    TimerOutputs.enable_debug_timings(SpinorBEC)
    enable_timer!(TIMER)
end

function disable_tracing!()
    disable_timer!(TIMER)
end

function reset_tracing!()
    TimerOutputs.reset_timer!(TIMER)
end

# Visualization (defined in extension, exported here for discoverability)
function plot_density end
function plot_spinor end
function plot_spin_texture end
function animate_dynamics end
export plot_density, plot_spinor, plot_spin_texture, animate_dynamics

end # module
