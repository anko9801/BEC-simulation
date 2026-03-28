#!/usr/bin/env julia
# Phase diagram scan: sweep c1_ratio and classify ground state phases.
#
# Usage:
#   julia --project=. examples/eu151/phase_diagram_scan.jl
#
# Scans c1_ratio ∈ [-1/72, 1/36] (~10 points), finds ground states
# using multistart, classifies phases, and writes results to stdout.

using SpinorBEC
using FFTW

include(joinpath(@__DIR__, "eu151_params.jl"))

const GRID_N = 32
const BOX = 8.0
const DT = 0.005
const N_STEPS = 3000

const ratios = range(-1.0 / 72, 1.0 / 36; length=10)

function run_scan()
    grid = make_grid(GridConfig(GRID_N, BOX))
    trap = HarmonicTrap(1.0)
    sm = spin_matrices(6)

    println("c1_ratio,phase,energy,spin_order,nematic_order,initial_state")

    for ratio in ratios
        ip = eu_interaction_params(ratio)

        result = find_ground_state_multistart(;
            grid, atom=Eu151, interactions=ip, potential=trap,
            dt=DT, n_steps=N_STEPS,
            initial_states=[:polar, :ferromagnetic, :uniform],
            fft_flags=FFTW.ESTIMATE,
        )

        psi = result.workspace.state.psi
        phase_info = classify_phase(psi, 6, grid, sm)

        println(join([
            round(ratio; sigdigits=4),
            phase_info.phase,
            round(result.energy; sigdigits=6),
            round(phase_info.spin_order; sigdigits=4),
            round(phase_info.nematic_order; sigdigits=4),
            result.initial_state,
        ], ","))
    end
end

run_scan()
