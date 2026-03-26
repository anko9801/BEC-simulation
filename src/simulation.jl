function run_simulation!(ws::Workspace{N};
    callback::Union{Nothing,Function}=nothing,
) where {N}
    sp = ws.sim_params
    sys = ws.spin_matrices.system
    it = sp.imaginary_time

    times = Float64[]
    energies = Float64[]
    norms = Float64[]
    mags = Float64[]
    snapshots = Array{ComplexF64}[]
    _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

    if it
        _run_simulation_standard!(ws, sp, sys, times, energies, norms, mags, snapshots; callback)
    else
        _run_simulation_leapfrog!(ws, sp, sys, times, energies, norms, mags, snapshots; callback)
    end

    SimulationResult(times, energies, norms, mags, snapshots)
end

function _run_simulation_standard!(ws::Workspace{N}, sp, sys, times, energies, norms, mags, snapshots;
    callback=nothing,
) where {N}
    for step in 1:sp.n_steps
        split_step!(ws)

        if step % sp.save_every == 0
            _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

            if callback !== nothing
                callback(ws, step)
            end
        end
    end
end

"""
Leapfrog-fused simulation loop for real-time dynamics.

Merges adjacent half potential steps V(dt/2)+V(dt/2)=V(dt) between time steps.
Splits at snapshot save points to ensure saved states are proper Strang-split results.
Mathematically identical to standard loop — no accuracy change.

Also uses batched FFT for kinetic step (all components in one FFTW call).
"""
function _run_simulation_leapfrog!(ws::Workspace{N}, sp, sys, times, energies, norms, mags, snapshots;
    callback=nothing,
) where {N}
    dt = sp.dt
    n_comp = sys.n_components
    bk = ws.batched_kinetic

    # Leapfrog: initial half potential step
    _half_potential_step!(ws, dt / 2, n_comp, N, false)

    for step in 1:sp.n_steps
        apply_kinetic_step_batched!(ws.state.psi, bk)

        is_save = (step % sp.save_every == 0)
        is_last = (step == sp.n_steps)
        need_split = is_save || is_last

        if need_split
            # Close current step with half potential
            _half_potential_step!(ws, dt / 2, n_comp, N, false)
        else
            # Merged: close current + open next = full potential step
            _half_potential_step!(ws, dt, n_comp, N, false)
        end

        ws.state.t += dt
        ws.state.step += 1

        if is_save
            _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

            if callback !== nothing
                callback(ws, step)
            end
        end

        # Open next step if we split
        if need_split && !is_last
            _half_potential_step!(ws, dt / 2, n_comp, N, false)
        end
    end
end
