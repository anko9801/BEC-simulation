"""
Adaptive 4th-order Yoshida integration with embedded Strang error estimator.

Uses S₄(dt) as propagator and ‖S₄(dt)ψ - S₂(dt)ψ‖/‖ψ‖ as error estimate.
The difference is dominated by the 3rd-order Strang error, giving a reliable
measure of splitting error that the L2 estimator misses.

PI controller exponent: (tol/err)^{1/(p+1)} with p=4 (4th-order global).
Cost: ~4 Strang steps per accepted step; benefits from larger dt at same accuracy.
"""
function run_simulation_yoshida!(ws::Workspace{N};
    adaptive::AdaptiveDtParams=AdaptiveDtParams(),
    t_end::Float64,
    save_interval::Float64,
    callback::Union{Nothing,Function}=nothing,
) where {N}
    n_comp = ws.spin_matrices.system.n_components
    sys = ws.spin_matrices.system

    dt = clamp(adaptive.dt_init, adaptive.dt_min, adaptive.dt_max)

    psi_old = similar(ws.state.psi)
    psi_strang = similar(ws.state.psi)

    times = Float64[]
    energies = Float64[]
    norms = Float64[]
    mags = Float64[]
    snapshots = Array{ComplexF64}[]
    _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

    next_save = ws.state.t + save_interval
    n_accepted = 0
    n_rejected = 0

    while ws.state.t < t_end - 1e-14
        dt_step = min(dt, t_end - ws.state.t)
        remaining_to_save = next_save - ws.state.t
        if remaining_to_save > 1e-14 && remaining_to_save < dt_step
            dt_step = remaining_to_save
        end
        dt_step = max(dt_step, adaptive.dt_min)

        is_clamped = dt_step < dt * 0.99

        psi_old .= ws.state.psi

        _strang_core!(ws, dt_step, n_comp)
        psi_strang .= ws.state.psi

        ws.state.psi .= psi_old
        _yoshida_core!(ws, dt_step, n_comp)

        err = _wavefunction_l2_change(ws.state.psi, psi_strang)

        if !is_clamped && err > adaptive.tol && dt_step > adaptive.dt_min * 1.01
            ws.state.psi .= psi_old
            factor = max(0.3, 0.9 * (adaptive.tol / err)^0.2)
            dt = max(dt_step * factor, adaptive.dt_min)
            n_rejected += 1
            continue
        end

        if ws.loss !== nothing
            apply_loss_step!(ws.state.psi, ws.loss, sys.F, dt_step, n_comp, N, ws.density_buf)
        end

        ws.state.t += dt_step
        ws.state.step += 1
        n_accepted += 1

        if !is_clamped
            factor = err > 1e-300 ? min(3.0, 0.9 * (adaptive.tol / err)^0.2) : 3.0
            dt = clamp(dt * factor, adaptive.dt_min, adaptive.dt_max)
        end

        if ws.state.t >= next_save - 1e-14
            E_now = total_energy(ws)
            nrm_now = total_norm(ws.state.psi, ws.grid)
            _check_energy_drift(energies, norms, E_now, nrm_now, ws.state.t)

            push!(times, ws.state.t)
            push!(energies, E_now)
            push!(norms, nrm_now)
            push!(mags, magnetization(ws.state.psi, ws.grid, sys))
            push!(snapshots, copy(ws.state.psi))
            callback !== nothing && callback(ws, n_accepted)
            next_save += save_interval
        end
    end

    if isempty(times) || abs(times[end] - ws.state.t) > 1e-12
        _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)
    end

    (result=SimulationResult(times, energies, norms, mags, snapshots),
     n_accepted=n_accepted, n_rejected=n_rejected, final_dt=dt)
end
