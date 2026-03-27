function _update_kinetic_phase!(kinetic_phase, k_squared, dt)
    @. kinetic_phase = cis(-0.5 * k_squared * dt)
    nothing
end

function _psi_relative_change(psi_new, psi_old)
    diff_sq = 0.0
    old_sq = 0.0
    @inbounds for i in eachindex(psi_new, psi_old)
        d = psi_new[i] - psi_old[i]
        diff_sq += abs2(d)
        old_sq += abs2(psi_old[i])
    end
    sqrt(diff_sq / max(old_sq, 1e-300))
end

function _density_relative_change(psi_new, psi_old)
    diff_sq = 0.0
    old_sq = 0.0
    @inbounds for i in eachindex(psi_new, psi_old)
        dn = abs2(psi_new[i])
        do_ = abs2(psi_old[i])
        diff_sq += (dn - do_)^2
        old_sq += do_^2
    end
    sqrt(diff_sq / max(old_sq, 1e-300))
end

"""
Wavefunction L2 relative change: ‖ψ_new - ψ_old‖² / ‖ψ_old‖².

Captures spatially-varying phase changes that density estimators miss.
Kinetic steps, DDI rotations, and diagonal potential are all unitary
(density-preserving), so their splitting errors are invisible to density.
This estimator sees them via the full wavefunction difference.

For unitary evolution (‖ψ_new‖ = ‖ψ_old‖, exact for split-step):
    ‖Δψ‖²/‖ψ‖² = 2(1 - Re⟨ψ_old|ψ_new⟩/‖ψ‖²) ≈ ⟨δφ²⟩

Cost: identical to density estimator (one O(N) pass).
"""
function _wavefunction_l2_change(psi_new, psi_old)
    diff_sq = 0.0
    old_sq = 0.0
    @inbounds for i in eachindex(psi_new, psi_old)
        diff_sq += abs2(psi_new[i] - psi_old[i])
        old_sq += abs2(psi_old[i])
    end
    diff_sq / max(old_sq, 1e-300)
end

@inline function _flush_fsal!(ws::Workspace{N}, fsal_dt, n_comp, ndim) where {N}
    _half_potential_step!(ws, fsal_dt / 2, n_comp, ndim, false)
    nothing
end

function run_simulation_adaptive!(ws::Workspace{N};
    adaptive::AdaptiveDtParams=AdaptiveDtParams(),
    t_end::Float64,
    save_interval::Float64,
    callback::Union{Nothing,Function}=nothing,
) where {N}
    n_comp = ws.spin_matrices.system.n_components
    sys = ws.spin_matrices.system

    dt = clamp(adaptive.dt_init, adaptive.dt_min, adaptive.dt_max)
    current_kinetic_dt = NaN
    bk = ws.batched_kinetic

    times = Float64[]
    energies = Float64[]
    norms = Float64[]
    mags = Float64[]
    snapshots = Array{ComplexF64}[]
    _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)

    psi_old = similar(ws.state.psi)
    next_save = ws.state.t + save_interval
    n_accepted = 0
    n_rejected = 0

    fsal_deferred = false
    fsal_dt = 0.0

    while ws.state.t < t_end - 1e-14
        dt_step = min(dt, t_end - ws.state.t)
        remaining_to_save = next_save - ws.state.t
        if remaining_to_save > 1e-14 && remaining_to_save < dt_step
            dt_step = remaining_to_save
        end
        dt_step = max(dt_step, adaptive.dt_min)

        is_clamped = dt_step < dt * 0.99
        may_reject = !is_clamped && dt_step > adaptive.dt_min * 1.01

        if dt_step != current_kinetic_dt
            _update_batched_kinetic_phase!(bk, ws.grid.k_squared, dt_step)
            current_kinetic_dt = dt_step
        end

        need_error_est = !is_clamped
        if need_error_est
            psi_old .= ws.state.psi
        end

        prev_fsal_deferred = fsal_deferred
        prev_fsal_dt = fsal_dt

        if fsal_deferred && abs(fsal_dt - dt_step) < 1e-14
            _half_potential_step!(ws, dt_step, n_comp, N, false)
        elseif fsal_deferred
            _half_potential_step!(ws, fsal_dt / 2, n_comp, N, false)
            _half_potential_step!(ws, dt_step / 2, n_comp, N, false)
        else
            _half_potential_step!(ws, dt_step / 2, n_comp, N, false)
        end
        fsal_deferred = false

        apply_kinetic_step_batched!(ws.state.psi, bk)

        rel_change = 0.0
        if may_reject
            rel_change = _wavefunction_l2_change(ws.state.psi, psi_old)
            if rel_change > adaptive.tol
                ws.state.psi .= psi_old
                fsal_deferred = prev_fsal_deferred
                fsal_dt = prev_fsal_dt
                factor = max(0.5, 0.9 * sqrt(adaptive.tol / rel_change))
                dt = max(dt_step * factor, adaptive.dt_min)
                current_kinetic_dt = NaN
                n_rejected += 1
                continue
            end
        end

        fsal_deferred = true
        fsal_dt = dt_step

        if ws.loss !== nothing
            apply_loss_step!(ws.state.psi, ws.loss, ws.spin_matrices.system.F, dt_step, n_comp, N, ws.density_buf)
        end

        ws.state.t += dt_step
        ws.state.step += 1
        n_accepted += 1

        if !is_clamped
            if !may_reject
                rel_change = _wavefunction_l2_change(ws.state.psi, psi_old)
            end
            factor = rel_change > 1e-300 ? min(2.0, 0.9 * sqrt(adaptive.tol / rel_change)) : 2.0
            dt = clamp(dt * factor, adaptive.dt_min, adaptive.dt_max)
        end

        if ws.state.t >= next_save - 1e-14
            _flush_fsal!(ws, fsal_dt, n_comp, N)
            fsal_deferred = false

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

    if fsal_deferred
        _flush_fsal!(ws, fsal_dt, n_comp, N)
    end

    if isempty(times) || abs(times[end] - ws.state.t) > 1e-12
        _record_snapshot!(times, energies, norms, mags, snapshots, ws, sys)
    end

    (result=SimulationResult(times, energies, norms, mags, snapshots),
     n_accepted=n_accepted, n_rejected=n_rejected, final_dt=dt)
end
