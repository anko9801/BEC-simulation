const _ITP_EXPONENT_LIMIT = 50.0

function _validate_itp_zeeman(zeeman::ZeemanParams, F, dt)
    sys = SpinSystem(F)
    max_zee = maximum(abs, zeeman_energies(zeeman, sys))
    max_exponent = max_zee * dt / 4
    if max_exponent > _ITP_EXPONENT_LIMIT
        throw(ArgumentError(
            "Zeeman p=$(zeeman.p) with F=$F and dt=$dt causes overflow in imaginary time " *
            "(exponent=$(round(max_exponent, digits=1)) > $_ITP_EXPONENT_LIMIT). " *
            "Reduce p or dt. For ferromagnetic ground state, p=10 suffices."
        ))
    end
end

_validate_itp_zeeman(::TimeDependentZeeman, F, dt) = nothing

function _validate_itp_interactions(interactions::InteractionParams, F, dt; psi=nothing)
    max_c = max(abs(interactions.c0), abs(interactions.c1))
    max_c < 1e-30 && return nothing
    n_peak = if psi !== nothing
        ndim = ndims(psi) - 1
        Float64(maximum(total_density(psi, ndim)))
    else
        1.0
    end
    max_exponent = max_c * n_peak * dt / 4
    if max_exponent > _ITP_EXPONENT_LIMIT
        throw(ArgumentError(
            "Spin interaction c0=$(interactions.c0), c1=$(interactions.c1) with " *
            "n_peak≈$(round(n_peak, sigdigits=3)) and dt=$dt " *
            "may cause overflow in imaginary time (estimated exponent=$(round(max_exponent, digits=1)) " *
            "> $_ITP_EXPONENT_LIMIT). Reduce c0/c1 magnitude or dt."
        ))
    end
end

function find_ground_state(;
    grid,
    atom,
    interactions,
    zeeman=ZeemanParams(),
    potential=NoPotential(),
    dt=0.001,
    n_steps=10000,
    tol=1e-10,
    initial_state=:polar,
    psi_init=nothing,
    enable_ddi::Bool=false,
    c_dd::Float64=NaN,
    secular_ddi::Bool=false,
    adaptive_dt::Bool=false,
    dt_max::Float64=10.0 * dt,
    fft_flags=FFTW.MEASURE,
)
    psi0 = if psi_init === nothing
        sys = SpinSystem(atom.F)
        init_psi(grid, sys; state=initial_state)
    else
        copy(psi_init)
    end

    _validate_itp_zeeman(zeeman, atom.F, dt)
    _validate_itp_interactions(interactions, atom.F, dt; psi=psi0)

    if adaptive_dt
        return _find_ground_state_adaptive(;
            grid, atom, interactions, zeeman, potential,
            dt, n_steps, tol, psi0, enable_ddi, c_dd, secular_ddi, dt_max, fft_flags,
        )
    end

    sp = SimParams(; dt, n_steps, imaginary_time=true, normalize_every=1, save_every=max(1, n_steps ÷ 10))
    ws = make_workspace(; grid, atom, interactions, zeeman, potential, sim_params=sp, psi_init=psi0, enable_ddi, c_dd, secular_ddi, fft_flags)

    E_prev = total_energy(ws)
    converged = false

    for step in 1:n_steps
        split_step!(ws)
        if step % sp.save_every == 0
            E = total_energy(ws)
            dE = abs(E - E_prev)
            if dE < tol
                converged = true
                break
            end
            E_prev = E
        end
    end

    (workspace=ws, converged=converged, energy=total_energy(ws))
end

"""
Adaptive dt ground state search.

Strategy: run check_every steps, then evaluate energy.
- Energy decreased → grow dt by 10% (capped at dt_max)
- Energy increased → revert psi, halve dt, retry
"""
function _find_ground_state_adaptive(;
    grid, atom, interactions, zeeman, potential,
    dt, n_steps, tol, psi0, enable_ddi, c_dd, secular_ddi=false, dt_max, fft_flags=FFTW.MEASURE,
)
    current_dt = dt
    check_every = max(1, n_steps ÷ 100)
    psi_current = copy(psi0)
    psi_backup = similar(psi0)

    sp = SimParams(; dt=current_dt, n_steps=check_every, imaginary_time=true,
                   normalize_every=1, save_every=check_every)
    ws = make_workspace(; grid, atom, interactions, zeeman, potential,
                        sim_params=sp, psi_init=psi_current, enable_ddi, c_dd, secular_ddi, fft_flags)
    E_prev = total_energy(ws)
    converged = false
    total_steps = 0

    while total_steps < n_steps
        copyto!(psi_backup, ws.state.psi)

        for _ in 1:check_every
            split_step!(ws)
        end
        total_steps += check_every

        E = total_energy(ws)

        if E > E_prev
            copyto!(ws.state.psi, psi_backup)
            current_dt = max(current_dt * 0.5, 1e-8)
            ws = _rebuild_workspace_with_dt(ws, current_dt)
        else
            dE = abs(E - E_prev)
            if dE < tol
                converged = true
                break
            end
            E_prev = E
            new_dt = min(current_dt * 1.1, dt_max)
            if new_dt != current_dt
                current_dt = new_dt
                ws = _rebuild_workspace_with_dt(ws, current_dt)
            end
        end
    end

    (workspace=ws, converged=converged, energy=total_energy(ws))
end

function _rebuild_workspace_with_dt(ws::Workspace{N}, new_dt::Float64) where {N}
    sp = SimParams(new_dt, ws.sim_params.n_steps, true,
                   ws.sim_params.normalize_every, ws.sim_params.save_every)
    kinetic_phase = prepare_kinetic_phase(ws.grid, new_dt; imaginary_time=true)
    batched_kinetic = _make_batched_kinetic_cache(ws.state.psi, kinetic_phase, N)

    Workspace(
        ws.state, ws.fft_plans, kinetic_phase, ws.potential_values, ws.density_buf,
        ws.spin_matrices, ws.grid, ws.atom, ws.interactions,
        ws.zeeman, ws.potential, sp, ws.ddi, ws.ddi_bufs, ws.raman, ws.loss,
        ws.ddi_padded, batched_kinetic, ws.tensor_cache,
    )
end
