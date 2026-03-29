const _ITP_EXPONENT_LIMIT = 50.0
const _ITP_DDI_WARN_EXPONENT = 20.0

function _check_itp_overflow(ws, step::Int)
    if any(isnan, ws.state.psi)
        throw(ArgumentError(
            "NaN detected in ITP at step $step. " *
            "Likely DDI potential overflow. Reduce dt."
        ))
    end
    ws.ddi === nothing && return nothing
    bufs = ws.ddi_bufs
    phi_max = max(
        maximum(abs, bufs.Phi_x),
        maximum(abs, bufs.Phi_y),
        maximum(abs, bufs.Phi_z),
    )
    dt = ws.sim_params.dt
    exponent = phi_max * dt / 2
    if exponent > _ITP_EXPONENT_LIMIT
        throw(ArgumentError(
            "DDI potential overflow in ITP at step $step: " *
            "max|Φ|=$(round(phi_max, sigdigits=3)), " *
            "exponent=$(round(exponent, digits=1)) > $_ITP_EXPONENT_LIMIT. " *
            "Reduce dt (current=$dt)."
        ))
    end
    nothing
end

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

function _validate_itp_interactions(interactions::InteractionParams, F, dt;
                                    psi=nothing, c_dd::Float64=0.0)
    n_peak = if psi !== nothing
        ndim = ndims(psi) - 1
        Float64(maximum(total_density(psi, ndim)))
    else
        1.0
    end

    max_c = max(abs(interactions.c0), abs(interactions.c1))
    if max_c > 1e-30
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

    if c_dd > 0
        ddi_exponent = c_dd * F * n_peak * dt / 2
        if ddi_exponent > _ITP_EXPONENT_LIMIT
            throw(ArgumentError(
                "DDI c_dd=$c_dd with F=$F, n_peak≈$(round(n_peak, sigdigits=3)) and dt=$dt " *
                "may cause overflow in imaginary time (estimated exponent=$(round(ddi_exponent, digits=1)) " *
                "> $_ITP_EXPONENT_LIMIT). Reduce c_dd or dt."
            ))
        end
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
    target_magnetization::Union{Nothing,Float64}=nothing,
)
    psi0 = if psi_init === nothing
        sys = SpinSystem(atom.F)
        init_psi(grid, sys; state=initial_state)
    else
        copy(psi_init)
    end

    _validate_itp_zeeman(zeeman, atom.F, dt)
    effective_c_dd = (enable_ddi && !isnan(c_dd)) ? c_dd : 0.0
    _validate_itp_interactions(interactions, atom.F, dt; psi=psi0, c_dd=effective_c_dd)

    if adaptive_dt
        return _find_ground_state_adaptive(;
            grid, atom, interactions, zeeman, potential,
            dt, n_steps, tol, psi0, enable_ddi, c_dd, secular_ddi, dt_max, fft_flags,
        )
    end

    use_constrained = target_magnetization !== nothing
    norm_every = use_constrained ? 0 : 1
    sp = SimParams(; dt, n_steps, imaginary_time=true, normalize_every=norm_every,
                   save_every=max(1, n_steps ÷ 10))
    ws = make_workspace(; grid, atom, interactions, zeeman, potential, sim_params=sp,
                        psi_init=psi0, enable_ddi, c_dd, secular_ddi, fft_flags)

    n_comp = ws.spin_matrices.system.n_components
    N_dim = length(grid.config.n_points)

    if use_constrained
        _normalize_psi_constrained!(ws.state.psi, ws.grid, n_comp, N_dim,
                                    target_magnetization, atom.F)
    end

    E_prev = total_energy(ws)
    converged = false
    psi_prev = copy(ws.state.psi)
    final_dE = NaN
    final_dpsi = NaN

    for step in 1:n_steps
        split_step!(ws)
        step <= 10 && _check_itp_overflow(ws, step)
        if use_constrained
            _normalize_psi_constrained!(ws.state.psi, ws.grid, n_comp, N_dim,
                                        target_magnetization, atom.F)
        end
        if step % sp.save_every == 0
            E = total_energy(ws)
            dE = abs(E - E_prev)
            psi_max = maximum(abs, ws.state.psi)
            dpsi = psi_max > 0 ? maximum(abs, ws.state.psi .- psi_prev) / psi_max : 0.0
            final_dE = dE
            final_dpsi = dpsi
            if dE < tol && dpsi < tol
                converged = true
                break
            end
            E_prev = E
            copyto!(psi_prev, ws.state.psi)
        end
    end

    (workspace=ws, converged=converged, energy=total_energy(ws),
     dE=final_dE, dpsi=final_dpsi)
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

    final_dE = NaN
    final_dpsi = NaN

    while total_steps < n_steps
        copyto!(psi_backup, ws.state.psi)

        for i in 1:check_every
            split_step!(ws)
            i == 1 && total_steps == 0 && _check_itp_overflow(ws, 1)
        end
        total_steps += check_every

        E = total_energy(ws)

        if isnan(E) || E > E_prev
            copyto!(ws.state.psi, psi_backup)
            current_dt = max(current_dt * 0.5, 1e-8)
            ws = _rebuild_workspace_with_dt(ws, current_dt)
        else
            dE = abs(E - E_prev)
            psi_max = maximum(abs, ws.state.psi)
            dpsi = psi_max > 0 ? maximum(abs, ws.state.psi .- psi_backup) / psi_max : 0.0
            final_dE = dE
            final_dpsi = dpsi
            if dE < tol && dpsi < tol
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

    (workspace=ws, converged=converged, energy=total_energy(ws),
     dE=final_dE, dpsi=final_dpsi)
end

"""
    find_ground_state_multistart(; initial_states, n_random, seed, kwargs...) → NamedTuple

Try multiple initial states and return the lowest-energy ground state.
All keyword arguments except `initial_states`, `n_random`, and `seed` are
forwarded to `find_ground_state`.

Returns `(workspace, converged, energy, initial_state, all_results)`.
"""
function find_ground_state_multistart(;
    initial_states::Vector{Symbol}=[:polar, :ferromagnetic, :uniform, :antiferromagnetic],
    n_random::Int=3,
    seed::Int=42,
    grid,
    atom,
    interactions,
    kwargs...,
)
    results = NamedTuple[]

    for state in initial_states
        if state == :random
            for i in 1:n_random
                sys = SpinSystem(atom.F)
                psi0 = init_psi(grid, sys; state=:random, seed=seed + i)
                r = find_ground_state(; grid, atom, interactions, psi_init=psi0, kwargs...)
                push!(results, (initial_state=:random, idx=i, workspace=r.workspace,
                                converged=r.converged, energy=r.energy,
                                dE=r.dE, dpsi=r.dpsi))
            end
        else
            r = find_ground_state(; grid, atom, interactions, initial_state=state, kwargs...)
            push!(results, (initial_state=state, idx=0, workspace=r.workspace,
                            converged=r.converged, energy=r.energy,
                            dE=r.dE, dpsi=r.dpsi))
        end
    end

    best = argmin(r -> r.energy, results)
    (workspace=best.workspace, converged=best.converged, energy=best.energy,
     initial_state=best.initial_state, all_results=results)
end

"""
Normalize psi while constraining magnetization ⟨Fz⟩ to target_Mz.

Applies exp(λ·m) weights to each component, then normalizes.
Uses Newton iteration to find λ such that Mz(λ) = target_Mz.
"""
function _normalize_psi_constrained!(psi, grid, n_components, ndim, target_Mz, F)
    dV = cell_volume(grid)
    n_pts = ntuple(d -> size(psi, d), ndim)

    _normalize_psi!(psi, grid, n_components, ndim)

    lambda = 0.0
    for _iter in 1:20
        norms = Vector{Float64}(undef, n_components)
        for c in 1:n_components
            m = F - (c - 1)
            idx = _component_slice(ndim, n_pts, c)
            w = exp(lambda * m)
            norms[c] = sum(abs2, view(psi, idx...)) * dV * w^2
        end
        total = sum(norms)
        total < 1e-30 && break

        Mz = sum((F - (c - 1)) * norms[c] for c in 1:n_components) / total
        abs(Mz - target_Mz) < 1e-12 && break

        dMz = 0.0
        for c in 1:n_components
            m = F - (c - 1)
            dMz += 2 * m * (m - Mz) * norms[c] / total
        end
        abs(dMz) < 1e-30 && break

        lambda -= (Mz - target_Mz) / dMz
        lambda = clamp(lambda, -10.0, 10.0)
    end

    for c in 1:n_components
        m = F - (c - 1)
        w = exp(lambda * m)
        idx = _component_slice(ndim, n_pts, c)
        view(psi, idx...) .*= w
    end

    _normalize_psi!(psi, grid, n_components, ndim)
    nothing
end

"""
    scan_continuation(; param_values, make_interactions, grid, atom, ...) → Vector{NamedTuple}

Sweep a parameter using continuation: use previous ground state as initial guess for
next point. Falls back to multistart search on energy jumps.
"""
function scan_continuation(;
    param_values::AbstractVector{Float64},
    make_interactions::Function,
    grid,
    atom,
    initial_state::Symbol=:polar,
    energy_jump_threshold::Float64=0.1,
    n_steps_continuation::Int=500,
    n_steps_fresh::Int=5000,
    kwargs...,
)
    results = NamedTuple[]
    prev_psi = nothing
    prev_energy = NaN

    sm = spin_matrices(atom.F)

    for (i, val) in enumerate(param_values)
        interactions = make_interactions(val)

        r = if prev_psi !== nothing
            find_ground_state(; grid, atom, interactions,
                              psi_init=copy(prev_psi),
                              n_steps=n_steps_continuation, kwargs...)
        else
            find_ground_state(; grid, atom, interactions,
                              initial_state, n_steps=n_steps_fresh, kwargs...)
        end

        if !isnan(prev_energy) && abs(r.energy - prev_energy) / max(abs(prev_energy), 1e-30) > energy_jump_threshold
            r = find_ground_state_multistart(; grid, atom, interactions,
                                              n_steps=n_steps_fresh, kwargs...)
        end

        ws = r.workspace
        phase_info = classify_phase(ws.state.psi, atom.F, grid, sm)

        push!(results, (
            param=val,
            energy=r.energy,
            converged=r.converged,
            phase=phase_info.phase,
            psi=copy(ws.state.psi),
        ))

        prev_psi = copy(ws.state.psi)
        prev_energy = r.energy
    end

    results
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
