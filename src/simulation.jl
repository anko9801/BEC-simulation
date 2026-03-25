function init_psi(grid::Grid{N}, sys::SpinSystem; state::Symbol=:polar) where {N}
    n_pts = grid.config.n_points
    psi = zeros(ComplexF64, n_pts..., sys.n_components)

    sigma = ntuple(d -> grid.config.box_size[d] / 8, N)
    gauss = _gaussian(grid, sigma)

    if state == :polar
        mid = (sys.n_components + 1) ÷ 2
        _set_component!(psi, gauss, N, n_pts, mid)
    elseif state == :ferromagnetic
        _set_component!(psi, gauss, N, n_pts, 1)
    elseif state == :uniform
        for c in 1:sys.n_components
            _set_component!(psi, gauss / sqrt(sys.n_components), N, n_pts, c)
        end
    else
        throw(ArgumentError("Unknown initial state: $state"))
    end

    dV = cell_volume(grid)
    norm = sqrt(sum(abs2, psi) * dV)
    psi ./= norm
    psi
end

function _gaussian(grid::Grid{N}, sigma::NTuple{N,Float64}) where {N}
    g = zeros(Float64, grid.config.n_points)
    @inbounds for I in CartesianIndices(grid.config.n_points)
        s = 0.0
        for d in 1:N
            s += grid.x[d][I[d]]^2 / (2 * sigma[d]^2)
        end
        g[I] = exp(-s)
    end
    g
end

function _set_component!(psi, vals, ndim, n_pts, c)
    idx = _component_slice(ndim, n_pts, c)
    view(psi, idx...) .= vals
end

function make_workspace(;
    grid::Grid{N},
    atom::AtomSpecies,
    interactions::InteractionParams,
    zeeman::Union{ZeemanParams,TimeDependentZeeman}=ZeemanParams(),
    potential::AbstractPotential=NoPotential(),
    sim_params::SimParams,
    psi_init::Union{Nothing,AbstractArray{ComplexF64}}=nothing,
    enable_ddi::Bool=false,
    c_dd::Float64=NaN,
    raman::Union{Nothing,RamanCoupling{N}}=nothing,
    loss::Union{Nothing,LossParams}=nothing,
    fft_flags=FFTW.MEASURE,
) where {N}
    sys = SpinSystem(atom.F)
    sm = spin_matrices(atom.F)

    psi = if psi_init === nothing
        init_psi(grid, sys)
    else
        copy(psi_init)
    end

    fft_buf = zeros(ComplexF64, grid.config.n_points)
    state = SimState{N,typeof(psi)}(psi, fft_buf, 0.0, 0)

    plans = make_fft_plans(grid.config.n_points; flags=fft_flags)
    kinetic_phase = prepare_kinetic_phase(grid, sim_params.dt; imaginary_time=sim_params.imaginary_time)
    V = evaluate_potential(potential, grid)

    ddi = if enable_ddi
        c_dd_val = isnan(c_dd) ? compute_c_dd(atom) : c_dd
        make_ddi_params(grid, atom; c_dd=c_dd_val)
    else
        nothing
    end

    ddi_bufs = if ddi !== nothing
        make_ddi_buffers(grid.config.n_points)
    else
        nothing
    end

    density_buf = zeros(Float64, grid.config.n_points)

    Workspace(
        state, plans, kinetic_phase, V, density_buf, sm, grid, atom, interactions, zeeman, potential, sim_params,
        ddi, ddi_bufs, raman, loss,
    )
end

struct SimulationResult
    times::Vector{Float64}
    energies::Vector{Float64}
    norms::Vector{Float64}
    magnetizations::Vector{Float64}
    psi_snapshots::Vector{Array{ComplexF64}}
end

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

    if adaptive_dt
        return _find_ground_state_adaptive(;
            grid, atom, interactions, zeeman, potential,
            dt, n_steps, tol, psi0, enable_ddi, c_dd, dt_max, fft_flags,
        )
    end

    sp = SimParams(; dt, n_steps, imaginary_time=true, normalize_every=1, save_every=max(1, n_steps ÷ 10))
    ws = make_workspace(; grid, atom, interactions, zeeman, potential, sim_params=sp, psi_init=psi0, enable_ddi, c_dd, fft_flags)

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
    dt, n_steps, tol, psi0, enable_ddi, c_dd, dt_max, fft_flags=FFTW.MEASURE,
)
    current_dt = dt
    check_every = max(1, n_steps ÷ 100)
    psi_current = copy(psi0)
    psi_backup = similar(psi0)

    sp = SimParams(; dt=current_dt, n_steps=check_every, imaginary_time=true,
                   normalize_every=1, save_every=check_every)
    ws = make_workspace(; grid, atom, interactions, zeeman, potential,
                        sim_params=sp, psi_init=psi_current, enable_ddi, c_dd, fft_flags)
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

    Workspace(
        ws.state, ws.fft_plans, kinetic_phase, ws.potential_values, ws.density_buf,
        ws.spin_matrices, ws.grid, ws.atom, ws.interactions,
        ws.zeeman, ws.potential, sp, ws.ddi, ws.ddi_bufs, ws.raman, ws.loss,
    )
end

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

    push!(times, ws.state.t)
    push!(energies, total_energy(ws))
    push!(norms, total_norm(ws.state.psi, ws.grid))
    push!(mags, magnetization(ws.state.psi, ws.grid, sys))
    push!(snapshots, copy(ws.state.psi))

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
            push!(times, ws.state.t)
            push!(energies, total_energy(ws))
            push!(norms, total_norm(ws.state.psi, ws.grid))
            push!(mags, magnetization(ws.state.psi, ws.grid, sys))
            push!(snapshots, copy(ws.state.psi))

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

    # Batched FFT plans: transform spatial dims, batch over component dim
    psi_plan_buf = similar(ws.state.psi)
    dims = ntuple(identity, N)
    batched_fwd = plan_fft!(psi_plan_buf, dims; flags=FFTW.MEASURE)
    batched_inv = plan_ifft!(psi_plan_buf, dims; flags=FFTW.MEASURE)
    kp_bc = reshape(ws.kinetic_phase, size(ws.kinetic_phase)..., 1)

    # Leapfrog: initial half potential step
    _half_potential_step!(ws, dt / 2, n_comp, N, false)

    for step in 1:sp.n_steps
        # Batched kinetic step (all components simultaneously)
        batched_fwd * ws.state.psi
        ws.state.psi .*= kp_bc
        batched_inv * ws.state.psi

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
            push!(times, ws.state.t)
            push!(energies, total_energy(ws))
            push!(norms, total_norm(ws.state.psi, ws.grid))
            push!(mags, magnetization(ws.state.psi, ws.grid, sys))
            push!(snapshots, copy(ws.state.psi))

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

# --- Adaptive time stepping ---

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

    psi_plan_buf = similar(ws.state.psi)
    dims = ntuple(identity, N)
    batched_fwd = plan_fft!(psi_plan_buf, dims; flags=FFTW.MEASURE)
    batched_inv = plan_ifft!(psi_plan_buf, dims; flags=FFTW.MEASURE)
    kp_bc = reshape(ws.kinetic_phase, size(ws.kinetic_phase)..., 1)

    times = Float64[ws.state.t]
    energies = Float64[total_energy(ws)]
    norms = Float64[total_norm(ws.state.psi, ws.grid)]
    mags = Float64[magnetization(ws.state.psi, ws.grid, sys)]
    snapshots = [copy(ws.state.psi)]

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
            _update_kinetic_phase!(ws.kinetic_phase, ws.grid.k_squared, dt_step)
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

        batched_fwd * ws.state.psi
        ws.state.psi .*= kp_bc
        batched_inv * ws.state.psi

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
            E_per_N = E_now / max(nrm_now, 1e-300)
            if length(energies) >= 2
                E_per_N_prev = energies[end] / max(norms[end], 1e-300)
                de_rel = abs(E_per_N - E_per_N_prev) / max(abs(E_per_N_prev), 1e-300)
                if de_rel > 0.01
                    @warn "E/N drift $(round(de_rel*100, digits=2))% between snapshots at t=$(round(ws.state.t, digits=4))"
                end
            end

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
        push!(times, ws.state.t)
        push!(energies, total_energy(ws))
        push!(norms, total_norm(ws.state.psi, ws.grid))
        push!(mags, magnetization(ws.state.psi, ws.grid, sys))
        push!(snapshots, copy(ws.state.psi))
    end

    (result=SimulationResult(times, energies, norms, mags, snapshots),
     n_accepted=n_accepted, n_rejected=n_rejected, final_dt=dt)
end

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

    times = Float64[ws.state.t]
    energies = Float64[total_energy(ws)]
    norms = Float64[total_norm(ws.state.psi, ws.grid)]
    mags = Float64[magnetization(ws.state.psi, ws.grid, sys)]
    snapshots = [copy(ws.state.psi)]

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

        # Strang step for error estimation
        _strang_core!(ws, dt_step, n_comp)
        psi_strang .= ws.state.psi

        # Yoshida step from same initial state
        ws.state.psi .= psi_old
        _yoshida_core!(ws, dt_step, n_comp)

        # Embedded error: difference between 4th and 2nd order
        err = _wavefunction_l2_change(ws.state.psi, psi_strang)

        if !is_clamped && err > adaptive.tol && dt_step > adaptive.dt_min * 1.01
            ws.state.psi .= psi_old
            factor = max(0.3, 0.9 * (adaptive.tol / err)^0.2)
            dt = max(dt_step * factor, adaptive.dt_min)
            n_rejected += 1
            continue
        end

        # Loss after accepted step
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
            E_per_N = E_now / max(nrm_now, 1e-300)
            if length(energies) >= 2
                E_per_N_prev = energies[end] / max(norms[end], 1e-300)
                de_rel = abs(E_per_N - E_per_N_prev) / max(abs(E_per_N_prev), 1e-300)
                if de_rel > 0.01
                    @warn "E/N drift $(round(de_rel*100, digits=2))% between snapshots at t=$(round(ws.state.t, digits=4))"
                end
            end

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
        push!(times, ws.state.t)
        push!(energies, total_energy(ws))
        push!(norms, total_norm(ws.state.psi, ws.grid))
        push!(mags, magnetization(ws.state.psi, ws.grid, sys))
        push!(snapshots, copy(ws.state.psi))
    end

    (result=SimulationResult(times, energies, norms, mags, snapshots),
     n_accepted=n_accepted, n_rejected=n_rejected, final_dt=dt)
end
