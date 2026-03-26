"""
Perform one Strang-split time step.

Nested splitting:
1. Half potential step:
   a. Quarter diagonal potential
   b. Half spin-mixing
   c. Quarter diagonal potential (recompute density)
2. Full kinetic step
3. Half potential step (symmetric)

For imaginary time: replace i with 1 in exponentials, optionally renormalize.
"""
function split_step!(ws::Workspace{N}) where {N}
    dt = ws.sim_params.dt
    it = ws.sim_params.imaginary_time
    n_comp = ws.spin_matrices.system.n_components

    @timeit_debug TIMER "half_potential" _half_potential_step!(ws, dt / 2, n_comp, N, it)

    @timeit_debug TIMER "kinetic" apply_kinetic_step_batched!(ws.state.psi, ws.batched_kinetic)

    @timeit_debug TIMER "half_potential" _half_potential_step!(ws, dt / 2, n_comp, N, it)

    if !it && ws.loss !== nothing
        @timeit_debug TIMER "loss" apply_loss_step!(ws.state.psi, ws.loss, ws.spin_matrices.system.F, dt, n_comp, N, ws.density_buf)
    end

    ws.state.t += it ? 0.0 : dt
    ws.state.step += 1

    if it && ws.sim_params.normalize_every > 0
        if ws.state.step % ws.sim_params.normalize_every == 0
            _normalize_psi!(ws.state.psi, ws.grid, n_comp, N)
        end
    end

    nothing
end

function _half_potential_step!(ws::Workspace{N}, dt_half, n_comp, ndim, imaginary_time) where {N}
    zee = zeeman_at(ws.zeeman, ws.state.t)
    zeeman_diag = zeeman_diagonal(zee, ws.spin_matrices)

    @timeit_debug TIMER "diagonal" _diagonal_step_svec!(
        Val(N), ws.state.psi, ws.potential_values, zeeman_diag,
        ws.interactions.c0, ws.interactions.c_lhy, dt_half / 2, ws.density_buf, imaginary_time,
    )

    @timeit_debug TIMER "spin_mixing" apply_spin_mixing_step!(
        ws.state.psi, ws.spin_matrices, ws.interactions.c1,
        dt_half, ndim;
        imaginary_time,
    )

    if ws.ddi !== nothing
        if ws.ddi_padded !== nothing
            @timeit_debug TIMER "ddi" apply_ddi_step!(
                ws.state.psi, ws.spin_matrices, ws.ddi, ws.ddi_bufs,
                ws.fft_plans, dt_half, ndim, ws.ddi_padded;
                imaginary_time,
            )
        else
            @timeit_debug TIMER "ddi" apply_ddi_step!(
                ws.state.psi, ws.spin_matrices, ws.ddi, ws.ddi_bufs,
                ws.fft_plans, dt_half, ndim;
                imaginary_time,
            )
        end
    end

    if ws.raman !== nothing
        @timeit_debug TIMER "raman" apply_raman_step!(
            ws.state.psi, ws.spin_matrices, ws.raman,
            ws.grid, dt_half;
            imaginary_time,
        )
    end

    @timeit_debug TIMER "diagonal" _diagonal_step_svec!(
        Val(N), ws.state.psi, ws.potential_values, zeeman_diag,
        ws.interactions.c0, ws.interactions.c_lhy, dt_half / 2, ws.density_buf, imaginary_time,
    )
end

"""
Yoshida 4th-order triple-jump coefficients.
S₄(dt) = S₂(w₁·dt) ∘ S₂(w₀·dt) ∘ S₂(w₁·dt)  with w₀ + 2w₁ = 1.
"""
const _YOSHIDA_W1 = 1.0 / (2.0 - 2.0^(1 / 3))
const _YOSHIDA_W0 = 1.0 - 2.0 * _YOSHIDA_W1

"""
One Strang step with explicit dt (no sim_params dependency).
V(dt/2) K(dt) V(dt/2).
"""
function _strang_core!(ws::Workspace{N}, dt::Float64, n_comp::Int) where {N}
    _half_potential_step!(ws, dt / 2, n_comp, N, false)
    _update_batched_kinetic_phase!(ws.batched_kinetic, ws.grid.k_squared, dt)
    apply_kinetic_step_batched!(ws.state.psi, ws.batched_kinetic)
    _half_potential_step!(ws, dt / 2, n_comp, N, false)
    nothing
end

"""
One 4th-order Yoshida step with merged boundary V-steps: 4V + 3K stages.

w₀ < 0 causes reverse evolution in the middle substep.
All operators (kinetic, diagonal, DDI, spin-mixing) are unitary and time-reversible,
so negative dt is valid.
"""
function _yoshida_core!(ws::Workspace{N}, dt::Float64, n_comp::Int) where {N}
    w1 = _YOSHIDA_W1
    w0 = _YOSHIDA_W0
    wm = (w1 + w0) / 2

    _half_potential_step!(ws, w1 * dt / 2, n_comp, N, false)

    _update_batched_kinetic_phase!(ws.batched_kinetic, ws.grid.k_squared, w1 * dt)
    apply_kinetic_step_batched!(ws.state.psi, ws.batched_kinetic)

    _half_potential_step!(ws, wm * dt, n_comp, N, false)

    _update_batched_kinetic_phase!(ws.batched_kinetic, ws.grid.k_squared, w0 * dt)
    apply_kinetic_step_batched!(ws.state.psi, ws.batched_kinetic)

    _half_potential_step!(ws, wm * dt, n_comp, N, false)

    _update_batched_kinetic_phase!(ws.batched_kinetic, ws.grid.k_squared, w1 * dt)
    apply_kinetic_step_batched!(ws.state.psi, ws.batched_kinetic)

    _half_potential_step!(ws, w1 * dt / 2, n_comp, N, false)
    nothing
end

function _normalize_psi!(psi, grid, n_components, ndim)
    dV = cell_volume(grid)
    norm_sq = 0.0
    n_pts = ntuple(d -> size(psi, d), ndim)
    for c in 1:n_components
        idx = _component_slice(ndim, n_pts, c)
        norm_sq += sum(abs2, view(psi, idx...)) * dV
    end
    psi ./= sqrt(norm_sq)
    nothing
end
