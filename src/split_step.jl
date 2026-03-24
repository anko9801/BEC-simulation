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

    _half_potential_step!(ws, dt / 2, n_comp, N, it)

    apply_kinetic_step!(
        ws.state.psi, ws.state.fft_buf, ws.kinetic_phase,
        ws.fft_plans, n_comp, N,
    )

    _half_potential_step!(ws, dt / 2, n_comp, N, it)

    if !it && ws.loss !== nothing
        apply_loss_step!(ws.state.psi, ws.loss, ws.spin_matrices.system.F, dt, n_comp, N)
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
    zeeman_diag = zeeman_diagonal(zee, ws.spin_matrices.system)

    apply_diagonal_potential_step!(
        ws.state.psi, ws.potential_values, zeeman_diag,
        ws.interactions.c0, dt_half / 2, n_comp, ndim, ws.density_buf;
        imaginary_time,
    )

    apply_spin_mixing_step!(
        ws.state.psi, ws.spin_matrices, ws.interactions.c1,
        dt_half, ndim;
        imaginary_time,
    )

    if ws.ddi !== nothing
        apply_ddi_step!(
            ws.state.psi, ws.spin_matrices, ws.ddi, ws.ddi_bufs,
            ws.fft_plans, dt_half, ndim;
            imaginary_time,
        )
    end

    if ws.raman !== nothing
        apply_raman_step!(
            ws.state.psi, ws.spin_matrices, ws.raman,
            ws.grid, dt_half;
            imaginary_time,
        )
    end

    apply_diagonal_potential_step!(
        ws.state.psi, ws.potential_values, zeeman_diag,
        ws.interactions.c0, dt_half / 2, n_comp, ndim, ws.density_buf;
        imaginary_time,
    )
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
