"""
Total energy (approximate, using current wavefunction).

E = E_kin + E_trap + E_Zeeman + E_int(c0) + E_int(c1)
"""
function total_energy(ws::Workspace{N}) where {N}
    psi = ws.state.psi
    grid = ws.grid
    n_comp = ws.spin_matrices.system.n_components
    dV = cell_volume(grid)
    n_pts = ntuple(d -> size(psi, d), N)

    E_kin = _kinetic_energy(psi, grid, ws.fft_plans, ws.state.fft_buf, n_comp, N, n_pts, dV)
    E_trap = _trap_energy(psi, ws.potential_values, n_comp, N, n_pts, dV)
    zee = zeeman_at(ws.zeeman, ws.state.t)
    E_zee = _zeeman_energy(psi, zee, ws.spin_matrices.system, n_comp, N, n_pts, dV)
    E_c0 = _density_interaction_energy(psi, ws.interactions.c0, n_comp, N, n_pts, dV)
    E_c1 = _spin_interaction_energy(psi, ws.spin_matrices, ws.interactions.c1, n_comp, N, n_pts, dV)

    E_ddi = if ws.ddi !== nothing
        _ddi_energy(psi, ws.spin_matrices, ws.ddi, ws.ddi_bufs, n_comp, N, n_pts, dV)
    else
        0.0
    end

    E_lhy = ws.interactions.c_lhy != 0.0 ? _lhy_energy(psi, ws.interactions.c_lhy, n_comp, N, n_pts, dV) : 0.0

    E_tensor = if ws.tensor_cache !== nothing
        _tensor_interaction_energy(psi, ws.tensor_cache, N, n_pts, dV)
    else
        c2 = get_cn(ws.interactions, 2)
        c2 != 0.0 ? _nematic_energy(psi, ws.spin_matrices.system.F, c2, N, n_pts, dV) : 0.0
    end

    E_kin + E_trap + E_zee + E_c0 + E_c1 + E_ddi + E_lhy + E_tensor
end

function _kinetic_energy(psi, grid, plans, fft_buf, n_comp, ndim, n_pts, dV)
    E = 0.0
    for c in 1:n_comp
        idx = _component_slice(ndim, n_pts, c)
        fft_buf .= view(psi, idx...)
        plans.forward * fft_buf
        E += real(sum(grid.k_squared .* abs2.(fft_buf))) * dV / prod(n_pts)
    end
    0.5 * E
end

function _trap_energy(psi, V_trap, n_comp, ndim, n_pts, dV)
    E = 0.0
    for c in 1:n_comp
        idx = _component_slice(ndim, n_pts, c)
        E += sum(V_trap .* abs2.(view(psi, idx...))) * dV
    end
    E
end

function _zeeman_energy(psi, zeeman, sys, n_comp, ndim, n_pts, dV)
    zee = zeeman_energies(zeeman, sys)
    E = 0.0
    for c in 1:n_comp
        idx = _component_slice(ndim, n_pts, c)
        E += zee[c] * sum(abs2, view(psi, idx...)) * dV
    end
    E
end

function _density_interaction_energy(psi, c0, n_comp, ndim, n_pts, dV)
    n = total_density(psi, ndim)
    0.5 * c0 * sum(n .^ 2) * dV
end

function _lhy_energy(psi, c_lhy, n_comp, ndim, n_pts, dV)
    n = total_density(psi, ndim)
    E = 0.0
    @inbounds for I in CartesianIndices(n_pts)
        ni = n[I]
        E += ni * ni * sqrt(ni)
    end
    (2.0 / 5.0) * c_lhy * E * dV
end

function _spin_interaction_energy(psi, sm, c1, n_comp, ndim, n_pts, dV)
    fx, fy, fz = spin_density_vector(psi, sm, ndim)
    0.5 * c1 * sum(fx .^ 2 .+ fy .^ 2 .+ fz .^ 2) * dV
end

function _nematic_energy(psi, F, c2, ndim, n_pts, dV)
    A = singlet_pair_amplitude(psi, F, ndim)
    0.5 * c2 * sum(abs2, A) * dV
end

function _ddi_energy(psi, sm::SpinMatrices{D}, ddi, ddi_bufs, n_comp, ndim, n_pts, dV) where {D}
    _compute_spin_density!(ddi_bufs.Fx_r, ddi_bufs.Fy_r, ddi_bufs.Fz_r, psi, sm, Val(D), ndim, n_pts)
    compute_ddi_potential!(ddi, ddi_bufs)
    E = 0.0
    @inbounds for I in CartesianIndices(n_pts)
        E += ddi_bufs.Phi_x[I] * ddi_bufs.Fx_r[I] +
             ddi_bufs.Phi_y[I] * ddi_bufs.Fy_r[I] +
             ddi_bufs.Phi_z[I] * ddi_bufs.Fz_r[I]
    end
    0.5 * E * dV
end
