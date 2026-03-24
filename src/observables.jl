function total_density(psi::AbstractArray{ComplexF64}, ndim::Int)
    n_comp = size(psi, ndim + 1)
    n_pts = ntuple(d -> size(psi, d), ndim)
    idx1 = _component_slice(ndim, n_pts, 1)
    n = abs2.(view(psi, idx1...))
    for c in 2:n_comp
        idx = _component_slice(ndim, n_pts, c)
        n .+= abs2.(view(psi, idx...))
    end
    n
end

function component_density(psi::AbstractArray{ComplexF64}, ndim::Int, c::Int)
    n_pts = ntuple(d -> size(psi, d), ndim)
    idx = _component_slice(ndim, n_pts, c)
    abs2.(view(psi, idx...))
end

function total_norm(psi::AbstractArray{ComplexF64}, grid::Grid{N}) where {N}
    dV = cell_volume(grid)
    n = total_density(psi, N)
    sum(n) * dV
end

"""
Magnetization ⟨Fz⟩ = Σ_m m |ψ_m|² integrated over space.
"""
function magnetization(psi::AbstractArray{ComplexF64}, grid::Grid{N}, sys::SpinSystem) where {N}
    dV = cell_volume(grid)
    Mz = 0.0
    n_pts = ntuple(d -> size(psi, d), N)
    for (c, m) in enumerate(sys.m_values)
        idx = _component_slice(N, n_pts, c)
        Mz += m * sum(abs2, view(psi, idx...)) * dV
    end
    Mz
end

"""
Local spin density vector (Fx, Fy, Fz) at each spatial point.
Returns a tuple of 3 arrays.
"""
function spin_density_vector(psi::AbstractArray{ComplexF64}, sm::SpinMatrices, ndim::Int)
    n_comp = sm.system.n_components
    n_pts = ntuple(d -> size(psi, d), ndim)

    fx = zeros(Float64, n_pts)
    fy = zeros(Float64, n_pts)
    fz = zeros(Float64, n_pts)

    _compute_spin_density!(fx, fy, fz, psi, sm, n_comp, ndim, n_pts)

    (fx, fy, fz)
end

function _compute_spin_density!(fx, fy, fz, psi, sm, n_comp, ndim, n_pts)
    @inbounds for I in CartesianIndices(n_pts)
        spinor = _get_spinor(psi, I, n_comp)
        fx[I] = real(dot(spinor, sm.Fx * spinor))
        fy[I] = real(dot(spinor, sm.Fy * spinor))
        fz[I] = real(dot(spinor, sm.Fz * spinor))
    end
end

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
        _ddi_energy(psi, ws.spin_matrices, ws.ddi, ws.ddi_bufs, ws.fft_plans, n_comp, N, n_pts, dV)
    else
        0.0
    end

    E_kin + E_trap + E_zee + E_c0 + E_c1 + E_ddi
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

function _spin_interaction_energy(psi, sm, c1, n_comp, ndim, n_pts, dV)
    fx, fy, fz = spin_density_vector(psi, sm, ndim)
    0.5 * c1 * sum(fx .^ 2 .+ fy .^ 2 .+ fz .^ 2) * dV
end

function _ddi_energy(psi, sm, ddi, ddi_bufs, plans, n_comp, ndim, n_pts, dV)
    _compute_spin_density!(ddi_bufs.Fx_r, ddi_bufs.Fy_r, ddi_bufs.Fz_r, psi, sm, n_comp, ndim, n_pts)
    compute_ddi_potential!(ddi, ddi_bufs, plans)
    E = 0.0
    @inbounds for I in CartesianIndices(n_pts)
        E += real(ddi_bufs.Phi_x[I]) * ddi_bufs.Fx_r[I] +
             real(ddi_bufs.Phi_y[I]) * ddi_bufs.Fy_r[I] +
             real(ddi_bufs.Phi_z[I]) * ddi_bufs.Fz_r[I]
    end
    0.5 * E * dV
end

"""
Probability current density j(r) = Σ_c Im(ψ_c* ∇ψ_c).
Returns NTuple{N, Array{Float64,N}} of current components.
"""
function probability_current(psi::AbstractArray{ComplexF64}, grid::Grid{N}, plans::FFTPlans) where {N}
    n_comp = size(psi, N + 1)
    n_pts = ntuple(d -> size(psi, d), N)

    j = ntuple(_ -> zeros(Float64, n_pts), N)
    psi_k = zeros(ComplexF64, n_pts)
    dpsi = zeros(ComplexF64, n_pts)

    for c in 1:n_comp
        idx = _component_slice(N, n_pts, c)
        psi_c = view(psi, idx...)

        psi_k .= psi_c
        plans.forward * psi_k

        for d in 1:N
            @inbounds for I in CartesianIndices(n_pts)
                dpsi[I] = im * grid.k[d][I[d]] * psi_k[I]
            end
            plans.inverse * dpsi
            @inbounds for I in CartesianIndices(n_pts)
                j[d][I] += imag(conj(psi_c[I]) * dpsi[I])
            end
        end
    end

    j
end

"""
Orbital angular momentum ⟨Lz⟩ = ∫ Σ_c ψ_c* (-i)(x ∂_y - y ∂_x) ψ_c d^N r.
Returns 0.0 for 1D grids.
"""
function orbital_angular_momentum(psi::AbstractArray{ComplexF64}, grid::Grid{N}, plans::FFTPlans) where {N}
    N >= 2 || return 0.0

    n_comp = size(psi, N + 1)
    n_pts = ntuple(d -> size(psi, d), N)
    dV = cell_volume(grid)

    psi_k = zeros(ComplexF64, n_pts)
    dpsi_x = zeros(ComplexF64, n_pts)
    dpsi_y = zeros(ComplexF64, n_pts)

    Lz = 0.0

    for c in 1:n_comp
        idx = _component_slice(N, n_pts, c)
        psi_c = view(psi, idx...)

        psi_k .= psi_c
        plans.forward * psi_k

        @inbounds for I in CartesianIndices(n_pts)
            dpsi_x[I] = im * grid.k[1][I[1]] * psi_k[I]
            dpsi_y[I] = im * grid.k[2][I[2]] * psi_k[I]
        end
        plans.inverse * dpsi_x
        plans.inverse * dpsi_y

        @inbounds for I in CartesianIndices(n_pts)
            x = grid.x[1][I[1]]
            y = grid.x[2][I[2]]
            Lz += real(conj(psi_c[I]) * (-im) * (x * dpsi_y[I] - y * dpsi_x[I])) * dV
        end
    end

    Lz
end
