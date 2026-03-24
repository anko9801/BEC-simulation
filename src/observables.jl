function total_density(psi::AbstractArray{ComplexF64}, ndim::Int)
    n_comp = size(psi, ndim + 1)
    n_pts = ntuple(d -> size(psi, d), ndim)
    _total_density(psi, n_comp, ndim, n_pts)
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

"""
Superfluid velocity v_d = j_d / n at each spatial point.
Returns `NTuple{N, Array{Float64,N}}`.
Points with density below `density_cutoff` are set to zero.
"""
function superfluid_velocity(psi::AbstractArray{ComplexF64}, grid::Grid{N}, plans::FFTPlans;
                             density_cutoff::Float64=1e-10) where {N}
    j = probability_current(psi, grid, plans)
    n = total_density(psi, N)
    n_pts = ntuple(d -> size(psi, d), N)

    v = ntuple(_ -> zeros(Float64, n_pts), N)
    @inbounds for I in CartesianIndices(n_pts)
        if n[I] > density_cutoff
            inv_n = 1.0 / n[I]
            for d in 1:N
                v[d][I] = j[d][I] * inv_n
            end
        end
    end
    v
end

"""
Total angular momentum J_z = L_z + S_z.
L_z = `orbital_angular_momentum`, S_z = `magnetization`.
"""
function total_angular_momentum(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                                plans::FFTPlans, sys::SpinSystem) where {N}
    Lz = orbital_angular_momentum(psi, grid, plans)
    Sz = magnetization(psi, grid, sys)
    Lz + Sz
end

"""
Superfluid vorticity ω = ∇ × v_s.
- 1D: returns 0.0
- 2D: returns `Array{Float64,2}` (scalar ω_z = ∂v_y/∂x - ∂v_x/∂y)
- 3D: returns `NTuple{3, Array{Float64,3}}` (full curl)
"""
function superfluid_vorticity(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                              plans::FFTPlans;
                              density_cutoff::Float64=1e-10) where {N}
    N >= 2 || return 0.0
    v = superfluid_velocity(psi, grid, plans; density_cutoff)
    if N == 2
        dvydx = _fft_partial_derivative(v[2], grid, plans, 1)
        dvxdy = _fft_partial_derivative(v[1], grid, plans, 2)
        return dvydx .- dvxdy
    else
        dvzdx = _fft_partial_derivative(v[3], grid, plans, 1)
        dvxdz = _fft_partial_derivative(v[1], grid, plans, 3)
        dvxdy = _fft_partial_derivative(v[1], grid, plans, 2)
        dvydx = _fft_partial_derivative(v[2], grid, plans, 1)
        dvzdy = _fft_partial_derivative(v[3], grid, plans, 2)
        dvydz = _fft_partial_derivative(v[2], grid, plans, 3)
        return (dvzdy .- dvydz, dvxdz .- dvzdx, dvydx .- dvxdy)
    end
end

function _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, i, j)
    omega = zeros(Float64, n_pts)
    @inbounds for I in CartesianIndices(n_pts)
        cross_x = dsy[i][I] * dsz[j][I] - dsz[i][I] * dsy[j][I]
        cross_y = dsz[i][I] * dsx[j][I] - dsx[i][I] * dsz[j][I]
        cross_z = dsx[i][I] * dsy[j][I] - dsy[i][I] * dsx[j][I]
        omega[I] = sx[I] * cross_x + sy[I] * cross_y + sz[I] * cross_z
    end
    omega
end

"""
Gauge-invariant Berry curvature from the Mermin-Ho relation:
  Ω = ŝ · (∂_i ŝ × ∂_j ŝ)
where ŝ = ⟨F⟩/|⟨F⟩| is the local spin direction.

- 1D: returns `zeros(Float64, n_pts)`
- 2D: returns `Array{Float64,2}` (Ω_z)
- 3D: returns `NTuple{3, Array{Float64,3}}` (pseudo-vector)
"""
function berry_curvature(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                         plans::FFTPlans, sm::SpinMatrices;
                         density_cutoff::Float64=1e-10) where {N}
    n_pts = ntuple(d -> size(psi, d), N)
    N >= 2 || return zeros(Float64, n_pts)

    fx, fy, fz = spin_density_vector(psi, sm, N)
    n = total_density(psi, N)

    sx = zeros(Float64, n_pts)
    sy = zeros(Float64, n_pts)
    sz = zeros(Float64, n_pts)

    @inbounds for I in CartesianIndices(n_pts)
        if n[I] > density_cutoff
            inv_n = 1.0 / n[I]
            sx[I] = fx[I] * inv_n
            sy[I] = fy[I] * inv_n
            sz[I] = fz[I] * inv_n
        end
    end

    dsx = _fft_gradient(sx, grid, plans)
    dsy = _fft_gradient(sy, grid, plans)
    dsz = _fft_gradient(sz, grid, plans)

    if N == 2
        return _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 1, 2)
    else
        omega_x = _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 2, 3)
        omega_y = _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 3, 1)
        omega_z = _berry_curvature_component(sx, sy, sz, dsx, dsy, dsz, n_pts, 1, 2)
        return (omega_x, omega_y, omega_z)
    end
end

"""
Topological skyrmion charge Q = (1/4πF) ∫ Ω d²r.
2D only; returns 0.0 for other dimensions.
Delegates to `berry_curvature`.
"""
function spin_texture_charge(psi::AbstractArray{ComplexF64}, grid::Grid{N},
                             plans::FFTPlans, sm::SpinMatrices;
                             density_cutoff::Float64=1e-10) where {N}
    N == 2 || return 0.0
    omega = berry_curvature(psi, grid, plans, sm; density_cutoff)
    dV = cell_volume(grid)
    sum(omega) * dV / (4π * sm.system.F)
end
