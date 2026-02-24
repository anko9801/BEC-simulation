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
    if ndim == 1
        for i in 1:n_pts[1]
            spinor = SVector{n_comp,ComplexF64}(ntuple(c -> psi[i, c], n_comp))
            fx[i] = real(dot(spinor, sm.Fx * spinor))
            fy[i] = real(dot(spinor, sm.Fy * spinor))
            fz[i] = real(dot(spinor, sm.Fz * spinor))
        end
    elseif ndim == 2
        for j in 1:n_pts[2], i in 1:n_pts[1]
            spinor = SVector{n_comp,ComplexF64}(ntuple(c -> psi[i, j, c], n_comp))
            fx[i, j] = real(dot(spinor, sm.Fx * spinor))
            fy[i, j] = real(dot(spinor, sm.Fy * spinor))
            fz[i, j] = real(dot(spinor, sm.Fz * spinor))
        end
    end
end

"""
Total energy (approximate, using current wavefunction).

E = E_kin + E_trap + E_Zeeman + E_int(c0) + E_int(c1)
"""
function total_energy(ws::Workspace{N}) where {N}
    psi = ws.state.psi
    grid = ws.grid
    nc = ws.spin_matrices.system.n_components
    ndim = N
    dV = cell_volume(grid)
    n_pts = ntuple(d -> size(psi, d), ndim)

    E_kin = _kinetic_energy(psi, grid, ws.fft_plans, ws.state.fft_buf, nc, ndim, n_pts, dV)
    E_trap = _trap_energy(psi, ws.potential_values, nc, ndim, n_pts, dV)
    E_zee = _zeeman_energy(psi, ws.zeeman, ws.spin_matrices.system, nc, ndim, n_pts, dV)
    E_c0 = _density_interaction_energy(psi, ws.interactions.c0, nc, ndim, n_pts, dV)
    E_c1 = _spin_interaction_energy(psi, ws.spin_matrices, ws.interactions.c1, nc, ndim, n_pts, dV)

    E_kin + E_trap + E_zee + E_c0 + E_c1
end

function _kinetic_energy(psi, grid, plans, fft_buf, nc, ndim, n_pts, dV)
    E = 0.0
    for c in 1:nc
        idx = _component_slice(ndim, n_pts, c)
        fft_buf .= view(psi, idx...)
        plans.forward * fft_buf
        E += real(sum(grid.k_squared .* abs2.(fft_buf))) * dV / prod(n_pts)
    end
    0.5 * E
end

function _trap_energy(psi, V_trap, nc, ndim, n_pts, dV)
    E = 0.0
    for c in 1:nc
        idx = _component_slice(ndim, n_pts, c)
        E += sum(V_trap .* abs2.(view(psi, idx...))) * dV
    end
    E
end

function _zeeman_energy(psi, zeeman, sys, nc, ndim, n_pts, dV)
    zee = zeeman_energies(zeeman, sys)
    E = 0.0
    for c in 1:nc
        idx = _component_slice(ndim, n_pts, c)
        E += zee[c] * sum(abs2, view(psi, idx...)) * dV
    end
    E
end

function _density_interaction_energy(psi, c0, nc, ndim, n_pts, dV)
    n = total_density(psi, ndim)
    0.5 * c0 * sum(n .^ 2) * dV
end

function _spin_interaction_energy(psi, sm, c1, nc, ndim, n_pts, dV)
    fx, fy, fz = spin_density_vector(psi, sm, ndim)
    0.5 * c1 * sum(fx .^ 2 .+ fy .^ 2 .+ fz .^ 2) * dV
end
